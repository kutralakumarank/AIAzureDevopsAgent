"""Versioned ingestion extending Rag's splitter, embeddings and FAISS implementation."""
from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import json
import sqlite3

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(value):
    parser = _Text()
    parser.feed(str(value or ''))
    return ' '.join(parser.parts)


@dataclass(frozen=True)
class Scope:
    organization_id: str
    project_id: str
    sources: frozenset[str]

    @property
    def key(self):
        return sha256(f'{self.organization_id}\0{self.project_id}'.encode()).hexdigest()


def story_document(story, scope, source_type='azure_devops'):
    fields = story['fields']
    title = fields.get('System.Title', '')
    content = '\n'.join(f'{key}: {plain(value)}' for key, value in fields.items())
    parents = [r['url'].rsplit('/', 1)[-1] for r in story.get('relations', [])
               if r.get('rel') == 'System.LinkTypes.Hierarchy-Reverse']
    return Document(page_content=content, metadata={
        'organization_id': scope.organization_id, 'project_id': scope.project_id,
        'source_type': source_type, 'source_document_id': str(story['id']),
        'work_item_id': str(story['id']), 'work_item_type': fields.get('System.WorkItemType', ''),
        'title': title, 'parent_id': parents[0] if parents else '',
        'updated_at': fields.get('System.ChangedDate', ''), 'version': str(story.get('rev', 1)),
        'source_link': f'{scope.organization_id}/{scope.project_id}/_workitems/edit/{story["id"]}',
        'evidence_type': 'historical_information',
    })


def confluence_documents(connector, scope, space_id=None):
    for page in connector.pages(space_id):
        yield Document(page_content=plain(page.get('body', {}).get('storage', {}).get('value', '')),
                       metadata={'organization_id': scope.organization_id, 'project_id': scope.project_id,
                                 'source_type': 'confluence', 'source_document_id': connector.http.base + ':' + str(page['id']),
                                 'ingest_partition': connector.http.base,
                                 'title': page['title'], 'version': str(page.get('version', {}).get('number', 1)),
                                 'updated_at': page.get('version', {}).get('createdAt', ''),
                                 'source_link': connector.http.base + '/wiki' + page.get('_links', {}).get('webui', ''),
                                 'evidence_type': 'historical_information'})


def file_documents(path, scope):
    """Lazy optional parsers; text/code, CSV, PDF, Word and Excel."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        from pypdf import PdfReader
        text = '\n'.join(page.extract_text() or '' for page in PdfReader(path).pages)
    elif suffix == '.docx':
        from docx import Document as WordDocument
        doc = WordDocument(path)
        text = '\n'.join([p.text for p in doc.paragraphs] +
                         [' | '.join(c.text for c in row.cells) for table in doc.tables for row in table.rows])
    elif suffix == '.xlsx':
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            text = '\n'.join(f'{sheet.title}: ' + ' | '.join('' if v is None else str(v) for v in row)
                             for sheet in workbook for row in sheet.values)
        finally:
            workbook.close()
    elif suffix in {'.txt', '.md', '.csv', '.json', '.ts', '.js', '.py', '.feature', '.html'}:
        text = path.read_text(encoding='utf-8')
    else:
        raise ValueError('Supported: PDF, DOCX, XLSX, CSV, text, Markdown and source code')
    return [Document(page_content=text, metadata={
        'organization_id': scope.organization_id, 'project_id': scope.project_id,
        'source_type': 'files', 'source_document_id': path.name, 'title': path.name,
        'version': sha256(text.encode()).hexdigest(), 'source_link': '',
        'evidence_type': 'historical_information'})]


class KnowledgeIndex:
    def __init__(self, root, scope, rag, embedding_version='google/embeddinggemma-300m:v1'):
        self.scope, self.rag, self.embedding_version = scope, rag, embedding_version
        directory = Path(root) / scope.key
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'knowledge.sqlite3'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS documents (source TEXT, id TEXT, digest TEXT, payload TEXT, PRIMARY KEY(source,id))')
            db.execute('CREATE TABLE IF NOT EXISTS syncs (source TEXT PRIMARY KEY, status TEXT, count INTEGER, updated_at TEXT)')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _check(self, metadata):
        if (metadata.get('organization_id'), metadata.get('project_id')) != (self.scope.organization_id, self.scope.project_id):
            raise PermissionError('Document is outside the authorized project')
        if metadata.get('source_type') not in self.scope.sources:
            raise PermissionError('Knowledge source access denied')

    def sync(self, documents, source, *, full_snapshot=False, partition=None, force=False):
        if source not in self.scope.sources:
            raise PermissionError('Knowledge source access denied')
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO syncs VALUES (?, 'running', 0, datetime('now'))", (source,))
        try:
            return self._sync(documents, source, full_snapshot=full_snapshot, partition=partition, force=force)
        except Exception:
            with self.connect() as db:
                db.execute("UPDATE syncs SET status='failed', updated_at=datetime('now') WHERE source=?", (source,))
            raise

    def _sync(self, documents, source, *, full_snapshot=False, partition=None, force=False):
        # Materialize before mutations: a failed paginated fetch cannot delete prior data.
        documents = list(documents)
        ids, changed, unchanged = set(), 0, 0
        prepared = []
        with self.connect() as db:
            existing = {}
            for identifier, digest, payload in db.execute('SELECT id,digest,payload FROM documents WHERE source=?', (source,)):
                saved = json.loads(payload)
                chunks = saved['chunks']
                metadata = saved.get('metadata', chunks[0]['metadata'] if chunks else {})
                if partition is None or metadata.get('ingest_partition') == partition:
                    existing[identifier] = digest
        for doc in documents:
            self._check(doc.metadata)
            if doc.metadata['source_type'] != source:
                raise ValueError('Mixed sources in a sync batch')
            if partition is not None and doc.metadata.get('ingest_partition') != partition:
                raise ValueError('Document is outside the ingestion partition')
            identifier = str(doc.metadata['source_document_id'])
            if identifier in ids:
                raise ValueError('Duplicate source document ID')
            ids.add(identifier)
            digest = sha256(json.dumps([doc.page_content, doc.metadata, self.embedding_version], sort_keys=True).encode()).hexdigest()
            if existing.get(identifier) == digest and not force:
                unchanged += 1
                continue
            chunks = self.rag.split_documents([doc])
            vectors = self.rag.create_embeddings().embed_documents([c.page_content for c in chunks]) if chunks else []
            payload = {'embedding_version': self.embedding_version, 'metadata': doc.metadata, 'chunks': [
                {'content': c.page_content, 'metadata': c.metadata, 'vector': v} for c, v in zip(chunks, vectors, strict=True)]}
            prepared.append((source, identifier, digest, json.dumps(payload)))
            changed += 1
        removed = set(existing) - ids if full_snapshot else set()
        with self.connect() as db:
            db.executemany('INSERT OR REPLACE INTO documents VALUES (?,?,?,?)', prepared)
            db.executemany('DELETE FROM documents WHERE source=? AND id=?', [(source, i) for i in removed])
            db.execute("INSERT OR REPLACE INTO syncs VALUES (?, 'complete', ?, datetime('now'))", (source, len(ids)))
        return {'changed': changed, 'unchanged': unchanged, 'deleted': len(removed)}

    def status(self):
        with self.connect() as db:
            return [dict(zip(('source', 'status', 'count', 'updated_at'), row))
                    for row in db.execute('SELECT * FROM syncs') if row[0] in self.scope.sources]

    def retrieve(self, story, k=8):
        fields = story['fields']
        queries = [plain(fields.get('System.Title', '')),
                   plain(fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', '')),
                   plain(fields.get('System.Description', ''))]
        chunks = []
        with self.connect() as db:
            for source, payload in db.execute('SELECT source,payload FROM documents'):
                if source not in self.scope.sources:
                    continue
                saved = json.loads(payload)
                if saved['embedding_version'] != self.embedding_version:
                    raise ValueError('Embedding version changed; synchronize all sources before retrieval')
                for chunk in saved['chunks']:
                    self._check(chunk['metadata'])
                    chunks.append(chunk)
        if not chunks:
            return []
        # Build only from authorized sources BEFORE search. Persisted numeric vectors avoid pickle.
        store = FAISS.from_embeddings([(c['content'], c['vector']) for c in chunks],
                                      self.rag.create_embeddings(), metadatas=[c['metadata'] for c in chunks])
        ranked = {}
        import re
        terms = set(re.findall(r'\w+', ' '.join(queries).lower()))
        for query in filter(None, queries):
            for rank, (doc, distance) in enumerate(store.similarity_search_with_score(query, k=min(k * 3, len(chunks)))):
                key = sha256((json.dumps(doc.metadata, sort_keys=True) + doc.page_content).encode()).hexdigest()
                lexical = len(terms & set(re.findall(r'\w+', doc.page_content.lower()))) / max(1, len(terms))
                entry = ranked.setdefault(key, {'reference_id': key, 'metadata': doc.metadata,
                                                'content': doc.page_content, 'distance': float(distance),
                                                'score': 0.0, 'reason': 'Semantic story/criteria match plus lexical overlap'})
                entry['score'] += 1 / (60 + rank) + lexical / 100
        return sorted(ranked.values(), key=lambda item: item['score'], reverse=True)[:k]

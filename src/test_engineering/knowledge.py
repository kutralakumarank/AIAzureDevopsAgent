"""Versioned ingestion extending Rag's splitter, embeddings and FAISS implementation."""
from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import json
import os
import sqlite3
import tempfile

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


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
        # A cache is scoped to precisely the permitted sources, before search.
        sources_key = sha256(json.dumps(sorted(scope.sources)).encode()).hexdigest()[:24]
        self.faiss_path = directory / f'knowledge-{sources_key}.faiss'
        self.manifest_path = directory / f'knowledge-{sources_key}.manifest.json'
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

    def _digest(self, doc):
        return sha256(json.dumps([doc.page_content, doc.metadata, self.embedding_version], sort_keys=True).encode()).hexdigest()

    def is_current(self, doc):
        """Check a document's version without making an embedding request."""
        self._check(doc.metadata)
        with self.connect() as db:
            row = db.execute('SELECT digest FROM documents WHERE source=? AND id=?',
                             (doc.metadata['source_type'], str(doc.metadata['source_document_id']))).fetchone()
        return bool(row and row[0] == self._digest(doc))

    def signature(self):
        """Identify the current authorized snapshot without calling the embedding service."""
        with self.connect() as db:
            _, signature = self._snapshot(db)
        return signature

    def has_other_documents(self, source, identifier):
        """Check whether any authorized source exists besides the current document."""
        with self.connect() as db:
            return any(item_source in self.scope.sources and (item_source, item_id) != (source, str(identifier))
                       for item_source, item_id in db.execute('SELECT source,id FROM documents'))

    def _snapshot(self, db):
        chunks, versions = [], []
        for source, identifier, payload in db.execute('SELECT source,id,payload FROM documents ORDER BY source,id'):
            if source not in self.scope.sources:
                continue
            saved = json.loads(payload)
            if saved['embedding_version'] != self.embedding_version:
                raise ValueError('Embedding version changed; synchronize all sources before retrieval')
            metadata = saved.get('metadata')
            if metadata:
                self._check(metadata)
            versions.append((source, identifier, sha256(payload.encode()).hexdigest()))
            for chunk in saved['chunks']:
                self._check(chunk['metadata'])
                chunks.append(chunk)
        signature = sha256(json.dumps([self.embedding_version, sorted(self.scope.sources), versions]).encode()).hexdigest()
        return chunks, signature

    @staticmethod
    def _atomic_write(path, data):
        descriptor, temporary = tempfile.mkstemp(prefix='.faiss-', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as output:
                output.write(data)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _native_index(self, chunks, signature):
        """Load or rebuild an atomic native FAISS cache from the SQLite snapshot."""
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            raw = self.faiss_path.read_bytes()
            if (manifest.get('signature') == signature and manifest.get('count') == len(chunks)
                    and manifest.get('sha256') == sha256(raw).hexdigest()):
                native = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8).copy())
                expected_dimension = len(chunks[0]['vector']) if chunks else 1
                if native.ntotal == len(chunks) and native.d == expected_dimension:
                    return native
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError):
            pass  # A missing, stale or corrupt cache can be rebuilt from stored vectors.
        try:
            vectors = np.asarray([chunk['vector'] for chunk in chunks], dtype=np.float32)
        except (TypeError, ValueError):
            raise ValueError('Stored embeddings are invalid; re-index the affected knowledge sources') from None
        if chunks:
            if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
                raise ValueError('Stored embeddings are invalid; re-index the affected knowledge sources')
            native = faiss.IndexFlatL2(vectors.shape[1])
            native.add(vectors)
        else:
            native = faiss.IndexFlatL2(1)
        raw = faiss.serialize_index(native).tobytes()
        manifest = {'signature': signature, 'count': len(chunks), 'sha256': sha256(raw).hexdigest()}
        self._atomic_write(self.faiss_path, raw)
        self._atomic_write(self.manifest_path, json.dumps(manifest).encode())
        return native

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
        pending = []
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
            digest = self._digest(doc)
            if existing.get(identifier) == digest and not force:
                unchanged += 1
                continue
            chunks = self.rag.split_documents([doc])
            pending.append((identifier, digest, doc.metadata, chunks))
            changed += 1
        all_chunks = [chunk for _, _, _, chunks in pending for chunk in chunks]
        all_vectors = []
        if all_chunks:
            embeddings = self.rag.create_embeddings()
            for offset in range(0, len(all_chunks), 16):
                batch = all_chunks[offset:offset + 16]
                vectors = embeddings.embed_documents([chunk.page_content for chunk in batch])
                if len(vectors) != len(batch):
                    raise ValueError('Embedding service returned an incomplete batch; retry the sync')
                all_vectors.extend(vectors)
        prepared, position = [], 0
        for identifier, digest, metadata, chunks in pending:
            vectors = all_vectors[position:position + len(chunks)]
            position += len(chunks)
            payload = {'embedding_version': self.embedding_version, 'metadata': metadata, 'chunks': [
                {'content': chunk.page_content, 'metadata': chunk.metadata, 'vector': vector}
                for chunk, vector in zip(chunks, vectors, strict=True)]}
            prepared.append((source, identifier, digest, json.dumps(payload)))
        removed = set(existing) - ids if full_snapshot else set()
        with self.connect() as db:
            db.executemany('INSERT OR REPLACE INTO documents VALUES (?,?,?,?)', prepared)
            db.executemany('DELETE FROM documents WHERE source=? AND id=?', [(source, i) for i in removed])
            # Write a real FAISS index from persisted vectors. If an embedding model
            # changed, allow sources to be re-indexed one by one before rebuilding.
            try:
                chunks, signature = self._snapshot(db)
            except ValueError as error:
                if str(error) != 'Embedding version changed; synchronize all sources before retrieval':
                    raise
            else:
                self._native_index(chunks, signature)
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
        with self.connect() as db:
            chunks, signature = self._snapshot(db)
        native = self._native_index(chunks, signature)
        if not chunks:
            return []
        # The index contains only authorized sources; document text remains in SQLite.
        documents = {str(i): Document(page_content=c['content'], metadata=c['metadata']) for i, c in enumerate(chunks)}
        store = FAISS(embedding_function=self.rag.create_embeddings(), index=native,
                      docstore=InMemoryDocstore(documents), index_to_docstore_id={i: str(i) for i in range(len(chunks))})
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

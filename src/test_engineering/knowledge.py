"""Versioned ingestion extending Rag's splitter, embeddings and FAISS implementation."""
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import json
import os
import tempfile
import uuid

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
    """Persistent FAISS vectors with JSON document metadata and atomic updates."""

    def __init__(self, root, scope, rag, embedding_version='google/embeddinggemma-300m:v1'):
        self.scope, self.rag, self.embedding_version = scope, rag, embedding_version
        self.directory = Path(root) / scope.key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / 'knowledge.json'
        sources_key = sha256(json.dumps(sorted(scope.sources)).encode()).hexdigest()[:24]
        self.faiss_path = self.directory / f'knowledge-{sources_key}.faiss'
        self.manifest_path = self.directory / f'knowledge-{sources_key}.manifest.json'
        self.lock_path = self.directory / 'knowledge.lock'

    @contextmanager
    def _lock(self):
        # Serialize read-modify-write transactions across local application processes.
        with self.lock_path.open('a+b') as lock:
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == 'nt':
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write(path, data):
        descriptor, temporary = tempfile.mkstemp(prefix='.knowledge-', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _load(self):
        if not self.state_path.exists():
            return {'format': 1, 'documents': {}, 'syncs': {}}
        try:
            state = json.loads(self.state_path.read_text(encoding='utf-8'))
            if state['format'] != 1 or not isinstance(state['documents'], dict) or not isinstance(state['syncs'], dict):
                raise ValueError
            return state
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError('Knowledge metadata is invalid. Restore a backup before continuing.') from None

    def _save(self, state):
        self._atomic_write(self.state_path, json.dumps(state, ensure_ascii=False, sort_keys=True).encode('utf-8'))

    def _check_project(self, metadata):
        if (metadata.get('organization_id'), metadata.get('project_id')) != (self.scope.organization_id, self.scope.project_id):
            raise PermissionError('Document is outside the authorized project')

    def _check(self, metadata):
        self._check_project(metadata)
        if metadata.get('source_type') not in self.scope.sources:
            raise PermissionError('Knowledge source access denied')

    @staticmethod
    def _key(source, identifier):
        return json.dumps([source, str(identifier)], ensure_ascii=False)

    def _digest(self, doc):
        return sha256(json.dumps([doc.page_content, doc.metadata, self.embedding_version], sort_keys=True).encode()).hexdigest()

    def _write_vectors(self, vectors):
        if not vectors:
            return None
        try:
            array = np.asarray(vectors, dtype=np.float32)
            if array.ndim != 2 or not array.shape[1] or not np.isfinite(array).all():
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError('Stored embeddings are invalid; re-index the affected knowledge sources') from None
        index = faiss.IndexFlatL2(array.shape[1])
        index.add(array)
        raw = faiss.serialize_index(index).tobytes()
        name = f'vectors-{uuid.uuid4().hex}.faiss'
        self._atomic_write(self.directory / name, raw)
        return {'file': name, 'sha256': sha256(raw).hexdigest(), 'count': len(vectors)}

    def _read_vectors(self, record):
        chunks = record['chunks']
        shard = record.get('vectors')
        if not chunks and shard is None:
            return []
        try:
            name = shard['file']
            if not isinstance(name, str) or not name.startswith('vectors-') or Path(name).name != name or not name.endswith('.faiss'):
                raise ValueError
            raw = (self.directory / name).read_bytes()
            if sha256(raw).hexdigest() != shard['sha256']:
                raise ValueError
            index = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8).copy())
            if index.ntotal != len(chunks) or index.ntotal != shard['count']:
                raise ValueError
            return index.reconstruct_n(0, index.ntotal).tolist()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, AttributeError):
            raise ValueError('FAISS knowledge data is damaged. Restore a backup or re-index the affected sources.') from None

    def is_current(self, doc):
        self._check(doc.metadata)
        record = self._load()['documents'].get(self._key(doc.metadata['source_type'], doc.metadata['source_document_id']))
        return bool(record and record['digest'] == self._digest(doc))

    def _snapshot(self, state):
        records = []
        for key, record in sorted(state['documents'].items()):
            if record['source'] not in self.scope.sources:
                continue
            self._check(record['metadata'])
            if record['embedding_version'] != self.embedding_version:
                raise ValueError('Embedding version changed; synchronize all sources before retrieval')
            for chunk in record['chunks']:
                self._check(chunk['metadata'])
            records.append((key, record))
        versions = [(key, record['digest'], record.get('vectors')) for key, record in records]
        signature = sha256(json.dumps([self.embedding_version, sorted(self.scope.sources), versions], sort_keys=True).encode()).hexdigest()
        return records, signature

    def signature(self):
        return self._snapshot(self._load())[1]

    def has_other_documents(self, source, identifier):
        return any(record['source'] in self.scope.sources and (record['source'], record['id']) != (source, str(identifier))
                   for record in self._load()['documents'].values())

    def _native_index(self, records, signature):
        chunks = [chunk for _, record in records for chunk in record['chunks']]
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            raw = self.faiss_path.read_bytes()
            if (manifest.get('signature') == signature and manifest.get('count') == len(chunks)
                    and manifest.get('sha256') == sha256(raw).hexdigest()):
                native = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8).copy())
                if native.ntotal == len(chunks):
                    return native, chunks
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError):
            pass
        vectors = [vector for _, record in records for vector in self._read_vectors(record)]
        if vectors:
            try:
                array = np.asarray(vectors, dtype=np.float32)
                if array.ndim != 2 or not array.shape[1] or not np.isfinite(array).all():
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError('Stored embeddings have incompatible dimensions; re-index all sources') from None
            native = faiss.IndexFlatL2(array.shape[1])
            native.add(array)
        else:
            native = faiss.IndexFlatL2(1)
        raw = faiss.serialize_index(native).tobytes()
        self._atomic_write(self.faiss_path, raw)
        self._atomic_write(self.manifest_path, json.dumps({'signature': signature, 'count': len(chunks),
                                                          'sha256': sha256(raw).hexdigest()}).encode())
        return native, chunks

    @staticmethod
    def _status(source, status, count):
        return {'source': source, 'status': status, 'count': count,
                'updated_at': datetime.now(timezone.utc).isoformat()}

    def sync(self, documents, source, *, full_snapshot=False, partition=None, force=False):
        if source not in self.scope.sources:
            raise PermissionError('Knowledge source access denied')
        with self._lock():
            state = self._load()
            try:
                documents = list(documents)  # Finish fetching before replacing anything.
                existing = {record['id']: record for record in state['documents'].values()
                            if record['source'] == source and
                            (partition is None or record['metadata'].get('ingest_partition') == partition)}
                ids, pending, unchanged = set(), [], 0
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
                    if identifier in existing and existing[identifier]['digest'] == digest and not force:
                        unchanged += 1
                        continue
                    pending.append((identifier, digest, doc.metadata, self.rag.split_documents([doc])))
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
                position = 0
                for identifier, digest, metadata, chunks in pending:
                    vectors = all_vectors[position:position + len(chunks)]
                    position += len(chunks)
                    state['documents'][self._key(source, identifier)] = {
                        'source': source, 'id': identifier, 'digest': digest,
                        'embedding_version': self.embedding_version, 'metadata': metadata,
                        'chunks': [{'content': chunk.page_content, 'metadata': chunk.metadata} for chunk in chunks],
                        'vectors': self._write_vectors(vectors)}
                removed = set(existing) - ids if full_snapshot else set()
                for identifier in removed:
                    del state['documents'][self._key(source, identifier)]
                state['syncs'][source] = self._status(source, 'complete', len(ids))
                # Build the scoped index before publishing metadata. Mixed embedding
                # versions remain in separate FAISS shards until all are resynced.
                try:
                    records, signature = self._snapshot(state)
                except ValueError as error:
                    if str(error) != 'Embedding version changed; synchronize all sources before retrieval':
                        raise
                else:
                    self._native_index(records, signature)
                self._save(state)
                return {'changed': len(pending), 'unchanged': unchanged, 'deleted': len(removed)}
            except Exception:
                # Keep the last committed documents even when fetching or embedding fails.
                previous = self._load()
                previous['syncs'][source] = self._status(source, 'failed', 0)
                self._save(previous)
                raise

    def status(self):
        return [value for key, value in self._load()['syncs'].items() if key in self.scope.sources]

    def retrieve(self, story, k=8):
        fields = story['fields']
        queries = [plain(fields.get('System.Title', '')),
                   plain(fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', '')),
                   plain(fields.get('System.Description', ''))]
        with self._lock():
            records, signature = self._snapshot(self._load())
            native, chunks = self._native_index(records, signature)
        if not chunks:
            return []
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

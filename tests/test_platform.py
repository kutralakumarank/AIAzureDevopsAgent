import copy
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess
from unittest.mock import patch

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from src.RAG.Rag import Rag
from src.integrations.azure_devops import AzureDevOps
from src.integrations.confluence import Confluence
from src.integrations.http import IntegrationError
from src.test_engineering.knowledge import KnowledgeIndex, Scope, story_document, file_documents
from src.test_engineering.workflow import WorkflowStore, validate_draft, criteria_for, generate_tests, regenerate_selected
from src.test_engineering.publishing import Publisher, approve_and_publish, steps_xml
from src.test_engineering.models import TestDraft
from src.automation.workspace import AutomationWorkspace, safe_file
from src.agents.framework_evolution import build_diff, create_review_branch
from src.LLMS.provider_errors import is_groq_tool_failure, provider_error_message
from src.LLMS.groqllm import GroqLLM


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        value = text.lower()
        return [float(value.count('customer')), float(value.count('required')), float(value.count('bug')), 1.0]


class ProviderTests(unittest.TestCase):
    def test_ollama_discovery_and_local_only_connection(self):
        from src.LLMS.ollamallm import list_ollama_models, local_url, ollama_llm
        response = httpx.Response(200, json={'models': [{'name': 'z:latest'}, {'name': 'a:latest'}, {'name': 'a:latest'}]},
                                  request=httpx.Request('GET', 'http://127.0.0.1:11434/api/tags'))
        with patch('src.LLMS.ollamallm.httpx.get', return_value=response) as get:
            self.assertEqual(list_ollama_models('http://127.0.0.1:11434'), ['a:latest', 'z:latest'])
            self.assertFalse(get.call_args.kwargs['trust_env'])
        for url in ('https://example.com', 'http://localhost:11434/path', 'http://user:pass@localhost:11434'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                local_url(url)
        llm = ollama_llm('a:latest', 'http://127.0.0.1:11434')
        self.assertEqual(llm.model, 'a:latest')
        self.assertEqual(llm.base_url, 'http://127.0.0.1:11434')
        with self.assertRaises(ValueError):
            ollama_llm('', 'http://127.0.0.1:11434')

    def test_ollama_structured_generation_against_local_service(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from src.LLMS.ollamallm import ollama_llm
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(data)
                result = {'model': data['model'], 'created_at': '2026-01-01T00:00:00Z',
                          'message': {'role': 'assistant', 'content': json.dumps(draft())},
                          'done': True, 'done_reason': 'stop'}
                content = (json.dumps(result) + '\n').encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            llm = ollama_llm('local-model', f'http://127.0.0.1:{server.server_port}')
            result = generate_tests(llm, story(), [])
            self.assertEqual(result.test_cases[0].story_id, 123)
            self.assertEqual(requests[0]['model'], 'local-model')
            self.assertIsInstance(requests[0]['format'], dict)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_groq_reasoning_parameters_match_selected_model(self):
        for model, expected in [('openai/gpt-oss-120b', None), ('llama-3.1-8b-instant', None),
                                ('qwen/qwen3.8-27b', 'hidden')]:
            with self.subTest(model=model), patch('src.LLMS.groqllm.ChatGroq') as client:
                GroqLLM({'GROQ_API_KEY': 'test-key', 'selected_groq_model': model}).get_llm_model()
                options = client.call_args.kwargs
                self.assertEqual(options.get('reasoning_format'), expected)
                self.assertEqual(options['model'], model)

    def test_provider_errors_are_actionable_without_raw_messages(self):
        from groq import BadRequestError
        from huggingface_hub.errors import HfHubHTTPError
        request = httpx.Request('POST', 'https://example.invalid')
        response = httpx.Response(402, request=request)
        error = HfHubHTTPError('private-token-and-document', response=response)
        message = provider_error_message(error)
        self.assertIn('HTTP 402', message)
        self.assertIn('credits and billing', message)
        self.assertNotIn('private-token-and-document', message)
        response = httpx.Response(400, request=request)
        error = BadRequestError('private-token-and-document', response=response,
                                body={'code': 'tool_use_failed', 'failed_generation': 'private-token-and-document'})
        message = provider_error_message(error)
        self.assertIn('structured response', message)
        self.assertNotIn('private-token-and-document', message)

    def test_failed_tool_call_retries_json_and_validates_result(self):
        from groq import BadRequestError
        from langchain_core.runnables import RunnableLambda
        request = httpx.Request('POST', 'https://example.invalid')
        error = BadRequestError('private output', response=httpx.Response(400, request=request),
                                body={'code': 'tool_use_failed', 'failed_generation': 'private output'})
        class Model:
            def __init__(self):
                self.methods = []
            def with_structured_output(self, schema, **kwargs):
                self.methods.append(kwargs.get('method', 'function_calling'))
                if len(self.methods) == 1:
                    def fail(_):
                        raise error
                    return RunnableLambda(fail)
                def reply(messages):
                    self.assert_schema_present = 'JSON schema' in messages[0].content
                    return schema.model_validate(draft())
                return RunnableLambda(reply)
        model = Model()
        result = generate_tests(model, story(), [])
        self.assertEqual(model.methods, ['function_calling', 'json_mode'])
        self.assertTrue(model.assert_schema_present)
        self.assertEqual(result.test_cases[0].story_id, 123)

    def test_other_groq_bad_requests_are_not_retried(self):
        from groq import BadRequestError
        from langchain_core.runnables import RunnableLambda
        request = httpx.Request('POST', 'https://example.invalid')
        error = BadRequestError('private output', response=httpx.Response(400, request=request),
                                body={'code': 'model_decommissioned', 'failed_generation': 'private output'})
        class Model:
            calls = 0
            def with_structured_output(self, schema, **kwargs):
                self.calls += 1
                def fail(_):
                    raise error
                return RunnableLambda(fail)
        model = Model()
        with self.assertRaises(BadRequestError):
            generate_tests(model, story(), [])
        self.assertEqual(model.calls, 1)

    def test_documented_tool_failure_without_code_is_recognized(self):
        from groq import BadRequestError
        request = httpx.Request('POST', 'https://example.invalid')
        error = BadRequestError('private output', response=httpx.Response(400, request=request),
                                body={'error': {'type': 'invalid_request_error', 'failed_generation': {'reason': 'bad JSON'}}})
        self.assertTrue(is_groq_tool_failure(error))
        self.assertNotIn('private output', provider_error_message(error))


def story():
    return {'id': 123, 'rev': 2, 'fields': {'System.Title': 'Create customer', 'System.TeamProject': 'Example',
            'System.Description': 'Create customer with required name',
            'Microsoft.VSTS.Common.AcceptanceCriteria': '<ul><li>Name is required</li><li>Save customer</li></ul>'}}


def draft():
    return {'test_cases': [{'title': 'Verify required name', 'description': 'Validate name and saving', 'story_id': 123,
        'preconditions': ['Signed in'], 'steps': [{'step_number': 1, 'action': 'Save customer <empty>',
         'test_data': 'name=&', 'expected_result': 'Required name error'}], 'priority': 1, 'category': 'Functional',
        'acceptance_criteria_reference': ['AC1', 'AC2'], 'rag_source_references': [], 'automation_feasibility': 'High'}],
        'gaps': [], 'ambiguities': [], 'assumptions': []}


class ConnectorTests(unittest.TestCase):
    def test_connection_uses_preview_version_and_projects_keep_stable_version(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path.endswith('/connectionData'):
                if request.url.params.get('api-version') != '7.1-preview.1':
                    return httpx.Response(400, json={'typeKey': 'VssInvalidPreviewVersionException', 'message': 'Preview required'})
                return httpx.Response(200, json={'authenticatedUser': {'id': 'user'}})
            self.assertEqual(request.url.params['api-version'], '7.1')
            return httpx.Response(200, json={'value': []})
        api = AzureDevOps('https://dev.azure.com/example/', SecretStr('secret'), transport=httpx.MockTransport(handler))
        try:
            self.assertEqual(api.validate()['id'], 'user')
            self.assertEqual(api.projects(), [])
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0].url.path, '/example/_apis/connectionData')
        finally:
            api.close()

    def test_connection_preview_tracks_configured_version(self):
        def handler(request):
            self.assertEqual(request.url.params['api-version'], '7.0-preview.1')
            return httpx.Response(200, json={'authenticatedUser': {'id': 'user'}})
        api = AzureDevOps('https://dev.azure.com/example', SecretStr('secret'), ' 7.0 ', httpx.MockTransport(handler))
        try:
            self.assertEqual(api.validate()['id'], 'user')
        finally:
            api.close()

    def test_400_reports_stage_and_known_code_without_echoing_remote_message(self):
        def handler(request):
            return httpx.Response(400, json={'typeKey': 'VssInvalidPreviewVersionException', 'message': 'PAT-SECRET'})
        api = AzureDevOps('https://dev.azure.com/example', SecretStr('PAT-SECRET'), transport=httpx.MockTransport(handler))
        try:
            with self.assertRaises(IntegrationError) as identity_error:
                api.validate()
            self.assertIn('identity validation', str(identity_error.exception))
            self.assertIn('preview API version', str(identity_error.exception))
            self.assertNotIn('PAT-SECRET', str(identity_error.exception))
            with self.assertRaises(IntegrationError) as project_error:
                api.projects()
            self.assertIn('project discovery', str(project_error.exception))
            self.assertNotIn('PAT-SECRET', str(project_error.exception))
        finally:
            api.close()

    def test_auth_pagination_and_secrets(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path.endswith('connectionData'):
                return httpx.Response(200, json={'authenticatedUser': {'id': 'user'}})
            if 'continuationToken' not in request.url.params:
                return httpx.Response(200, json={'value': [{'id': 'a'}]}, headers={'x-ms-continuationtoken': 'next'})
            return httpx.Response(200, json={'value': [{'id': 'b'}]})
        api = AzureDevOps('https://dev.azure.com/example', SecretStr('sensitive-pat'), transport=httpx.MockTransport(handler))
        self.assertEqual(api.validate()['id'], 'user')
        self.assertEqual([p['id'] for p in api.projects()], ['a', 'b'])
        self.assertTrue(all('sensitive-pat' not in str(r.url) for r in requests))
        self.assertTrue(all('authorization' in r.headers for r in requests))
        self.assertNotIn('sensitive-pat', repr(api))
        api.close()

    def test_remote_error_body_not_exposed(self):
        api = AzureDevOps('https://dev.azure.com/example', SecretStr('PAT-SECRET'),
                          transport=httpx.MockTransport(lambda _: httpx.Response(403, text='PAT-SECRET')))
        with self.assertRaises(IntegrationError) as error:
            api.projects()
        self.assertNotIn('PAT-SECRET', str(error.exception))
        api.close()

    def test_wiql_escaping_and_batching(self):
        bodies = []
        def handler(request):
            data = json.loads(request.content)
            bodies.append(data)
            if request.url.path.endswith('wiql'):
                return httpx.Response(200, json={'workItems': [{'id': 1}]})
            return httpx.Response(200, json={'value': [{'id': i} for i in data['ids']]})
        api = AzureDevOps('https://dev.azure.com/example', SecretStr('secret'), transport=httpx.MockTransport(handler))
        api.query_ids("Project's", "Project's\\Sprint", ['User Story'])
        self.assertIn("'Project''s'", bodies[0]['query'])
        self.assertEqual(len(api.work_items('p', list(range(401)))), 401)
        self.assertEqual([len(b['ids']) for b in bodies[1:]], [200, 200, 1])
        api.close()

    def test_wrong_project_story_is_rejected(self):
        def handler(request):
            if request.method == 'POST':
                return httpx.Response(200, json={'value': [story()]})
            return httpx.Response(200, json={'name': 'Other'})
        api = AzureDevOps('https://dev.azure.com/example', SecretStr('secret'), transport=httpx.MockTransport(handler))
        with self.assertRaises(IntegrationError):
            api.story('other-id', 123)
        api.close()

    def test_confluence_pagination_blocks_credential_redirect(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json={'results': [{'id': '1'}], '_links': {'next': 'https://evil.invalid/wiki/api/v2/pages'}})
        client = Confluence('https://example.atlassian.net', 'user', SecretStr('secret'), httpx.MockTransport(handler))
        with self.assertRaises(IntegrationError):
            list(client.pages())
        self.assertEqual(len(calls), 1)
        client.close()

    def test_confluence_cursor_keeps_same_origin(self):
        calls = []
        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(200, json={'results': [{'id': str(len(calls))}],
                '_links': {'next': '/wiki/api/v2/pages?cursor=next'} if len(calls) == 1 else {}})
        client = Confluence('https://example.atlassian.net/wiki', 'user', SecretStr('secret'), httpx.MockTransport(handler))
        self.assertEqual(len(list(client.pages('space'))), 2)
        self.assertIn('cursor=next', calls[1])
        client.close()


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.scope = Scope('https://dev.azure.com/example', 'p', frozenset({'azure_devops', 'files'}))
        self.embeddings = LocalEmbeddings()
        self.rag = Rag(embeddings=self.embeddings)
        self.index = KnowledgeIndex(self.temp.name, self.scope, self.rag)

    def tearDown(self):
        self.temp.cleanup()

    def test_incremental_versions_deletions_and_resume(self):
        doc = story_document(story(), self.scope)
        self.assertEqual(self.index.sync([doc], 'azure_devops')['changed'], 1)
        embedded = self.embeddings.calls
        self.assertEqual(self.index.sync([doc], 'azure_devops')['unchanged'], 1)
        self.assertEqual(self.embeddings.calls, embedded)
        self.assertEqual(self.index.sync([doc], 'azure_devops', force=True)['changed'], 1)
        embedded = self.embeddings.calls
        resumed = KnowledgeIndex(self.temp.name, self.scope, self.rag)
        results = resumed.retrieve(story())
        self.assertTrue(results)
        self.assertEqual(results[0]['metadata']['work_item_id'], '123')
        self.assertEqual(self.embeddings.calls, embedded)
        doc.metadata['version'] = '3'
        self.assertEqual(resumed.sync([doc], 'azure_devops')['changed'], 1)
        self.assertEqual(resumed.sync([], 'azure_devops', full_snapshot=True)['deleted'], 1)
        self.assertEqual(resumed.retrieve(story()), [])

    def test_faiss_persists_stories_and_documents_and_recovers_from_corruption(self):
        import faiss
        first_story = story_document(story(), self.scope)
        self.index.sync([first_story], 'azure_devops')
        self.assertTrue(self.index.faiss_path.is_file())
        self.assertEqual(faiss.read_index(str(self.index.faiss_path)).ntotal, 1)

        file_path = Path(self.temp.name) / 'requirements.txt'
        file_path.write_text('Customer name is required for all customers.', encoding='utf-8')
        document = file_documents(file_path, self.scope)[0]
        self.index.sync([document], 'files')
        self.assertTrue(self.index.is_current(first_story))
        self.assertTrue(self.index.is_current(document))
        self.assertEqual(faiss.read_index(str(self.index.faiss_path)).ntotal, 2)

        changed_story = story()
        changed_story['rev'] = 3
        changed_story['fields']['System.Description'] = 'Updated customer requirements'
        updated = story_document(changed_story, self.scope)
        self.assertFalse(self.index.is_current(updated))
        previous_calls = self.embeddings.calls
        self.index.sync([updated], 'azure_devops')
        self.assertEqual(self.embeddings.calls, previous_calls + 1)
        self.assertTrue(self.index.is_current(document))
        self.assertEqual(faiss.read_index(str(self.index.faiss_path)).ntotal, 2)

        resumed = KnowledgeIndex(self.temp.name, self.scope, self.rag)
        with patch('src.test_engineering.knowledge.faiss.IndexFlatL2', side_effect=AssertionError('Index rebuilt unnecessarily')):
            self.assertEqual(len(resumed.retrieve(changed_story)), 2)
        resumed.faiss_path.write_bytes(b'invalid cache')
        self.assertEqual(len(resumed.retrieve(changed_story)), 2)
        self.assertEqual(faiss.read_index(str(resumed.faiss_path)).ntotal, 2)

        restricted = Scope(self.scope.organization_id, self.scope.project_id, frozenset({'files'}))
        restricted_index = KnowledgeIndex(self.temp.name, restricted, self.rag)
        self.assertEqual(len(restricted_index.retrieve(changed_story)), 1)
        self.assertEqual(faiss.read_index(str(restricted_index.faiss_path)).ntotal, 1)

    def test_new_documents_are_embedded_in_batches_and_unchanged_are_skipped(self):
        class RecordingEmbeddings(LocalEmbeddings):
            def __init__(self):
                super().__init__()
                self.batches = []
            def embed_documents(self, texts):
                self.batches.append(len(texts))
                return super().embed_documents(texts)
        embeddings = RecordingEmbeddings()
        index = KnowledgeIndex(self.temp.name, self.scope, Rag(embeddings=embeddings))
        docs = []
        for identifier in range(100, 117):
            item = story()
            item['id'] = identifier
            docs.append(story_document(item, self.scope))
        result = index.sync(docs, 'azure_devops')
        self.assertEqual(result['changed'], 17)
        self.assertEqual(embeddings.batches, [16, 1])
        self.assertEqual(index.sync(docs, 'azure_devops')['unchanged'], 17)
        self.assertEqual(embeddings.batches, [16, 1])

    def test_failed_embedding_batch_preserves_existing_documents_and_faiss(self):
        existing = story_document(story(), self.scope)
        self.index.sync([existing], 'azure_devops')
        original_index = self.index.faiss_path.read_bytes()
        class FailingEmbeddings(LocalEmbeddings):
            def __init__(self):
                super().__init__()
                self.batches = 0
            def embed_documents(self, texts):
                self.batches += 1
                if self.batches == 2:
                    raise RuntimeError('Embedding service unavailable')
                return super().embed_documents(texts)
        failing = KnowledgeIndex(self.temp.name, self.scope, Rag(embeddings=FailingEmbeddings()))
        docs = []
        for identifier in range(100, 117):
            item = story()
            item['id'] = identifier
            docs.append(story_document(item, self.scope))
        with self.assertRaises(RuntimeError):
            failing.sync(docs, 'azure_devops', full_snapshot=True)
        self.assertTrue(self.index.is_current(existing))
        self.assertFalse(self.index.is_current(docs[0]))
        self.assertEqual(self.index.faiss_path.read_bytes(), original_index)
        self.assertEqual(self.index.status()[0]['status'], 'failed')

    def test_project_and_source_isolation_before_search(self):
        doc = story_document(story(), self.scope)
        doc.metadata['project_id'] = 'another'
        with self.assertRaises(PermissionError):
            self.index.sync([doc], 'azure_devops')
        doc.metadata['project_id'] = 'p'
        self.index.sync([doc], 'azure_devops')
        restricted = Scope(self.scope.organization_id, 'p', frozenset({'files'}))
        self.assertEqual(KnowledgeIndex(self.temp.name, restricted, self.rag).retrieve(story()), [])
        other = Scope(self.scope.organization_id, 'other', self.scope.sources)
        self.assertEqual(KnowledgeIndex(self.temp.name, other, self.rag).retrieve(story()), [])

    def test_failed_snapshot_preserves_index(self):
        doc = story_document(story(), self.scope)
        self.index.sync([doc], 'azure_devops')
        def broken():
            yield doc
            raise RuntimeError('fetch failed')
        with self.assertRaises(RuntimeError):
            self.index.sync(broken(), 'azure_devops', full_snapshot=True)
        self.assertTrue(self.index.retrieve(story()))

    def test_embedding_version_change_requires_sync(self):
        self.index.sync([story_document(story(), self.scope)], 'azure_devops')
        other = KnowledgeIndex(self.temp.name, self.scope, self.rag, embedding_version='next')
        with self.assertRaises(ValueError):
            other.retrieve(story())

    def test_text_loader_and_shared_splitter(self):
        path = Path(self.temp.name) / 'requirements.md'
        path.write_text('Customer ' * 500, encoding='utf-8')
        docs = file_documents(path, self.scope)
        chunks = self.rag.split_documents(docs)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.page_content) <= 1000 for c in chunks))
        self.assertEqual(chunks[0].metadata['source_type'], 'files')

    def test_full_snapshot_is_partitioned_and_failure_status_visible(self):
        docs = []
        for identifier, partition in [('1', 'site-a'), ('2', 'site-b')]:
            doc = story_document(story(), self.scope)
            doc.metadata.update(source_document_id=identifier, ingest_partition=partition)
            docs.append(doc)
        self.index.sync(docs, 'azure_devops')
        self.assertEqual(self.index.sync([], 'azure_devops', full_snapshot=True, partition='site-a')['deleted'], 1)
        self.assertTrue(self.index.retrieve(story()))
        def broken():
            raise RuntimeError('service unavailable')
            yield
        with self.assertRaises(RuntimeError):
            self.index.sync(broken(), 'azure_devops', full_snapshot=True)
        self.assertEqual(self.index.status()[0]['status'], 'failed')

    def test_optional_word_excel_pdf_parsers(self):
        try:
            from docx import Document as WordDocument
            from openpyxl import Workbook
            from pypdf import PdfWriter
        except ImportError:
            self.skipTest('Install the documents extra to validate optional parsers')
        root = Path(self.temp.name)
        word = WordDocument()
        word.add_paragraph('Customer business requirements')
        word.add_table(rows=1, cols=1).cell(0, 0).text = 'Name required'
        word.save(root / 'requirements.docx')
        self.assertIn('Name required', file_documents(root / 'requirements.docx', self.scope)[0].page_content)
        workbook = Workbook()
        workbook.active.append(['Boundary', 0, False])
        workbook.save(root / 'boundaries.xlsx')
        workbook.close()
        text = file_documents(root / 'boundaries.xlsx', self.scope)[0].page_content
        self.assertIn('0 | False', text)
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(root / 'empty.pdf')
        self.assertEqual(file_documents(root / 'empty.pdf', self.scope)[0].metadata['source_type'], 'files')


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.scope = Scope('https://dev.azure.com/example', 'p', frozenset({'azure_devops'}))
        self.store = WorkflowStore(self.temp.name, self.scope)

    def tearDown(self):
        self.temp.cleanup()

    def test_revision_approval_and_resume(self):
        identifier = self.store.save(story(), [], draft())
        current = self.store.load(identifier)
        with self.assertRaises(PermissionError):
            self.store.approve(identifier, 1, current['digest'])
        self.store.approve(identifier, 1, current['digest'], explicit=True)
        self.assertTrue(self.store.load(identifier)['approved'])
        changed = draft()
        changed['test_cases'][0]['title'] = 'Edited title'
        self.store.save(story(), [], changed, identifier)
        self.assertFalse(WorkflowStore(self.temp.name, self.scope).load(identifier)['approved'])
        with self.assertRaises(ValueError):
            self.store.approve(identifier, 1, current['digest'], explicit=True)
        self.assertEqual(len(self.store.history(identifier)), 2)

    def test_one_confirmation_saves_approves_and_publishes_displayed_cases(self):
        identifier = self.store.save(story(), [], draft())
        current = self.store.load(identifier)
        updated = copy.deepcopy(draft())
        updated['test_cases'][0]['title'] = 'Edited case title'
        class FakePublisher:
            def __init__(self, store):
                self.store, self.calls = store, []
            def publish(self, workflow_id, plan_id, suite_id, **kwargs):
                current = self.store.load(workflow_id)
                self.calls.append((workflow_id, plan_id, suite_id, kwargs, current))
                return [{'case_id': 101, 'title': current['draft']['test_cases'][0]['title'], 'status': 'published'}]
        publisher = FakePublisher(self.store)
        with self.assertRaises(PermissionError):
            approve_and_publish(publisher, identifier, 1, current['digest'], updated, '5', '7')
        with self.assertRaises(ValueError):
            approve_and_publish(publisher, identifier, 1, current['digest'], updated, 'bad', '7', explicit=True)
        self.assertEqual(publisher.calls, [])
        self.assertFalse(self.store.load(identifier)['approved'])
        result = approve_and_publish(publisher, identifier, 1, current['digest'], updated, '5', '7', explicit=True)
        self.assertEqual(result[0]['title'], 'Edited case title')
        self.assertEqual(publisher.calls[0][1:3], (5, 7))
        self.assertTrue(publisher.calls[0][3]['explicit'])
        self.assertEqual(publisher.calls[0][4]['revision'], 2)
        self.assertTrue(publisher.calls[0][4]['approved'])

    def test_approval_rejects_a_stale_displayed_revision(self):
        identifier = self.store.save(story(), [], draft())
        current = self.store.load(identifier)
        self.store.save(story(), [], draft(), identifier, expected_revision=1)
        class FakePublisher:
            store = self.store
            def publish(self, *args, **kwargs):
                raise AssertionError('A stale draft must not be published')
        with self.assertRaises(ValueError):
            approve_and_publish(FakePublisher(), identifier, 1, current['digest'], draft(), '5', '7', explicit=True)
        with self.assertRaises(ValueError):
            self.store.save(story(), [], draft(), identifier, expected_revision=1)

    def test_coverage_traceability_and_duplicates(self):
        self.assertEqual(set(criteria_for(story())), {'AC1', 'AC2'})
        for mutation in ('coverage', 'source', 'story', 'duplicate', 'steps'):
            value = draft()
            case = value['test_cases'][0]
            if mutation == 'coverage': case['acceptance_criteria_reference'] = ['AC1']
            if mutation == 'source': case['rag_source_references'] = ['invented']
            if mutation == 'story': case['story_id'] = 456
            if mutation == 'duplicate': value['test_cases'].append(copy.deepcopy(case))
            if mutation == 'steps': case['steps'][0]['step_number'] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_draft(value, story(), [])

    def test_steps_xml_escapes_data(self):
        from xml.etree.ElementTree import fromstring
        case = TestDraft.model_validate(draft()).test_cases[0]
        xml = steps_xml(case.steps)
        self.assertIn('&lt;empty&gt;', xml)
        self.assertIn('name=&amp;', xml)
        self.assertEqual(fromstring(xml).find('step').get('type'), 'ActionStep')

    def test_structured_langchain_generation(self):
        from langchain_core.runnables import RunnableLambda
        class Model:
            def with_structured_output(self, schema):
                return RunnableLambda(lambda prompt: schema.model_validate(draft()))
        result = generate_tests(Model(), story(), [])
        self.assertEqual(result.test_cases[0].story_id, 123)

    def test_selected_regeneration_preserves_other_cases(self):
        from langchain_core.runnables import RunnableLambda
        original = draft()
        second = copy.deepcopy(original['test_cases'][0])
        second['title'] = 'Preserve this scenario'
        original['test_cases'].append(second)
        returned = copy.deepcopy(original)
        returned['test_cases'][0]['description'] = 'Selected case regenerated'
        returned['test_cases'][1]['description'] = 'Unwanted model edit'
        class Model:
            def with_structured_output(self, schema):
                return RunnableLambda(lambda _: schema.model_validate(returned))
        result = regenerate_selected(Model(), story(), [], original, [original['test_cases'][0]['title']])
        self.assertEqual(result.test_cases[0].description, 'Selected case regenerated')
        self.assertEqual(result.test_cases[1].model_dump(), second)

    def test_reduced_source_grant_blocks_saved_context(self):
        source = {'reference_id': 'a', 'metadata': {'organization_id': self.scope.organization_id,
                  'project_id': 'p', 'source_type': 'azure_devops'}, 'content': 'customer'}
        identifier = self.store.save(story(), [source], draft())
        restricted = Scope(self.scope.organization_id, 'p', frozenset({'files'}))
        with self.assertRaises(PermissionError):
            WorkflowStore(self.temp.name, restricted).load(identifier)

    def test_publish_approval_retry_and_partial_remote_recovery(self):
        identifier = self.store.save(story(), [], draft())
        current = self.store.load(identifier)
        created, memberships, remote, suite_members = [], [], {}, set()
        def handler(request):
            path = request.url.path
            if path.endswith('/_apis/projects/p'):
                return httpx.Response(200, json={'id': 'p', 'name': 'Example'})
            if path.endswith('/workitemsbatch'):
                return httpx.Response(200, json={'value': [story()]})
            if path.endswith('/plans/1'):
                return httpx.Response(200, json={'id': 1, 'project': {'id': 'p'}})
            if path.endswith('/suites/2'):
                return httpx.Response(200, json={'id': 2})
            if path.endswith('/wiql'):
                return httpx.Response(200, json={'workItems': [{'id': 77}] if remote else []})
            if '/workitems/$Test Case' in path:
                created.append(json.loads(request.content))
                remote['id'] = 77
                return httpx.Response(200, json={'id': 77})
            if path.endswith('/TestCase'):
                if request.method == 'GET':
                    return httpx.Response(200, json={'value': [{'workItem': {'id': case_id}} for case_id in suite_members]})
                memberships.append(json.loads(request.content))
                if len(memberships) == 1:
                    return httpx.Response(500, json={})
                suite_members.add(memberships[-1][0]['workItem']['id'])
                return httpx.Response(200, json=[])
            raise AssertionError(path)
        azure = AzureDevOps(self.scope.organization_id, SecretStr('secret'), transport=httpx.MockTransport(handler))
        publisher = Publisher(azure, self.store, 'p')
        with self.assertRaises(PermissionError):
            publisher.publish(identifier, 1, 2, explicit=True, expected_digest=current['digest'])
        with self.assertRaises(IntegrationError):
            approve_and_publish(publisher, identifier, 1, current['digest'], draft(), 1, 2, explicit=True)
        self.assertTrue(self.store.load(identifier)['approved'])
        result = approve_and_publish(publisher, identifier, 1, current['digest'], draft(), 1, 2, explicit=True)
        self.assertEqual(result[0]['case_id'], 77)
        self.assertEqual(len(created), 1)
        publisher.publish(identifier, 1, 2, explicit=True, expected_digest=current['digest'])
        self.assertEqual(len(created), 1)
        self.assertEqual(len(memberships), 2)
        self.assertTrue(any(p['path'] == '/relations/-' for p in created[0]))
        azure.close()


class AutomationTests(unittest.TestCase):
    def test_paths_and_explicit_execution(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ('../escape.ts', '/escape', 'C:/escape', 'pages/../../escape', 'pages\\escape'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_file(root, name)
            workspace = AutomationWorkspace(root)
            with self.assertRaises(PermissionError):
                workspace.run('test', 'https://example.com')
            with self.assertRaises(PermissionError):
                workspace.approve(workspace.digest())

    def test_digest_invalidated_after_edit(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = AutomationWorkspace(root)
            proposal = {'summary': 'Example', 'files': [{'path': 'pages/customer.ts', 'content': 'export {};'}],
                        'assumptions': [], 'locator_evidence': []}
            workspace.write(proposal, {'approved': True, 'id': 'workflow', 'digest': 'draft-digest'})
            digest = workspace.digest()
            workspace.approve(digest, explicit=True)
            (workspace.path / 'pages/customer.ts').write_text('changed', encoding='utf-8')
            with self.assertRaises(PermissionError):
                workspace.run('test', 'https://example.com', explicit=True)

    def test_generated_configs_and_main_branch_are_protected(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = AutomationWorkspace(root)
            with self.assertRaises(ValueError):
                workspace.write({'summary': 'Bad', 'files': [{'path': 'package.json', 'content': '{}'}]},
                                {'approved': True, 'id': 'w', 'digest': 'd'})
            with self.assertRaises(PermissionError):
                create_review_branch(root, {'digest': 'd'}, 'd', 'main', root + '-branch')

    def test_evolution_diff_requires_tests_docs_and_stale_review_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            proposal = {'title': 'Reuse selector', 'rationale': 'Shared behavior', 'existing_patterns': [],
                        'files': [{'path': 'src/shared.py', 'content': 'def shared(): return 1\n'},
                                  {'path': 'tests/test_shared.py', 'content': 'import unittest\n'},
                                  {'path': 'docs/shared.md', 'content': '# Shared\n'}], 'regression_notes': []}
            review = build_diff(root, proposal)
            self.assertIn('+def shared()', review['diff'])
            path = Path(root) / 'src/shared.py'
            path.parent.mkdir()
            path.write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                create_review_branch(root, review, review['digest'], 'ai-evolution/shared', root + '-new', explicit=True)

    def test_typecheck_and_bdd_must_pass_before_test_execution(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = AutomationWorkspace(root)
            workspace.write({'summary': 'Trusted fixture', 'files': [{'path': 'pages/sample.ts', 'content': 'export {};'}]},
                            {'approved': True, 'id': 'w', 'digest': 'd'})
            workspace.approve(workspace.digest(), explicit=True)
            with self.assertRaises(PermissionError):
                workspace.run('test', 'https://example.com', explicit=True)

    @unittest.skipUnless(shutil.which('git'), 'Git not installed')
    def test_approved_worktree_keeps_original_branch_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / 'repo'
            repository.mkdir()
            git = shutil.which('git')
            def run(*args):
                return subprocess.run([git, '-C', str(repository), *args], capture_output=True, text=True, check=True).stdout.strip()
            run('init', '-b', 'main')
            (repository / 'README.md').write_text('Trusted fixture', encoding='utf-8')
            run('add', 'README.md')
            run('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'Fixture')
            proposal = {'title': 'Shared component', 'rationale': 'Reuse', 'existing_patterns': [],
                        'files': [{'path': 'src/shared.py', 'content': 'VALUE = 1\n'},
                                  {'path': 'tests/test_shared.py', 'content': 'import unittest\n'},
                                  {'path': 'docs/shared.md', 'content': 'Shared component\n'}], 'regression_notes': []}
            review = build_diff(repository, proposal)
            destination = Path(directory) / 'review'
            result = create_review_branch(repository, review, review['digest'], 'ai-evolution/shared', destination, explicit=True)
            self.assertEqual(result['branch'], 'ai-evolution/shared')
            self.assertEqual(run('branch', '--show-current'), 'main')
            self.assertFalse((repository / 'src/shared.py').exists())
            self.assertTrue((destination / 'src/shared.py').exists())


if __name__ == '__main__':
    unittest.main()

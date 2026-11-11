import copy
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from src.RAG.Rag import Rag
from src.integrations.azure_devops import AzureDevOps
from src.integrations.confluence import Confluence
from src.integrations.http import IntegrationError
from src.test_engineering.knowledge import KnowledgeIndex, Scope, story_document, file_documents
from src.test_engineering.workflow import WorkflowStore, validate_draft, criteria_for, generate_tests, regenerate_selected
from src.test_engineering.publishing import Publisher, steps_xml
from src.test_engineering.models import TestDraft
from src.automation.workspace import AutomationWorkspace, safe_file
from src.agents.framework_evolution import build_diff, create_review_branch


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        value = text.lower()
        return [float(value.count('customer')), float(value.count('required')), float(value.count('bug')), 1.0]


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
        created, memberships, remote = [], [], {}
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
                memberships.append(json.loads(request.content))
                if len(memberships) == 1:
                    return httpx.Response(500, json={})
                return httpx.Response(200, json=[])
            raise AssertionError(path)
        azure = AzureDevOps(self.scope.organization_id, SecretStr('secret'), transport=httpx.MockTransport(handler))
        publisher = Publisher(azure, self.store, 'p')
        with self.assertRaises(PermissionError):
            publisher.publish(identifier, 1, 2, explicit=True, expected_digest=current['digest'])
        self.store.approve(identifier, 1, current['digest'], explicit=True)
        with self.assertRaises(IntegrationError):
            publisher.publish(identifier, 1, 2, explicit=True, expected_digest=current['digest'])
        result = publisher.publish(identifier, 1, 2, explicit=True, expected_digest=current['digest'])
        self.assertEqual(result[0]['case_id'], 77)
        self.assertEqual(len(created), 1)
        publisher.publish(identifier, 1, 2, explicit=True, expected_digest=current['digest'])
        self.assertEqual(len(created), 1)
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

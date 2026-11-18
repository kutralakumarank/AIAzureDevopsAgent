import os
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest


class UITests(unittest.TestCase):
    def test_related_knowledge_groups_chunks_by_source(self):
        code = '''from src.ui.streamlitui.presentation import show_evidence
base = {"organization_id": "org", "project_id": "project", "source_type": "files", "title": "Requirements.pdf"}
show_evidence([
    {"reference_id": "a", "metadata": {**base, "source_document_id": "first"}, "content": "First passage"},
    {"reference_id": "b", "metadata": {**base, "source_document_id": "first"}, "content": "Second passage"},
    {"reference_id": "c", "metadata": {**base, "source_document_id": "first"}, "content": "First passage"},
    {"reference_id": "d", "metadata": {**base, "source_document_id": "second"}, "content": "Different document"},
])
'''
        app = AppTest.from_string(code, default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.expander), 2)
        rendered = [item.value for item in app.markdown]
        self.assertEqual(rendered.count('First passage'), 1)
        self.assertIn('Second passage', rendered)
        self.assertIn('Different document', rendered)

    def test_selector_names_and_provider_switch(self):
        code = '''from src.ui.streamlitui.test_engineering import pick, model_controls
for label in ("Project", "Team", "Sprint"):
    pick(label, [{"id": "hidden-guid", "name": label + " name"}], label)
model_controls()
'''
        with patch('src.ui.streamlitui.test_engineering.list_ollama_models', return_value=['llama3.2:latest', 'qwen:latest']):
            app = AppTest.from_string(code, default_timeout=60).run()
            for label in ('Project', 'Team', 'Sprint'):
                selector = next(item for item in app.selectbox if item.label == label)
                self.assertEqual(selector.options, [label + ' name'])
                self.assertEqual(selector.value['id'], 'hidden-guid')
            self.assertTrue(any(item.label == 'Groq API key' for item in app.text_input))
            app.selectbox(key='engineering_llm_provider').select('Ollama').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='engineering_ollama_model').options, ['llama3.2:latest', 'qwen:latest'])
            self.assertFalse(any(item.label == 'Groq API key' for item in app.text_input))

    def test_unavailable_ollama_shows_help_and_refreshes(self):
        code = 'from src.ui.streamlitui.test_engineering import model_controls\nmodel_controls()'
        with patch('src.ui.streamlitui.test_engineering.list_ollama_models', side_effect=[ValueError('Start Ollama.'), ['local:latest']]):
            app = AppTest.from_string(code, default_timeout=60).run()
            app.selectbox(key='engineering_llm_provider').select('Ollama').run()
            self.assertIn('Start Ollama', app.warning[0].value)
            app.button(key='engineering_refresh_ollama').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='engineering_ollama_model').options, ['local:latest'])

    def test_app_opens_test_engineering_directly(self):
        app = AppTest.from_file('app.py', default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, 'AI Test Engineering')
        self.assertTrue(any(field.label == 'Organization URL' for field in app.text_input))
        self.assertTrue(any(field.label == 'Azure DevOps PAT' for field in app.text_input))
        self.assertTrue(any(button.label == 'Connect and validate access' for button in app.button))
        self.assertFalse(app.sidebar.radio)
        self.assertFalse(app.error)

    def test_connected_project_renders_tabs_without_remote_calls(self):
        class Azure:
            organization_id = 'https://dev.azure.com/offline'
            def authorize_project(self, project): return {'id': project, 'name': 'Demo'}
            def teams(self, project): return [{'id': 'team', 'name': 'Demo Team'}]
            def iterations(self, project, team): return [{'id': 'sprint', 'name': 'Sprint', 'path': 'Demo\\Sprint'}]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'ENGINEERING_DATA_DIR': directory}):
            app = AppTest.from_string('from src.ui.streamlitui.test_engineering import render\nrender()', default_timeout=60)
            app.session_state['engineering_azure'] = Azure()
            app.session_state['engineering_identity'] = 'offline-user'
            app.session_state['engineering_projects'] = [{'id': 'project', 'name': 'Demo', 'description': 'Demo project'}]
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual([tab.label for tab in app.tabs], ['Stories', 'Knowledge', 'Review and publish'])
            self.assertFalse(any('Discover other accessible organizations' in button.label for button in app.button))

    def test_connect_uses_streamlit_credentials_instead_of_environment(self):
        azure = MagicMock()
        azure.validate.return_value = {'id': 'ui-user'}
        azure.projects.return_value = []
        env = {'AZURE_DEVOPS_ORGANIZATION_URL': 'https://dev.azure.com/unused-env',
               'AZURE_DEVOPS_PAT': 'unused-env-token', 'AZURE_DEVOPS_API_VERSION': '7.1'}
        with patch.dict(os.environ, env), patch('src.ui.streamlitui.test_engineering.AzureDevOps', return_value=azure) as factory:
            app = AppTest.from_file('app.py', default_timeout=60).run()
            app.text_input(key='engineering_organization_url').set_value('https://dev.azure.com/from-ui')
            app.text_input(key='engineering_pat').set_value('ui-test-token')
            next(button for button in app.button if button.label == 'Connect and validate access').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            args = factory.call_args.args
            self.assertEqual(args[0], 'https://dev.azure.com/from-ui')
            self.assertEqual(args[1].get_secret_value(), 'ui-test-token')
            self.assertEqual(args[2], '7.1')
            factory.assert_called_once()
            azure.validate.assert_called_once_with()
            self.assertEqual(app.session_state['engineering_identity'], 'ui-user')

    def test_empty_connection_form_does_not_fall_back_to_env_credentials(self):
        env = {'AZURE_DEVOPS_ORGANIZATION_URL': 'https://dev.azure.com/unused-env', 'AZURE_DEVOPS_PAT': 'unused-env-token'}
        with patch.dict(os.environ, env), patch('src.ui.streamlitui.test_engineering.AzureDevOps') as factory:
            app = AppTest.from_file('app.py', default_timeout=60).run()
            next(button for button in app.button if button.label == 'Connect and validate access').click().run()
            factory.assert_not_called()
            self.assertFalse(app.exception)
            self.assertIn('Enter your Azure DevOps organization URL and PAT', app.error[0].value)

    def test_provider_errors_show_safe_actionable_messages(self):
        import httpx
        from groq import BadRequestError
        from huggingface_hub.errors import HfHubHTTPError
        request = httpx.Request('POST', 'https://example.invalid')
        cases = [
            (HfHubHTTPError('private-provider-data', response=httpx.Response(402, request=request)), 'credits and billing'),
            (BadRequestError('private-provider-data', response=httpx.Response(400, request=request),
                             body={'code': 'tool_use_failed', 'failed_generation': 'private-provider-data'}), 'structured response'),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__), patch('src.ui.streamlitui.test_engineering._render', side_effect=error):
                app = AppTest.from_file('app.py', default_timeout=60).run()
                self.assertFalse(app.exception)
                self.assertIn(expected, app.error[0].value)
                self.assertNotIn('private-provider-data', app.error[0].value)

    def test_story_and_uploaded_documents_are_added_incrementally(self):
        import faiss
        from test_platform import LocalEmbeddings, story
        from src.RAG.Rag import Rag
        from src.test_engineering.knowledge import KnowledgeIndex, Scope, story_document
        class Azure:
            organization_id = 'https://dev.azure.com/offline'
            def authorize_project(self, project): return {'id': project, 'name': 'Demo'}
            def teams(self, project): return [{'id': 'team', 'name': 'Demo Team'}]
            def iterations(self, project, team): return [{'id': 'sprint', 'name': 'Sprint', 'path': 'Demo\\Sprint'}]
            def story(self, project, identifier): return story()
        azure, embeddings = Azure(), LocalEmbeddings()
        allowed = 'azure_devops,confluence,files,approved_tests'
        scope = Scope(azure.organization_id, 'project', frozenset(allowed.split(',')))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'ENGINEERING_DATA_DIR': directory, 'ENGINEERING_ALLOWED_SOURCES': allowed}), \
                patch('src.ui.streamlitui.test_engineering.Rag', side_effect=lambda: Rag(embeddings=embeddings)):
            app = AppTest.from_string('from src.ui.streamlitui.test_engineering import render\nrender()', default_timeout=60)
            app.session_state['engineering_azure'] = azure
            app.session_state['engineering_identity'] = 'offline-user'
            app.session_state['engineering_projects'] = [{'id': 'project', 'name': 'Demo'}]
            app.session_state['engineering_stories'] = {'scope': scope.key, 'iteration': 'sprint', 'items': [story()]}
            app.run()
            next(item for item in app.selectbox if item.label == 'Story').select(story()).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            root = Path(directory) / sha256(b'offline-user').hexdigest()
            index = KnowledgeIndex(root, scope, Rag(embeddings=embeddings))
            self.assertTrue(index.is_current(story_document(story(), scope)))
            self.assertEqual(faiss.read_index(str(index.faiss_path)).ntotal, 1)
            self.assertFalse(app.get('json'))
            first_calls = embeddings.calls
            app.run()
            self.assertEqual(embeddings.calls, first_calls)
            uploader = next(item for item in app.file_uploader if item.label == 'Add business documents')
            uploader.set_value([('requirements.txt', b'Customer name is required.', 'text/plain')]).run()
            self.assertFalse(app.exception)
            self.assertEqual(embeddings.calls, first_calls + 1)
            self.assertEqual(faiss.read_index(str(index.faiss_path)).ntotal, 2)
            app.run()
            self.assertEqual(embeddings.calls, first_calls + 1)
            uploader = next(item for item in app.file_uploader if item.label == 'Add business documents')
            uploader.set_value([('requirements.txt', b'Customer name is required and unique.', 'text/plain'),
                                ('workflow.txt', b'Customer approval workflow.', 'text/plain')]).run()
            self.assertFalse(app.exception)
            self.assertEqual(embeddings.calls, first_calls + 3)
            self.assertEqual(faiss.read_index(str(index.faiss_path)).ntotal, 3)
            self.assertTrue(index.is_current(story_document(story(), scope)))

    def test_retrieving_sprint_stories_also_indexes_new_items(self):
        import faiss
        from test_platform import LocalEmbeddings, story
        from src.RAG.Rag import Rag
        from src.test_engineering.knowledge import KnowledgeIndex, Scope, story_document
        class Azure:
            organization_id = 'https://dev.azure.com/offline'
            def authorize_project(self, project): return {'id': project, 'name': 'Demo'}
            def teams(self, project): return [{'id': 'team', 'name': 'Demo Team'}]
            def iterations(self, project, team): return [{'id': 'sprint', 'name': 'Sprint', 'path': 'Demo\\Sprint'}]
            def query_ids(self, project, iteration, types): return [story()['id']]
            def work_items(self, project, identifiers): return [story()]
        azure, embeddings = Azure(), LocalEmbeddings()
        allowed = 'azure_devops,confluence,files,approved_tests'
        scope = Scope(azure.organization_id, 'project', frozenset(allowed.split(',')))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'ENGINEERING_DATA_DIR': directory, 'ENGINEERING_ALLOWED_SOURCES': allowed}), \
                patch('src.ui.streamlitui.test_engineering.Rag', side_effect=lambda: Rag(embeddings=embeddings)):
            app = AppTest.from_string('from src.ui.streamlitui.test_engineering import render\nrender()', default_timeout=60)
            app.session_state['engineering_azure'] = azure
            app.session_state['engineering_identity'] = 'offline-user'
            app.session_state['engineering_projects'] = [{'id': 'project', 'name': 'Demo'}]
            app.run()
            next(button for button in app.button if button.label == 'Retrieve sprint work items').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            root = Path(directory) / sha256(b'offline-user').hexdigest()
            index = KnowledgeIndex(root, scope, Rag(embeddings=embeddings))
            self.assertTrue(index.is_current(story_document(story(), scope)))
            self.assertEqual(faiss.read_index(str(index.faiss_path)).ntotal, 1)
            calls = embeddings.calls
            next(button for button in app.button if button.label == 'Retrieve sprint work items').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(embeddings.calls, calls)

    def test_failed_story_indexing_shows_retry_without_repeated_requests(self):
        import httpx
        from huggingface_hub.errors import HfHubHTTPError
        from test_platform import LocalEmbeddings, story
        from src.RAG.Rag import Rag
        from src.test_engineering.knowledge import KnowledgeIndex, Scope, story_document
        class RecoverableEmbeddings(LocalEmbeddings):
            attempts = 0
            fail = True
            def embed_documents(self, texts):
                self.attempts += 1
                if self.fail:
                    request = httpx.Request('POST', 'https://example.invalid')
                    raise HfHubHTTPError('private-provider-data', response=httpx.Response(402, request=request))
                return super().embed_documents(texts)
        class Azure:
            organization_id = 'https://dev.azure.com/offline'
            def authorize_project(self, project): return {'id': project, 'name': 'Demo'}
            def teams(self, project): return [{'id': 'team', 'name': 'Demo Team'}]
            def iterations(self, project, team): return [{'id': 'sprint', 'name': 'Sprint', 'path': 'Demo\\Sprint'}]
            def story(self, project, identifier): return story()
        azure, embeddings = Azure(), RecoverableEmbeddings()
        allowed = 'azure_devops,confluence,files,approved_tests'
        scope = Scope(azure.organization_id, 'project', frozenset(allowed.split(',')))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'ENGINEERING_DATA_DIR': directory, 'ENGINEERING_ALLOWED_SOURCES': allowed}), \
                patch('src.ui.streamlitui.test_engineering.Rag', side_effect=lambda: Rag(embeddings=embeddings)):
            app = AppTest.from_string('from src.ui.streamlitui.test_engineering import render\nrender()', default_timeout=60)
            app.session_state['engineering_azure'] = azure
            app.session_state['engineering_identity'] = 'offline-user'
            app.session_state['engineering_projects'] = [{'id': 'project', 'name': 'Demo'}]
            app.session_state['engineering_stories'] = {'scope': scope.key, 'iteration': 'sprint', 'items': [story()]}
            app.run()
            next(item for item in app.selectbox if item.label == 'Story').select(story()).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('HTTP 402' in item.value for item in app.warning))
            self.assertTrue(any(button.label == 'Retry saving Story' for button in app.button))
            self.assertEqual(embeddings.attempts, 1)
            app.run()
            self.assertEqual(embeddings.attempts, 1)
            embeddings.fail = False
            next(button for button in app.button if button.label == 'Retry saving Story').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(embeddings.attempts, 2)
            root = Path(directory) / sha256(b'offline-user').hexdigest()
            index = KnowledgeIndex(root, scope, Rag(embeddings=embeddings))
            self.assertTrue(index.is_current(story_document(story(), scope)))

    def test_one_ok_button_saves_edits_approves_and_publishes_by_ids(self):
        from test_platform import draft, story
        from src.test_engineering.knowledge import Scope
        from src.test_engineering.workflow import WorkflowStore
        class Azure:
            organization_id = 'https://dev.azure.com/offline'
            def authorize_project(self, project): return {'id': project, 'name': 'Demo'}
            def teams(self, project): return [{'id': 'team', 'name': 'Demo Team'}]
            def iterations(self, project, team): return [{'id': 'sprint', 'name': 'Sprint', 'path': 'Demo\\Sprint'}]
        azure = Azure()
        allowed = 'azure_devops,confluence,files,approved_tests'
        scope = Scope(azure.organization_id, 'project', frozenset(allowed.split(',')))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'ENGINEERING_DATA_DIR': directory, 'ENGINEERING_ALLOWED_SOURCES': allowed}):
            root = Path(directory) / sha256(b'offline-user').hexdigest()
            store = WorkflowStore(root, scope)
            identifier = store.save(story(), [], draft())
            publisher = MagicMock()
            publisher.store = store
            publisher.publish.return_value = [{'case_id': 101, 'title': 'User edited case', 'status': 'published'}]
            with patch('src.ui.streamlitui.test_engineering.Publisher', return_value=publisher):
                app = AppTest.from_string('from src.ui.streamlitui.test_engineering import render\nrender()', default_timeout=60)
                app.session_state['engineering_azure'] = azure
                app.session_state['engineering_identity'] = 'offline-user'
                app.session_state['engineering_projects'] = [{'id': 'project', 'name': 'Demo'}]
                app.run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertFalse(app.get('json'))
                self.assertFalse(any(item.label == 'Automation feasibility' for item in app.selectbox))
                prefix = f'case_{scope.key}_{identifier}_1_1'
                app.text_input(key=prefix + '_title').set_value('User edited case')
                app.text_input(key=f'engineering_plan_id_{scope.key}_{identifier}').set_value('12')
                app.text_input(key=f'engineering_suite_id_{scope.key}_{identifier}').set_value('34')
                next(button for button in app.button if button.label == 'OK - approve and upload').click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                current = store.load(identifier)
                self.assertEqual(current['revision'], 2)
                self.assertTrue(current['approved'])
                self.assertEqual(current['draft']['test_cases'][0]['title'], 'User edited case')
                self.assertEqual(current['draft']['test_cases'][0]['automation_feasibility'], 'High')
                args, kwargs = publisher.publish.call_args
                self.assertEqual(args, (identifier, 12, 34))
                self.assertTrue(kwargs['explicit'])
                self.assertEqual(kwargs['expected_digest'], current['digest'])
                self.assertIn('uploaded successfully', ' '.join(item.value for item in app.success).lower())


if __name__ == '__main__':
    unittest.main()

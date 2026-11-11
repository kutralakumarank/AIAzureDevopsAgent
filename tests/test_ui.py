import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest


class UITests(unittest.TestCase):
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
            self.assertEqual([tab.label for tab in app.tabs], ['Stories', 'Knowledge', 'Review and publish', 'Automation', 'Framework evolution'])

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


if __name__ == '__main__':
    unittest.main()

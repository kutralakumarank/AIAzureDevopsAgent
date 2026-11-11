"""Run: python -m examples.offline_demo --output .local/demo (no live credentials)."""
import argparse
import json
from pathlib import Path

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from src.RAG.Rag import Rag
from src.integrations.azure_devops import AzureDevOps
from src.integrations.confluence import Confluence
from src.test_engineering.knowledge import Scope, KnowledgeIndex, confluence_documents, story_document
from src.test_engineering.workflow import WorkflowStore, generate_tests
from src.test_engineering.publishing import Publisher
from src.automation.workspace import AutomationWorkspace


class DemoEmbeddings(Embeddings):
    """Deterministic offline substitute, not the configured production embedding model."""
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.lower().count(word)) for word in ('customer', 'name', 'required')] + [1.0]


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    item = {'id': 123, 'rev': 1, 'fields': {'System.TeamProject': 'Demo', 'System.Title': 'Create customer',
            'System.Description': 'A user creates a customer with a name.',
            'Microsoft.VSTS.Common.AcceptanceCriteria': '<li>Name is required</li><li>Valid customer is saved</li>'}}
    remote_cases = []
    def azure_response(request):
        path = request.url.path
        if path.endswith('/connectionData'):
            body = {'authenticatedUser': {'id': 'offline-user'}}
        elif path.endswith('/projects'):
            body = {'value': [{'id': 'demo-project', 'name': 'Demo'}]}
        elif path.endswith('/projects/demo-project'):
            body = {'id': 'demo-project', 'name': 'Demo'}
        elif path.endswith('/workitemsbatch'):
            body = {'value': [item]}
        elif path.endswith('/plans/1'):
            body = {'id': 1, 'name': 'Demo plan', 'project': {'id': 'demo-project'}}
        elif path.endswith('/suites/2'):
            body = {'id': 2, 'name': 'Demo suite'}
        elif path.endswith('/wiql'):
            query = json.loads(request.content)['query']
            body = {'workItems': [{'id': c['id']} for c in remote_cases if c['tag'] in query]}
        elif '/workitems/$Test Case' in path:
            patch = json.loads(request.content)
            case = {'id': 1000 + len(remote_cases), 'tag': next(p['value'] for p in patch if p['path'] == '/fields/System.Tags')}
            remote_cases.append(case)
            body = {'id': case['id']}
        elif path.endswith('/TestCase'):
            body = []
        else:
            raise AssertionError('Unexpected demo route: ' + path)
        return httpx.Response(200, json=body)
    azure = AzureDevOps('https://dev.azure.com/offline-demo', SecretStr('offline-placeholder'), transport=httpx.MockTransport(azure_response))
    confluence = Confluence('https://offline-demo.atlassian.net', 'offline', SecretStr('offline-placeholder'),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={'results': [{
            'id': '99', 'title': 'Customer form history', 'version': {'number': 1},
            'body': {'storage': {'value': '<p>Customer name is required. Historical rule: verify against current requirements.</p>'}},
            '_links': {'webui': '/spaces/DEMO/pages/99'}}]})))
    azure.validate()
    project = azure.projects()[0]
    scope = Scope(azure.organization_id, project['id'], frozenset({'azure_devops', 'confluence'}))
    index = KnowledgeIndex(output, scope, Rag(embeddings=DemoEmbeddings()), embedding_version='offline-demo:v1')
    initial = index.sync([story_document(item, scope)], 'azure_devops')
    wiki = index.sync(confluence_documents(confluence, scope), 'confluence', full_snapshot=True)
    incremental = index.sync([story_document(item, scope)], 'azure_devops')
    current = azure.story(project['id'], 123)
    context = index.retrieve(current)
    reference = context[0]['reference_id']
    cases = []
    for number, title, action, expected in [(1, 'Reject empty name', 'Leave name blank and save', 'Required name message is displayed'),
                                             (2, 'Save valid customer', 'Enter name Demo Customer and save', 'Customer appears in the customer list')]:
        cases.append({'title': title, 'description': title, 'story_id': 123, 'preconditions': ['User is signed in'],
                      'steps': [{'step_number': 1, 'action': action, 'test_data': '', 'expected_result': expected}],
                      'priority': 1, 'category': 'Functional', 'acceptance_criteria_reference': [f'AC{number}'],
                      'rag_source_references': [reference], 'automation_feasibility': 'High'})
    class DemoLLM:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: schema.model_validate({'test_cases': cases, 'gaps': [], 'ambiguities': [], 'assumptions': []}))
    draft = generate_tests(DemoLLM(), current, context)
    store = WorkflowStore(output, scope)
    workflow_id = store.save(current, context, draft.model_dump())
    workflow = store.load(workflow_id)
    # This is a deterministic demo of the approval API; the live UI requires a human click.
    store.approve(workflow_id, workflow['revision'], workflow['digest'], explicit=True)
    published = Publisher(azure, store, project['id']).publish(workflow_id, 1, 2, explicit=True, expected_digest=workflow['digest'])
    repeated = Publisher(azure, store, project['id']).publish(workflow_id, 1, 2, explicit=True, expected_digest=workflow['digest'])
    workspace = AutomationWorkspace(output / 'automation')
    workspace.write({'summary': 'Review-only demo artifacts', 'files': [
        {'path': 'features/customer.feature', 'content': 'Feature: Customer creation\n  Scenario: Reject empty name\n    When the user saves an empty customer\n    Then the required name message is visible\n'},
        {'path': 'pages/customer.ts', 'content': '// Review placeholder: supply verified DOM locators before implementation.\nexport {};\n'}],
        'assumptions': ['Application DOM is unavailable in this offline demo; step implementation requires inspection.'], 'locator_evidence': []}, store.load(workflow_id))
    report = {'mode': 'offline: mocked services and deterministic LLM/embeddings', 'project': project,
              'sync': initial, 'confluence_sync': wiki, 'incremental_sync': incremental,
              'retrieved_sources': len(context), 'workflow_id': workflow_id, 'approved': store.load(workflow_id)['approved'],
              'published': published, 'retry': repeated, 'remote_case_count': len(remote_cases),
              'automation_workspace': str(workspace.path), 'automation_executed': False}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    azure.close()
    confluence.close()
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='.local/demo')
    run(parser.parse_args().output)

"""Approval-bound, remote-tag-deduplicated Azure Test Plans publishing."""
from hashlib import sha256
from html import escape
import json
from xml.etree.ElementTree import Element, SubElement, tostring

from src.integrations.azure_devops import segment, wiql_literal
from src.test_engineering.workflow import validate_draft


def steps_xml(steps):
    root = Element('steps', id='0', last=str(len(steps)))
    for step in steps:
        node = SubElement(root, 'step', id=str(step.step_number), type='ActionStep')
        action = step.action + (f'\nTest data: {step.test_data}' if step.test_data else '')
        SubElement(node, 'parameterizedString', isformatted='true').text = action
        SubElement(node, 'parameterizedString', isformatted='true').text = step.expected_result
        SubElement(node, 'description')
    return tostring(root, encoding='unicode')


def approve_review(store, workflow_id, revision, digest, edited_draft, *, explicit=False):
    """Save displayed edits and approve exactly the revision the user reviewed."""
    if not explicit:
        raise PermissionError('Explicit approval is required')
    workflow = store.load(workflow_id)
    if workflow['revision'] != revision or workflow['digest'] != digest:
        raise ValueError('Draft changed since it was displayed; reload and review it again')
    validated = validate_draft(edited_draft, workflow['story'], workflow['context']).model_dump()
    if validated != workflow['draft']:
        store.save(workflow['story'], workflow['context'], validated, workflow_id, expected_revision=revision)
        workflow = store.load(workflow_id)
        if workflow['revision'] != revision + 1 or workflow['draft'] != validated:
            raise ValueError('Draft changed while saving; reload and review it again')
    store.approve(workflow_id, workflow['revision'], workflow['digest'], explicit=True)
    approved = store.load(workflow_id)
    if approved['revision'] != workflow['revision'] or approved['digest'] != workflow['digest'] or not approved['approved']:
        raise ValueError('Draft changed during approval; reload and review it again')
    return approved


def approve_and_publish(publisher, workflow_id, revision, digest, edited_draft, plan_id, suite_id, *, explicit=False):
    """A single explicit action approves the displayed revision and publishes it."""
    if not explicit:
        raise PermissionError('Explicit approval and publish confirmation is required')
    ids = [str(plan_id), str(suite_id)]
    if any(not value.isascii() or not value.isdigit() or len(value) > 10 for value in ids):
        raise ValueError('Enter valid numeric Test Plan and Test Suite IDs')
    plan_id, suite_id = map(int, ids)
    if not (1 <= plan_id <= 2147483647 and 1 <= suite_id <= 2147483647):
        raise ValueError('Test Plan and Test Suite IDs must be between 1 and 2147483647')
    workflow = approve_review(publisher.store, workflow_id, revision, digest, edited_draft, explicit=True)
    return publisher.publish(workflow_id, plan_id, suite_id, explicit=True, expected_digest=workflow['digest'])


class Publisher:
    def __init__(self, azure, store, project_id):
        if azure.organization_id != store.scope.organization_id or str(project_id) != store.scope.project_id:
            raise PermissionError('Publishing target differs from workflow scope')
        self.azure, self.store, self.project = azure, store, project_id

    def publish(self, workflow_id, plan_id, suite_id, *, explicit=False, expected_digest=None):
        if not explicit:
            raise PermissionError('Explicit publish confirmation is required')
        workflow = self.store.load(workflow_id)
        if not workflow['approved'] or expected_digest != workflow['digest']:
            raise PermissionError('Current reviewed revision must be approved')
        project = self.azure.authorize_project(self.project)
        plan, _ = self.azure.request('GET', f'/{segment(self.project)}/_apis/testplan/plans/{int(plan_id)}')
        if plan.get('project', {}).get('id') != self.project:
            raise PermissionError('Plan is outside the selected project')
        suite, _ = self.azure.request('GET', f'/{segment(self.project)}/_apis/testplan/Plans/{int(plan_id)}/suites/{int(suite_id)}')
        if int(suite['id']) != int(suite_id):
            raise ValueError('Suite validation failed')
        # Fetch current requirement for project membership, and invalidate stale story reviews.
        current_story = self.azure.story(self.project, workflow['story']['id'])
        if current_story.get('rev') != workflow['story'].get('rev'):
            raise ValueError('Story changed since generation; retrieve it and review new tests')
        draft = validate_draft(workflow['draft'], workflow['story'], workflow['context'])
        results = []
        # Serialize local publishers. The remote fingerprint also permits retry after a crash
        # between remote creation and local journaling (not a distributed uniqueness guarantee).
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT r.digest,r.approved FROM revisions r JOIN workflows w '
                                 'ON w.id=r.workflow AND w.current_revision=r.revision WHERE w.id=?', (workflow_id,)).fetchone()
            if not current or current[0] != expected_digest or not current[1]:
                raise PermissionError('Draft changed before publishing; review and approve the latest revision')
            suite_path = f'/{segment(self.project)}/_apis/testplan/Plans/{int(plan_id)}/Suites/{int(suite_id)}/TestCase'
            existing_cases = self.azure.values(suite_path)
            suite_ids = {str((item.get('workItem') or {}).get('id')) for item in existing_cases
                         if (item.get('workItem') or {}).get('id') is not None}
            for case in draft.test_cases:
                fingerprint = sha256(json.dumps([self.store.scope.key, case.model_dump()], sort_keys=True).encode()).hexdigest()
                tag = 'ai-test-' + fingerprint
                row = db.execute('SELECT case_id,status FROM publications WHERE fingerprint=?', (fingerprint,)).fetchone()
                case_id = row[0] if row else None
                if not case_id:
                    query = (f'SELECT [System.Id] FROM WorkItems WHERE [System.TeamProject] = {wiql_literal(project["name"])} '
                             f'AND [System.WorkItemType] = \'Test Case\' AND [System.Tags] CONTAINS {wiql_literal(tag)}')
                    matches, _ = self.azure.request('POST', f'/{segment(self.project)}/_apis/wit/wiql', json={'query': query})
                    found = matches.get('workItems', [])
                    if len(found) > 1:
                        raise ValueError('Duplicate remote fingerprint; resolve duplicates before continuing')
                    case_id = found[0]['id'] if found else None
                if not case_id:
                    references = {item['reference_id']: item for item in workflow['context']}
                    traceability = '\n'.join(
                        f'{ref}: {references[ref]["metadata"].get("title", "")} '
                        f'{references[ref]["metadata"].get("source_link", "")}' for ref in case.rag_source_references)
                    description = (case.description + '\nPreconditions:\n' + '\n'.join(case.preconditions)
                                   + '\nAcceptance criteria: ' + ', '.join(case.acceptance_criteria_reference)
                                   + '\nRAG sources:\n' + traceability
                                   + f'\nWorkflow: {workflow_id}; revision: {workflow["revision"]}')
                    fields = {'System.Title': case.title, 'System.Description': '<pre>' + escape(description) + '</pre>',
                              'Microsoft.VSTS.Common.Priority': case.priority, 'Microsoft.VSTS.TCM.Steps': steps_xml(case.steps),
                              'System.Tags': tag}
                    patch = [{'op': 'add', 'path': '/fields/' + key, 'value': value} for key, value in fields.items()]
                    patch.append({'op': 'add', 'path': '/relations/-', 'value': {
                        'rel': 'Microsoft.VSTS.Common.TestedBy-Reverse',
                        'url': f'{self.azure.http.base}/_apis/wit/workItems/{case.story_id}',
                        'attributes': {'comment': 'Approved AI test; revision ' + str(workflow['revision'])}}})
                    created, _ = self.azure.request('POST', f'/{segment(self.project)}/_apis/wit/workitems/$Test%20Case',
                                                   json=patch, headers={'Content-Type': 'application/json-patch+json'})
                    case_id = created['id']
                if str(case_id) not in suite_ids:
                    self.azure.request('POST', suite_path, json=[{'workItem': {'id': int(case_id)}}])
                    suite_ids.add(str(case_id))
                db.execute('INSERT OR REPLACE INTO publications(fingerprint,case_id,status) VALUES (?,?,?)',
                           (fingerprint, case_id, 'published'))
                results.append({'case_id': case_id, 'status': 'published', 'title': case.title})
        return results

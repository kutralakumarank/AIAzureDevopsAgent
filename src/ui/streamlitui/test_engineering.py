"""Streamlit interface for the test engineering platform."""
import json
import os
from hashlib import sha256
from pathlib import Path
import tempfile

import streamlit as st
from pydantic import SecretStr
from pydantic import ValidationError

from src.integrations.azure_devops import AzureDevOps
from src.integrations.confluence import Confluence
from src.LLMS.groqllm import GroqLLM
from src.LLMS.ollamallm import list_ollama_models, ollama_llm
from src.LLMS.provider_errors import provider_error_message
from src.RAG.Rag import Rag
from src.test_engineering.knowledge import Scope, KnowledgeIndex, story_document, confluence_documents, file_documents
from src.test_engineering.workflow import WorkflowStore, generate_tests, regenerate_selected, criteria_for, validate_draft
from src.test_engineering.publishing import Publisher, approve_review, approve_and_publish
# Automation and framework evolution are disabled in the UI.
# from src.automation.workspace import AutomationWorkspace, AutomationProposal, generate_automation, analyze_failures
# from src.agents.framework_evolution import propose_evolution, build_diff, create_review_branch, regression_and_pr
from src.ui.streamlitui.presentation import (show_details, show_story, show_comments, show_evidence,
                                              show_sync, show_results, edit_draft, validate_editor_fields)
# from src.ui.streamlitui.presentation import editable_proposal


def pick(label, items, key):
    if not items:
        st.info(f'No accessible {label.lower()} available.')
        return None
    return st.selectbox(label, items, format_func=lambda item: item.get('name') or item.get('title') or 'Unnamed', key=key)


def model_controls():
    from src.ui.uiconfigfile import Config
    provider = st.selectbox('LLM provider', ['Groq', 'Ollama'], key='engineering_llm_provider')
    if provider == 'Groq':
        model_name = st.selectbox('Generation model', Config().get_groq_model_options())
        key = st.text_input('Groq API key', type='password', key='engineering_groq_key')
        return {'provider': provider, 'GROQ_API_KEY': key, 'selected_groq_model': model_name}

    base_url = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
    refresh = st.button('Refresh local models', key='engineering_refresh_ollama')
    cache = st.session_state.get('engineering_ollama_models')
    if refresh or not cache or cache['url'] != base_url:
        try:
            cache = {'url': base_url, 'models': list_ollama_models(base_url), 'error': None}
        except ValueError as error:
            cache = {'url': base_url, 'models': [], 'error': str(error)}
        st.session_state['engineering_ollama_models'] = cache
    if cache['error']:
        st.warning(cache['error'])
    if cache['models']:
        model = st.selectbox('Generation model', cache['models'], key='engineering_ollama_model')
    else:
        model = ''
        st.info('Install a local model with `ollama pull <model>`, then refresh the model list.')
    return {'provider': provider, 'selected_ollama_model': model, 'ollama_base_url': base_url}


def llm_for(controls):
    if controls.get('provider') == 'Ollama':
        return ollama_llm(controls.get('selected_ollama_model'), controls['ollama_base_url'])
    llm = GroqLLM(controls).get_llm_model()
    if llm is None:
        raise ValueError('Configure a Groq API key first')
    return llm


def safe_error(error):
    message = provider_error_message(error)
    if message:
        return message
    if type(error) in (ValueError, PermissionError) and error.__class__.__module__ == 'builtins':
        return str(error)
    if error.__class__.__name__ == 'IntegrationError':
        return str(error)
    if isinstance(error, ValidationError):
        return 'Please review the test case fields and steps, then try again.'
    return f'{type(error).__name__}: operation failed. Check service configuration and permissions.'


def ensure_indexed(index, document, attempt_key, label):
    """Save a new or changed document once; retry failures only on explicit request."""
    failures = st.session_state.setdefault('engineering_index_failures', {})
    retry_key = 'retry_' + sha256(attempt_key.encode()).hexdigest()
    retry_rendered = False
    try:
        if index.is_current(document):
            failures.pop(attempt_key, None)
            return True
        if attempt_key in failures:
            st.warning(f'{label} was not saved to knowledge. {failures[attempt_key]}')
            retry_rendered = True
            if not st.button(f'Retry saving {label}', key=retry_key):
                return False
        with st.spinner(f'Saving {label} to knowledge...'):
            result = index.sync([document], document.metadata['source_type'])
        failures.pop(attempt_key, None)
        if result['changed']:
            st.success(f'{label} saved to the project knowledge index.')
        return True
    except Exception as error:
        failures[attempt_key] = safe_error(error)
        st.warning(f'{label} was not saved to knowledge. {failures[attempt_key]}')
        if not retry_rendered:
            st.button(f'Retry saving {label}', key=retry_key)
        return False


def render():
    st.set_page_config(page_title='AI Test Engineering', layout='wide')
    st.title('AI Test Engineering')
    st.caption('LangChain RAG · Azure DevOps · Confluence · manual test engineering')
    try:
        _render()
    except Exception as error:
        st.error(safe_error(error))


def _render():
    with st.expander('Connect Azure DevOps', expanded='engineering_azure' not in st.session_state):
        with st.form('azure_connect', clear_on_submit=True):
            organization = st.text_input('Organization URL', placeholder='https://dev.azure.com/your-organization', key='engineering_organization_url')
            pat = st.text_input('Azure DevOps PAT', type='password', key='engineering_pat')
            connect = st.form_submit_button('Connect and validate access')
        if connect:
            if not organization.strip() or not pat.strip():
                raise ValueError('Enter your Azure DevOps organization URL and PAT to connect')
            azure = AzureDevOps(organization.strip(), SecretStr(pat), os.getenv('AZURE_DEVOPS_API_VERSION', '7.1'))
            try:
                identity = azure.validate()
                projects = azure.projects()
            except Exception:
                azure.close()
                raise
            old = st.session_state.pop('engineering_azure', None)
            if old:
                old.close()
            st.session_state['engineering_azure'] = azure
            st.session_state['engineering_identity'] = identity['id']
            st.session_state['engineering_projects'] = projects
            for key in ('engineering_workflow', 'engineering_stories', 'engineering_project', 'engineering_team', 'engineering_iteration', 'engineering_automation'):
                st.session_state.pop(key, None)
            st.success('Authenticated. Project access is checked again before loading stored data.')
        if st.button('Disconnect'):
            azure = st.session_state.pop('engineering_azure', None)
            if azure:
                azure.close()
            for key in list(st.session_state):
                if key.startswith('engineering_'):
                    del st.session_state[key]
            st.rerun()
    azure = st.session_state.get('engineering_azure')
    if not azure:
        st.info('Enter your Azure DevOps organization URL and PAT above, then connect to discover projects.')
        return
    project = pick('Project', st.session_state['engineering_projects'], 'engineering_project')
    if not project:
        return
    st.caption(project.get('description') or 'No project description')
    azure.authorize_project(project['id'])
    allowed = frozenset(s.strip() for s in os.getenv('ENGINEERING_ALLOWED_SOURCES', 'azure_devops,confluence,files,approved_tests').split(',') if s.strip())
    scope = Scope(azure.organization_id, project['id'], allowed)
    # Separate authenticated identities, in addition to organization/project/source scope.
    root = Path(os.getenv('ENGINEERING_DATA_DIR', '.local/test-engineering')) / sha256(st.session_state['engineering_identity'].encode()).hexdigest()
    rag = Rag()
    index = KnowledgeIndex(root, scope, rag, os.getenv('ENGINEERING_EMBEDDING_VERSION', 'google/embeddinggemma-300m:v1'))
    store = WorkflowStore(root, scope)
    controls = model_controls()
    tabs = st.tabs(['Stories', 'Knowledge', 'Review and publish'])
    # Automation and Framework evolution tabs are intentionally disabled.
    with tabs[0]:
        team = pick('Team', azure.teams(project['id']), 'engineering_team')
        iteration = pick('Sprint', azure.iterations(project['id'], team['id']), 'engineering_iteration') if team else None
        types = st.text_input('Work item types (comma separated)', 'User Story,Product Backlog Item,Feature,Epic,Bug,Task')
        if iteration and st.button('Retrieve sprint work items'):
            ids = azure.query_ids(project['name'], iteration['path'], [t.strip() for t in types.split(',') if t.strip()])
            items = azure.work_items(project['id'], ids)
            st.session_state['engineering_stories'] = {'scope': scope.key, 'iteration': iteration['id'], 'items': items}
            st.success(f'Retrieved {len(items)} work item(s).')
            if 'azure_devops' in scope.sources:
                try:
                    with st.spinner('Saving new and changed work items to project knowledge...'):
                        show_sync(index.sync([story_document(item, scope) for item in items], 'azure_devops'), 'Sprint stories')
                except Exception as error:
                    st.warning('Stories could not be saved to knowledge. ' + safe_error(error))
        loaded = st.session_state.get('engineering_stories', {})
        if loaded.get('scope') == scope.key and iteration and loaded.get('iteration') == iteration['id']:
            search = st.text_input('Search stories by ID or title')
            stories = [s for s in loaded['items'] if search.lower() in f'{s["id"]} {s["fields"].get("System.Title", "")}'.lower()]
            selected = st.selectbox('Story', stories, format_func=lambda s: f'{s["id"]}: {s["fields"].get("System.Title")}', index=None)
            if selected:
                story = azure.story(project['id'], selected['id'])
                show_story(story)
                doc = story_document(story, scope)
                story_key = f'{scope.key}:{st.session_state["engineering_identity"]}:story:{story["id"]}:{story.get("rev", "")} '
                saved = ensure_indexed(index, doc, story_key, 'Story')
                context, retrieval_ready = [], False
                if saved:
                    try:
                        signature = index.signature()
                        cache = st.session_state.get('engineering_retrieval', {})
                        if cache.get('story') == story_key and cache.get('signature') == signature:
                            context = cache['items']
                            retrieval_ready = True
                        elif not index.has_other_documents('azure_devops', story['id']):
                            st.session_state['engineering_retrieval'] = {'story': story_key, 'signature': signature, 'items': []}
                            retrieval_ready = True
                        else:
                            failures = st.session_state.setdefault('engineering_retrieval_failures', {})
                            failure_key = story_key + signature
                            retry_key = 'retry_retrieval_' + sha256(failure_key.encode()).hexdigest()
                            retry = True
                            retry_rendered = False
                            if failure_key in failures:
                                st.warning('Related knowledge could not be retrieved. ' + failures[failure_key])
                                retry_rendered = True
                                retry = st.button('Retry related knowledge retrieval', key=retry_key)
                            if retry:
                                try:
                                    # Current requirements are already supplied separately to the model.
                                    context = [item for item in index.retrieve(story, k=12)
                                               if not (item['metadata'].get('source_type') == 'azure_devops'
                                                       and str(item['metadata'].get('work_item_id')) == str(story['id']))][:8]
                                    st.session_state['engineering_retrieval'] = {'story': story_key, 'signature': signature, 'items': context}
                                    failures.pop(failure_key, None)
                                    retrieval_ready = True
                                except Exception as error:
                                    failures[failure_key] = safe_error(error)
                                    st.warning('Related knowledge could not be retrieved. ' + failures[failure_key])
                                    if not retry_rendered:
                                        st.button('Retry related knowledge retrieval', key=retry_key)
                    except Exception as error:
                        st.warning('Related knowledge could not be retrieved. ' + safe_error(error))
                if st.button('Load discussion history'):
                    st.session_state['engineering_comments'] = {'story': story_key, 'items': azure.comments(project['id'], story['id'])}
                comments = st.session_state.get('engineering_comments', {})
                if comments.get('story') == story_key:
                    with st.expander('Discussion history', expanded=True):
                        show_comments(comments['items'])
                show_evidence(context)
                if not saved or not retrieval_ready:
                    st.info('Save the story and restore knowledge retrieval before generating test cases.')
                if st.button('Generate manual test cases', disabled=not (saved and retrieval_ready)):
                    with st.spinner('Generating and validating coverage…'):
                        draft = generate_tests(llm_for(controls), story, context)
                        st.session_state['engineering_workflow'] = store.save(story, context, draft.model_dump())
                    st.success('Draft saved. Open Review and publish.')
    with tabs[1]:
        statuses = index.status()
        if statuses:
            st.dataframe([{'Source': status['source'].replace('_', ' ').title(), 'Status': status['status'].title(),
                           'Documents': status['count'], 'Last updated': status['updated_at']} for status in statuses],
                         hide_index=True, width='stretch')
        else:
            st.info('No knowledge sources have been indexed yet.')
        force = st.checkbox('Force re-index synchronized documents')
        st.caption('Unchanged documents are not re-embedded. Full snapshots remove missing or newly inaccessible documents only after a successful complete fetch.')
        if 'azure_devops' in scope.sources and st.button('Sync project history (full snapshot)'):
            ids = azure.query_ids(project['name'])
            history = azure.work_items(project['id'], ids)
            show_sync(index.sync([story_document(s, scope) for s in history], 'azure_devops', full_snapshot=True, force=force), 'Azure DevOps history')
        with st.expander('Confluence Cloud ingestion'):
            site = st.text_input('Confluence site URL')
            email = st.text_input('Confluence email')
            token = st.text_input('Confluence API token', type='password', key='engineering_confluence_token')
            space = st.text_input('Space ID (optional; blank syncs all accessible pages)')
            if st.button('Sync Confluence pages'):
                connector = Confluence(site, email, SecretStr(token))
                try:
                    # Space-specific batches are incremental to avoid deleting other indexed spaces.
                    show_sync(index.sync(confluence_documents(connector, scope, space or None), 'confluence',
                                         full_snapshot=not bool(space), partition=connector.http.base, force=force), 'Confluence pages')
                finally:
                    connector.close()
        uploads = st.file_uploader('Add business documents',
                                   type=['pdf', 'docx', 'xlsx', 'csv', 'txt', 'md', 'ts', 'js', 'py', 'feature', 'json'],
                                   accept_multiple_files=True)
        if uploads:
            names = [Path(upload.name).name for upload in uploads]
            if len(names) != len(set(name.casefold() for name in names)):
                st.warning('Upload files with different names so each document can be tracked separately.')
            else:
                updated = False
                for upload in uploads:
                    contents = upload.getvalue()
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / Path(upload.name).name
                        path.write_bytes(contents)
                        for doc in file_documents(path, scope):
                            if not doc.page_content.strip():
                                st.warning(f'{path.name} has no readable text. Upload a searchable document to add it to knowledge.')
                                continue
                            key = f'{scope.key}:file:{path.name}:{sha256(contents).hexdigest()}'
                            try:
                                was_current = index.is_current(doc)
                            except Exception:
                                was_current = False
                            if ensure_indexed(index, doc, key, f'Document {path.name}'):
                                updated = updated or not was_current
                                st.caption(f'{path.name} is available in project knowledge.')
                if updated:
                    st.rerun()
    with tabs[2]:
        _render_review(store, index, azure, project, scope, controls)
    # with tabs[3]:
    #     _automation(workflow, root / scope.key / 'automation', controls)
    # with tabs[4]:
    #     _evolution(controls)


def _render_review(store, index, azure, project, scope, controls):
    summaries = []
    for identifier in store.list():
        try:
            summaries.append(store.load(identifier))
        except PermissionError:
            continue
    if not summaries:
        st.info('Generate test cases for a story to review them here.')
        return None
    preferred = st.session_state.get('engineering_workflow')
    index_value = next((i for i, item in enumerate(summaries) if item['id'] == preferred), 0)
    workflow = st.selectbox('Test case set', summaries, index=index_value,
        format_func=lambda item: f"Story {item['story']['id']}: {item['story']['fields'].get('System.Title', 'Untitled')} - revision {item['revision']}",
        key=f'engineering_review_selection_{scope.key}_{len(summaries)}')
    identifier = workflow['id']
    st.session_state['engineering_workflow'] = identifier
    status = 'Approved' if workflow['approved'] else 'Needs review'
    st.caption(f"Story {workflow['story']['id']} - Revision {workflow['revision']} - {status}")
    st.write('Review and edit the generated test cases before uploading them to Azure DevOps.')
    edited = edit_draft(workflow, scope.key)
    if st.button('Save test case changes'):
        validate_editor_fields(edited)
        validated = validate_draft(edited, workflow['story'], workflow['context']).model_dump()
        if validated != workflow['draft']:
            store.save(workflow['story'], workflow['context'], validated, identifier, expected_revision=workflow['revision'])
            st.rerun()
        st.info('There are no changes to save.')
    with st.expander('Add a test case'):
        prefix = f"new_{scope.key}_{identifier}_{workflow['revision']}"
        title = st.text_input('New test case title', key=prefix + '_title')
        description = st.text_area('New test case description', key=prefix + '_description')
        action = st.text_area('First step action', key=prefix + '_action')
        test_data = st.text_input('First step test data', key=prefix + '_data')
        expected = st.text_area('First step expected result', key=prefix + '_expected')
        criteria = criteria_for(workflow['story'])
        covered = st.multiselect('Criteria covered by the new case', list(criteria),
                                 format_func=lambda item: f'{item}: {criteria[item]}', key=prefix + '_criteria')
        if st.button('Add test case'):
            new_case = {'title': title.strip(), 'description': description.strip(), 'story_id': workflow['story']['id'],
                        'preconditions': [], 'steps': [{'step_number': 1, 'action': action.strip(),
                        'test_data': test_data.strip(), 'expected_result': expected.strip()}], 'priority': 2,
                        'category': 'Functional', 'acceptance_criteria_reference': covered,
                        'rag_source_references': [], 'automation_feasibility': 'Medium'}
            edited['test_cases'].append(new_case)
            validate_editor_fields(edited)
            store.save(workflow['story'], workflow['context'], edited, identifier, expected_revision=workflow['revision'])
            st.rerun()
    with st.expander('Regenerate selected test cases'):
        titles = [case['title'] for case in edited['test_cases'] if case['title']]
        selected_titles = st.multiselect('Cases to regenerate', titles, key=f'regenerate_{identifier}_{workflow["revision"]}')
        instruction = st.text_input('Instructions for regeneration', key=f'regenerate_instruction_{identifier}_{workflow["revision"]}')
        if st.button('Regenerate selected cases'):
            validate_editor_fields(edited)
            validate_draft(edited, workflow['story'], workflow['context'])
            draft = regenerate_selected(llm_for(controls), workflow['story'], workflow['context'], edited,
                                        selected_titles, instruction)
            store.save(workflow['story'], workflow['context'], draft.model_dump(), identifier, expected_revision=workflow['revision'])
            st.rerun()
    with st.expander('Revision history'):
        st.dataframe([{'Revision': item['revision'], 'Status': 'Approved' if item['approved'] else 'Needs review',
                       'Saved at': item['created_at']} for item in store.history(identifier)], hide_index=True, width='stretch')
    if st.button('Approve without uploading'):
        validate_editor_fields(edited)
        approve_review(store, identifier, workflow['revision'], workflow['digest'], edited, explicit=True)
        st.rerun()
    if workflow['approved'] and st.button('Add approved cases to project knowledge'):
        from langchain_core.documents import Document
        doc = Document(page_content=json.dumps(workflow['draft']), metadata={
            'organization_id': scope.organization_id, 'project_id': scope.project_id,
            'source_type': 'approved_tests', 'source_document_id': identifier,
            'work_item_id': str(workflow['story']['id']), 'title': 'Approved tests: ' + str(workflow['story']['id']),
            'version': str(workflow['revision']), 'evidence_type': 'historical_approved_tests'})
        show_sync(index.sync([doc], 'approved_tests'), 'Approved test cases')
    plan_key = f'engineering_plan_id_{scope.key}_{identifier}'
    suite_key = f'engineering_suite_id_{scope.key}_{identifier}'
    with st.expander('Find or create a test plan and suite'):
        if st.button('Load available test plans'):
            st.session_state['engineering_plans'] = {'scope': scope.key, 'items': azure.plans(project['id'])}
        plans = st.session_state.get('engineering_plans', {})
        plan = pick('Available test plan', plans.get('items', []), f'engineering_plan_{scope.key}') if plans.get('scope') == scope.key else None
        suite = pick('Available test suite', azure.suites(project['id'], plan['id']),
                     f'engineering_suite_{scope.key}_{plan["id"]}') if plan else None
        if plan and suite and st.button('Use selected plan and suite'):
            st.session_state[plan_key] = str(plan['id'])
            st.session_state[suite_key] = str(suite['id'])
        plan_name = st.text_input('New plan name')
        if st.button('Create test plan'):
            created = azure.create_plan(project['id'], plan_name)
            st.session_state[plan_key] = str(created['id'])
            st.success(f"Created test plan {created.get('name', plan_name)} (ID {created['id']}).")
        suite_name = st.text_input('New suite name')
        parent_suite = st.number_input('Parent suite ID', min_value=1, step=1)
        if plan and st.button('Create test suite'):
            created = azure.create_suite(project['id'], plan['id'], parent_suite, suite_name)
            st.session_state[plan_key] = str(plan['id'])
            st.session_state[suite_key] = str(created['id'])
            st.success(f"Created test suite {created.get('name', suite_name)} (ID {created['id']}).")
    st.subheader('Upload to Azure DevOps')
    st.write(f"Project: {project['name']} ({project['id']})")
    first, second = st.columns(2)
    with first:
        plan_id = st.text_input('Test Plan ID', key=plan_key, placeholder='For example: 123', max_chars=10)
    with second:
        suite_id = st.text_input('Test Suite ID', key=suite_key, placeholder='For example: 456', max_chars=10)
    ids = [plan_id.strip(), suite_id.strip()]
    ready = all(value.isascii() and value.isdigit() and 1 <= int(value) <= 2147483647 for value in ids)
    st.caption('Clicking OK approves the displayed cases and uploads them to the entered plan and suite.')
    if st.button('OK - approve and upload', type='primary', disabled=not ready):
        validate_editor_fields(edited)
        results = approve_and_publish(Publisher(azure, store, project['id']), identifier,
                                      workflow['revision'], workflow['digest'], edited,
                                      plan_id.strip(), suite_id.strip(), explicit=True)
        st.session_state['engineering_last_publication'] = {'scope': scope.key, 'workflow': identifier,
                                                            'plan': plan_id, 'suite': suite_id, 'items': results}
        st.rerun()
    publication = st.session_state.get('engineering_last_publication', {})
    if publication.get('scope') == scope.key and publication.get('workflow') == identifier:
        st.caption(f"Uploaded to plan {publication['plan']}, suite {publication['suite']}.")
        show_results(publication['items'])
    return workflow


# def _automation(workflow, root, controls):
#     st.caption('Generated application files live in a separate workspace. Review code before approving local execution.')
#     if not workflow or not workflow['approved']:
#         st.info('Select and approve a manual test workflow first.')
#         return
#     url = st.text_input('Application URL')
#     inspection_text = st.text_area('Page details or observed controls (optional)')
#     inspection_files = sorted(path for path in root.glob('*/reports/inspection.json')
#                               if path.is_file() and path.resolve().is_relative_to(root.resolve())) if root.exists() else []
#     inspection_choice = st.selectbox('Saved browser inspection', [None, *inspection_files],
#                                       format_func=lambda path: path.parent.parent.name if path else 'None')
#     screenshots = st.file_uploader('Application screenshots', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
#     conventions = st.text_area('Existing framework conventions or selected reusable source')
#     repository = st.text_input('Existing automation repository path (optional)')
#     if st.button('Generate automation proposal'):
#         if repository:
#             from src.agents.framework_evolution import inspect_framework
#             conventions += '\nExisting automation source: ' + json.dumps(inspect_framework(repository))
#         inspection = {'notes': inspection_text}
#         if inspection_choice:
#             inspection['browser_inspection'] = json.loads(inspection_choice.read_text(encoding='utf-8'))
#         proposal = generate_automation(llm_for(controls), workflow, inspection,
#                                        [(f.type, f.getvalue()) for f in screenshots], conventions)
#         st.session_state['engineering_automation_proposal'] = {'workflow': workflow['id'], 'digest': workflow['digest'],
#             'proposal': proposal.model_dump() if hasattr(proposal, 'model_dump') else proposal}
#     saved = st.session_state.get('engineering_automation_proposal', {})
#     proposal = saved.get('proposal')
#     if proposal and saved.get('workflow') == workflow['id'] and saved.get('digest') == workflow['digest']:
#         st.markdown('**Review the proposed automation**')
#         proposal_key = sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:16]
#         edited = editable_proposal(proposal, f'automation_{workflow["id"]}_{proposal_key}')
#         if st.button('Create isolated automation workspace'):
#             for file in edited['files']:
#                 if file['path'].lower().endswith('.json'):
#                     try:
#                         json.loads(file['content'])
#                     except ValueError:
#                         raise ValueError('Generated test data is invalid; regenerate the automation proposal') from None
#             workspace = AutomationWorkspace(root)
#             workspace.write(AutomationProposal.model_validate(edited), workflow)
#             st.session_state['engineering_automation'] = workspace.path.name
#     directories = sorted([p.name for p in root.iterdir() if p.is_dir()]) if root.exists() else []
#     preferred = st.session_state.get('engineering_automation')
#     name = st.selectbox('Automation workspace', directories, index=directories.index(preferred) if preferred in directories else 0) if directories else None
#     if not name:
#         return
#     workspace = AutomationWorkspace(root, name)
#     manifest = json.loads((workspace.path / 'manifest.json').read_text(encoding='utf-8'))
#     if manifest['workflow_id'] != workflow['id'] or manifest['draft_digest'] != workflow['digest']:
#         st.info('This workspace belongs to another manual test revision. Select its approved workflow or generate a new proposal.')
#         return
#     st.caption('Workspace: ' + str(workspace.path))
#     st.write(manifest.get('summary', 'Automation workspace'))
#     for assumption in manifest.get('assumptions', []):
#         st.write('Assumption: ' + assumption)
#     for evidence in manifest.get('locator_evidence', []):
#         st.write('Locator evidence: ' + evidence)
#     for file in sorted(workspace.path.rglob('*')):
#         if (file.is_file() and file.name not in {'manifest.json', 'package.json', 'tsconfig.json', '.approval.json', 'package-lock.json'}
#                 and not any(p in {'node_modules', 'reports'} for p in file.relative_to(workspace.path).parts)):
#             with st.expander(file.relative_to(workspace.path).as_posix()):
#                 if file.suffix.lower() == '.json':
#                     try:
#                         show_details(json.loads(file.read_text(encoding='utf-8')))
#                     except ValueError:
#                         st.warning('This test data file is invalid.')
#                 else:
#                     st.code(file.read_text(encoding='utf-8'))
#     digest = workspace.digest()
#     st.caption('Review digest: ' + digest)
#     if st.button('Approve displayed automation files'):
#         workspace.approve(digest, explicit=True)
#         st.success('Approved. Dependency installation changes the digest; review and approve again afterward.')
#     action = st.selectbox('Controlled action', ['install', 'browsers', 'inspect', 'check', 'validate', 'test', 'check + validate + test'])
#     execute = st.checkbox('I authorize this action on the reviewed code in my controlled local environment', key=f'execute_{name}_{digest}_{action}')
#     if st.button('Run selected action', disabled=not execute):
#         with st.spinner('Running approved action…'):
#             result = workspace.validate_and_test(url, explicit=True) if action == 'check + validate + test' else workspace.run(action, url, explicit=True)
#         for report in result if isinstance(result, list) else [result]:
#             status = 'Passed' if report['returncode'] == 0 else 'Failed'
#             st.write(f"**{report['action'].title()}: {status}**")
#             if report.get('stdout'):
#                 with st.expander('Output'):
#                     st.code(report['stdout'], language='text')
#             if report.get('stderr'):
#                 with st.expander('Errors'):
#                     st.code(report['stderr'], language='text')
#     for file in sorted((workspace.path / 'reports').glob('*')):
#         if file.suffix.lower() != '.json':
#             st.download_button('Download ' + file.name, file.read_bytes(), file_name=file.name, key=f'download_{name}_{file.name}')
#     if st.button('Analyze execution failures and propose corrections'):
#         reports = [json.loads(f.read_text(encoding='utf-8')) for f in (workspace.path / 'reports').glob('*.json')]
#         st.write(analyze_failures(llm_for(controls), reports))


# def _evolution(controls):
#     repository = st.text_input('Framework repository path')
#     automation = st.text_area('Generated automation to analyze')
#     reports = st.text_area('Execution report to analyze')
#     if st.button('Propose reusable framework improvements'):
#         proposal = propose_evolution(llm_for(controls), repository, automation, reports)
#         st.session_state['engineering_evolution'] = build_diff(repository, proposal)
#     review = st.session_state.get('engineering_evolution')
#     if not review:
#         return
#     proposal = review['proposal']
#     st.subheader(proposal['title'])
#     st.write(proposal['rationale'])
#     for pattern in proposal.get('existing_patterns', []):
#         st.write('Existing pattern: ' + pattern)
#     for note in proposal.get('regression_notes', []):
#         st.write('Regression note: ' + note)
#     for file in proposal['files']:
#         st.write('Proposed file: ' + file['path'])
#     st.code(review['diff'], language='diff')
#     st.caption('Review digest: ' + review['digest'])
#     branch = st.text_input('New branch', 'ai-evolution/reusable-components')
#     destination = st.text_input('New worktree path (outside repository)')
#     confirm = st.checkbox('I approve this diff and creating a separate review branch')
#     if st.button('Create approved review branch', disabled=not confirm):
#         st.session_state['engineering_evolution_branch'] = create_review_branch(repository, review, review['digest'], branch, destination, explicit=confirm)
#         created = st.session_state['engineering_evolution_branch']
#         st.success(f"Created branch {created['branch']} in {created['worktree']}.")
#     branch_info = st.session_state.get('engineering_evolution_branch')
#     if branch_info and branch_info['digest'] == review['digest']:
#         run = st.checkbox('I approve executing the proposed regression tests')
#         publish = st.checkbox('After passing regressions, I approve committing, pushing and creating a draft pull request')
#         if st.button('Run regression review', disabled=not run):
#             result = regression_and_pr(branch_info['worktree'], review, explicit=run, create_pr=publish)
#             st.write('Regression checks ' + ('passed.' if result['returncode'] == 0 else 'failed.'))
#             if result.get('pull_request'):
#                 st.write('Pull request: ' + result['pull_request'])
#             for name in ('stdout', 'stderr'):
#                 if result.get(name):
#                     with st.expander('Regression ' + ('output' if name == 'stdout' else 'errors')):
#                         st.code(result[name], language='text')

"""Streamlit interface for the test engineering platform."""
import json
import os
from pathlib import Path
import tempfile

import streamlit as st
from pydantic import SecretStr

from src.integrations.azure_devops import AzureDevOps
from src.integrations.confluence import Confluence
from src.LLMS.groqllm import GroqLLM
from src.RAG.Rag import Rag
from src.test_engineering.knowledge import Scope, KnowledgeIndex, story_document, confluence_documents, file_documents
from src.test_engineering.workflow import WorkflowStore, generate_tests, regenerate_selected, criteria_for
from src.test_engineering.publishing import Publisher
from src.automation.workspace import AutomationWorkspace, AutomationProposal, generate_automation, analyze_failures
from src.agents.framework_evolution import propose_evolution, build_diff, create_review_branch, regression_and_pr


def pick(label, items, key):
    if not items:
        st.info(f'No accessible {label.lower()} available.')
        return None
    return st.selectbox(label, items, format_func=lambda item: f'{item.get("name", item.get("title", ""))} ({item["id"]})', key=key)


def model_controls():
    from src.ui.uiconfigfile import Config
    model_name = st.selectbox('Generation model', Config().get_groq_model_options())
    key = st.text_input('Groq API key', type='password', key='engineering_groq_key')
    return {'GROQ_API_KEY': key, 'selected_groq_model': model_name}


def llm_for(controls):
    llm = GroqLLM(controls).get_llm_model()
    if llm is None:
        raise ValueError('Configure a Groq API key first')
    return llm


def render():
    st.set_page_config(page_title='AI Test Engineering', layout='wide')
    st.title('AI Test Engineering')
    st.caption('Existing LangChain RAG · Azure DevOps · Confluence · reviewed automation')
    try:
        _render()
    except Exception as error:
        # Provider exceptions may contain request data. Display only our known-safe errors.
        if type(error) in (ValueError, PermissionError) and error.__class__.__module__ == 'builtins':
            st.error(str(error))
        elif error.__class__.__name__ == 'IntegrationError':
            st.error(str(error))
        else:
            st.error(f'{type(error).__name__}: operation failed. Check service configuration, permissions and input schema.')


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
    if st.button('Discover other accessible organizations (where PAT policy permits)'):
        st.json(azure.accounts())
    project = pick('Project', st.session_state['engineering_projects'], 'engineering_project')
    if not project:
        return
    st.caption(project.get('description') or 'No project description')
    azure.authorize_project(project['id'])
    allowed = frozenset(s.strip() for s in os.getenv('ENGINEERING_ALLOWED_SOURCES', 'azure_devops,confluence,files,approved_tests').split(',') if s.strip())
    scope = Scope(azure.organization_id, project['id'], allowed)
    from hashlib import sha256
    # Separate authenticated identities, in addition to organization/project/source scope.
    root = Path(os.getenv('ENGINEERING_DATA_DIR', '.local/test-engineering')) / sha256(st.session_state['engineering_identity'].encode()).hexdigest()
    rag = Rag()
    index = KnowledgeIndex(root, scope, rag, os.getenv('ENGINEERING_EMBEDDING_VERSION', 'google/embeddinggemma-300m:v1'))
    store = WorkflowStore(root, scope)
    controls = model_controls()
    tabs = st.tabs(['Stories', 'Knowledge', 'Review and publish', 'Automation', 'Framework evolution'])
    with tabs[0]:
        team = pick('Team', azure.teams(project['id']), 'engineering_team')
        iteration = pick('Sprint', azure.iterations(project['id'], team['id']), 'engineering_iteration') if team else None
        types = st.text_input('Work item types (comma separated)', 'User Story,Product Backlog Item,Feature,Epic,Bug,Task')
        if iteration and st.button('Retrieve sprint work items'):
            ids = azure.query_ids(project['name'], iteration['path'], [t.strip() for t in types.split(',') if t.strip()])
            st.session_state['engineering_stories'] = {'scope': scope.key, 'iteration': iteration['id'], 'items': azure.work_items(project['id'], ids)}
        loaded = st.session_state.get('engineering_stories', {})
        if loaded.get('scope') == scope.key and iteration and loaded.get('iteration') == iteration['id']:
            search = st.text_input('Search stories by ID or title')
            stories = [s for s in loaded['items'] if search.lower() in f'{s["id"]} {s["fields"].get("System.Title", "")}'.lower()]
            selected = st.selectbox('Story', stories, format_func=lambda s: f'{s["id"]}: {s["fields"].get("System.Title")}', index=None)
            if selected:
                # Each selection retrieves current requirements and invokes the shared index; never re-ingests.
                story = azure.story(project['id'], selected['id'])
                context = index.retrieve(story)
                st.json(story)
                st.write('Acceptance criteria', criteria_for(story))
                if st.button('Load discussion history'):
                    st.json(azure.comments(project['id'], story['id']))
                st.write('Related work items and attachments', story.get('relations', []))
                st.subheader('Retrieved evidence')
                for item in context:
                    with st.expander(item['metadata'].get('title', 'Source')):
                        st.json({key: value for key, value in item.items() if key != 'content'})
                        st.text(item['content'])
                if not context:
                    st.info('No indexed context yet. Sync knowledge to add historical evidence.')
                if st.button('Generate manual test cases'):
                    with st.spinner('Generating and validating coverage…'):
                        draft = generate_tests(llm_for(controls), story, context)
                        st.session_state['engineering_workflow'] = store.save(story, context, draft.model_dump())
                    st.success('Draft saved. Open Review and publish.')
    with tabs[1]:
        st.write('Ingestion status', index.status())
        force = st.checkbox('Force re-index synchronized documents')
        st.caption('Unchanged documents are not re-embedded. Full snapshots remove missing or newly inaccessible documents only after a successful complete fetch.')
        if 'azure_devops' in scope.sources and st.button('Sync project history (full snapshot)'):
            ids = azure.query_ids(project['name'])
            history = azure.work_items(project['id'], ids)
            st.json(index.sync([story_document(s, scope) for s in history], 'azure_devops', full_snapshot=True, force=force))
        with st.expander('Confluence Cloud ingestion'):
            site = st.text_input('Confluence site URL')
            email = st.text_input('Confluence email')
            token = st.text_input('Confluence API token', type='password', key='engineering_confluence_token')
            space = st.text_input('Space ID (optional; blank syncs all accessible pages)')
            if st.button('Sync Confluence pages'):
                connector = Confluence(site, email, SecretStr(token))
                try:
                    # Space-specific batches are incremental to avoid deleting other indexed spaces.
                    st.json(index.sync(confluence_documents(connector, scope, space or None), 'confluence',
                                       full_snapshot=not bool(space), partition=connector.http.base, force=force))
                finally:
                    connector.close()
        upload = st.file_uploader('Business documents or automation source', type=['pdf', 'docx', 'xlsx', 'csv', 'txt', 'md', 'ts', 'js', 'py', 'feature', 'json'])
        if upload and st.button('Index uploaded document'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / Path(upload.name).name
                path.write_bytes(upload.getvalue())
                st.json(index.sync(file_documents(path, scope), 'files', force=force))
    workflow = None
    with tabs[2]:
        workflows = store.list()
        preferred = st.session_state.get('engineering_workflow')
        identifier = st.selectbox('Resume workflow', workflows, index=workflows.index(preferred) if preferred in workflows else 0) if workflows else None
        if identifier:
            st.session_state['engineering_workflow'] = identifier
            workflow = store.load(identifier)
            st.caption(f'Revision {workflow["revision"]} · Approved: {workflow["approved"]} · Story {workflow["story"]["id"]}')
            st.write('Would you like to review or make any changes to these generated test cases before publishing them to Azure DevOps?')
            edited = st.text_area('Editable test cases JSON (edit, add or remove cases and steps)', json.dumps(workflow['draft'], indent=2), height=420,
                                  key=f'draft_{scope.key}_{identifier}_{workflow["revision"]}')
            if st.button('Save reviewed revision'):
                store.save(workflow['story'], workflow['context'], json.loads(edited), identifier)
                st.rerun()
            instruction = st.text_input('Regeneration instructions (identify scenarios to replace or preserve)')
            selected_titles = st.multiselect('Scenarios to regenerate', [case['title'] for case in workflow['draft']['test_cases']])
            if st.button('Regenerate with review instructions'):
                draft = regenerate_selected(llm_for(controls), workflow['story'], workflow['context'],
                                            workflow['draft'], selected_titles, instruction)
                store.save(workflow['story'], workflow['context'], draft.model_dump(), identifier)
                st.rerun()
            st.json(store.history(identifier))
            if st.button('Approve this saved revision'):
                if json.loads(edited) != workflow['draft']:
                    raise ValueError('Save your edits before approving')
                store.approve(identifier, workflow['revision'], workflow['digest'], explicit=True)
                st.rerun()
            if workflow['approved']:
                if st.button('Index approved cases as reusable evidence'):
                    from langchain_core.documents import Document
                    doc = Document(page_content=json.dumps(workflow['draft']), metadata={
                        'organization_id': scope.organization_id, 'project_id': scope.project_id,
                        'source_type': 'approved_tests', 'source_document_id': identifier,
                        'work_item_id': str(workflow['story']['id']), 'title': 'Approved tests: ' + str(workflow['story']['id']),
                        'version': str(workflow['revision']), 'evidence_type': 'historical_approved_tests'})
                    st.json(index.sync([doc], 'approved_tests'))
                if st.button('Load Test Plans'):
                    st.session_state['engineering_plans'] = {'scope': scope.key, 'items': azure.plans(project['id'])}
                plans = st.session_state.get('engineering_plans', {})
                plan = pick('Test Plan', plans.get('items', []), 'engineering_plan') if plans.get('scope') == scope.key else None
                with st.expander('Create plan or static suite'):
                    plan_name = st.text_input('New plan name')
                    if st.button('Create Test Plan'):
                        st.json(azure.create_plan(project['id'], plan_name))
                    suite_name = st.text_input('New suite name')
                    parent_suite = st.number_input('Parent Suite ID', min_value=1, step=1)
                    if plan and st.button('Create Test Suite'):
                        st.json(azure.create_suite(project['id'], plan['id'], parent_suite, suite_name))
                if plan:
                    suite = pick('Test Suite', azure.suites(project['id'], plan['id']), 'engineering_suite')
                    if suite:
                        st.write({'Project ID': project['id'], 'Plan': plan['name'], 'Plan ID': plan['id'], 'Suite': suite['name'], 'Suite ID': suite['id']})
                        confirm = st.checkbox('I approve publishing this saved revision to the displayed project, plan and suite',
                                              key=f'publish_{scope.key}_{identifier}_{workflow["revision"]}_{plan["id"]}_{suite["id"]}')
                        if st.button('Publish approved test cases', disabled=not confirm):
                            if json.loads(edited) != workflow['draft']:
                                raise ValueError('Save and approve edited cases before publishing')
                            st.json(Publisher(azure, store, project['id']).publish(identifier, plan['id'], suite['id'], explicit=confirm, expected_digest=workflow['digest']))
    with tabs[3]:
        _automation(workflow, root / scope.key / 'automation', controls)
    with tabs[4]:
        _evolution(controls)


def _automation(workflow, root, controls):
    st.caption('Generated application files live in a separate workspace. Review code before approving local execution.')
    if not workflow or not workflow['approved']:
        st.info('Select and approve a manual test workflow first.')
        return
    url = st.text_input('Application URL')
    inspection_text = st.text_area('DOM/accessibility inspection JSON (optional)', '{}')
    screenshots = st.file_uploader('Application screenshots', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    conventions = st.text_area('Existing framework conventions or selected reusable source')
    repository = st.text_input('Existing automation repository path (optional)')
    if st.button('Generate automation proposal'):
        if repository:
            from src.agents.framework_evolution import inspect_framework
            conventions += '\nExisting automation source: ' + json.dumps(inspect_framework(repository))
        proposal = generate_automation(llm_for(controls), workflow, json.loads(inspection_text),
                                       [(f.type, f.getvalue()) for f in screenshots], conventions)
        st.session_state['engineering_automation_proposal'] = {'workflow': workflow['id'], 'digest': workflow['digest'],
            'proposal': proposal.model_dump() if hasattr(proposal, 'model_dump') else proposal}
    saved = st.session_state.get('engineering_automation_proposal', {})
    proposal = saved.get('proposal')
    if proposal and saved.get('workflow') == workflow['id'] and saved.get('digest') == workflow['digest']:
        edited = st.text_area('Review automation proposal JSON', json.dumps(proposal, indent=2), height=300)
        if st.button('Create isolated automation workspace'):
            workspace = AutomationWorkspace(root)
            workspace.write(AutomationProposal.model_validate_json(edited), workflow)
            st.session_state['engineering_automation'] = workspace.path.name
    directories = sorted([p.name for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    preferred = st.session_state.get('engineering_automation')
    name = st.selectbox('Automation workspace', directories, index=directories.index(preferred) if preferred in directories else 0) if directories else None
    if not name:
        return
    workspace = AutomationWorkspace(root, name)
    manifest = json.loads((workspace.path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['workflow_id'] != workflow['id'] or manifest['draft_digest'] != workflow['digest']:
        st.info('This workspace belongs to another manual test revision. Select its approved workflow or generate a new proposal.')
        return
    st.code(str(workspace.path))
    st.json(manifest)
    for file in sorted(workspace.path.rglob('*')):
        if file.is_file() and not any(p in {'node_modules', 'reports'} for p in file.relative_to(workspace.path).parts):
            with st.expander(file.relative_to(workspace.path).as_posix()):
                st.code(file.read_text(encoding='utf-8'))
    digest = workspace.digest()
    st.caption('Review digest: ' + digest)
    if st.button('Approve displayed automation files'):
        workspace.approve(digest, explicit=True)
        st.success('Approved. Dependency installation changes the digest; review and approve again afterward.')
    action = st.selectbox('Controlled action', ['install', 'browsers', 'inspect', 'check', 'validate', 'test', 'check + validate + test'])
    execute = st.checkbox('I authorize this action on the reviewed code in my controlled local environment', key=f'execute_{name}_{digest}_{action}')
    if st.button('Run selected action', disabled=not execute):
        with st.spinner('Running approved action…'):
            result = workspace.validate_and_test(url, explicit=True) if action == 'check + validate + test' else workspace.run(action, url, explicit=True)
        st.json(result)
    for file in sorted((workspace.path / 'reports').glob('*')):
        st.download_button('Download ' + file.name, file.read_bytes(), file_name=file.name, key=f'download_{name}_{file.name}')
    if st.button('Analyze execution failures and propose corrections'):
        reports = [json.loads(f.read_text(encoding='utf-8')) for f in (workspace.path / 'reports').glob('*.json')]
        st.write(analyze_failures(llm_for(controls), reports))


def _evolution(controls):
    repository = st.text_input('Framework repository path')
    automation = st.text_area('Generated automation to analyze')
    reports = st.text_area('Execution report to analyze')
    if st.button('Propose reusable framework improvements'):
        proposal = propose_evolution(llm_for(controls), repository, automation, reports)
        st.session_state['engineering_evolution'] = build_diff(repository, proposal)
    review = st.session_state.get('engineering_evolution')
    if not review:
        return
    st.json(review['proposal'])
    st.code(review['diff'], language='diff')
    st.caption('Review digest: ' + review['digest'])
    branch = st.text_input('New branch', 'ai-evolution/reusable-components')
    destination = st.text_input('New worktree path (outside repository)')
    confirm = st.checkbox('I approve this diff and creating a separate review branch')
    if st.button('Create approved review branch', disabled=not confirm):
        st.session_state['engineering_evolution_branch'] = create_review_branch(repository, review, review['digest'], branch, destination, explicit=confirm)
        st.json(st.session_state['engineering_evolution_branch'])
    branch_info = st.session_state.get('engineering_evolution_branch')
    if branch_info and branch_info['digest'] == review['digest']:
        run = st.checkbox('I approve executing the proposed regression tests')
        publish = st.checkbox('After passing regressions, I approve committing, pushing and creating a draft pull request')
        if st.button('Run regression review', disabled=not run):
            st.json(regression_and_pr(branch_info['worktree'], review, explicit=run, create_pr=publish))

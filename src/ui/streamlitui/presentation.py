"""Human-readable Streamlit views and editors for test engineering."""
import copy
# from hashlib import sha256  # Used by the disabled automation editor.
# import json  # Used by the disabled automation editor.
import math
import re
from urllib.parse import urlsplit

import streamlit as st

from src.test_engineering.knowledge import plain
from src.test_engineering.workflow import criteria_for


def text_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    return str(value).strip()


def human_label(value):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str(value)).replace('_', ' ').capitalize()


def show_details(value, prefix=''):
    """Render nested service results as labeled text and tables, never raw JSON."""
    if isinstance(value, dict):
        for key, item in value.items():
            if item is None or item == '' or item == [] or item == {}:
                continue
            label = human_label(key)
            label = f'{prefix} / {label}' if prefix else label
            if isinstance(item, (dict, list)):
                show_details(item, label)
            else:
                st.write(f'**{label}:** {text_value(item)}')
    elif isinstance(value, list):
        if not value:
            st.caption(f'{prefix or "Items"}: none')
        elif all(not isinstance(item, (dict, list)) for item in value):
            for item in value:
                st.write(f'**{prefix}:** {text_value(item)}' if prefix else f'- {text_value(item)}')
        else:
            for number, item in enumerate(value, 1):
                show_details(item, f'{prefix} / {number}' if prefix else f'Item {number}')
    else:
        st.write(text_value(value))


def show_story(story):
    fields = story.get('fields', {})
    st.subheader(f"{story['id']}: {fields.get('System.Title', 'Untitled story')}")
    assigned = fields.get('System.AssignedTo') or {}
    if isinstance(assigned, dict):
        assigned = assigned.get('displayName') or assigned.get('uniqueName') or 'Unassigned'
    st.dataframe([{'Type': fields.get('System.WorkItemType', 'N/A'),
                   'State': fields.get('System.State', 'N/A'),
                   'Priority': fields.get('Microsoft.VSTS.Common.Priority', 'N/A'),
                   'Assigned to': assigned or 'Unassigned',
                   'Area': fields.get('System.AreaPath', 'N/A'),
                   'Sprint': fields.get('System.IterationPath', 'N/A'),
                   'Tags': fields.get('System.Tags', 'N/A')}], hide_index=True, width='stretch')
    st.markdown('**Description**')
    st.write(plain(fields.get('System.Description')) or 'No description provided.')
    st.markdown('**Acceptance criteria**')
    criteria = criteria_for(story)
    if criteria:
        for identifier, description in criteria.items():
            st.write(f'{identifier}: {description}')
    else:
        st.caption('No acceptance criteria provided.')
    relations = story.get('relations', [])
    if relations:
        with st.expander('Related work items and attachments'):
            for relation in relations:
                rel = relation.get('rel', 'Related item').split('.')[-1].replace('-', ' ')
                attributes = relation.get('attributes') or {}
                label = attributes.get('name') or attributes.get('comment') or rel
                st.write(f'{rel}: {label}')
                url = relation.get('url', '')
                if urlsplit(url).scheme in ('http', 'https'):
                    st.link_button('Open related item', url)


def show_comments(comments):
    if not comments:
        st.info('No discussion comments are available.')
    for comment in comments:
        creator = comment.get('createdBy') or {}
        name = creator.get('displayName', 'Unknown') if isinstance(creator, dict) else str(creator)
        st.markdown(f"**{name}** - {comment.get('createdDate', '')}")
        st.write(plain(comment.get('text', '')))


def show_evidence(items):
    st.subheader('Related knowledge')
    if not items:
        st.info('No related knowledge is indexed yet.')
    # Retrieval returns matching chunks, so group them by their actual source
    # rather than by title (different documents can have the same title).
    groups = {}
    for number, item in enumerate(items):
        metadata = item.get('metadata') or {}
        identifier = (metadata.get('source_document_id') or metadata.get('work_item_id')
                      or metadata.get('source_link'))
        if identifier is None:
            key = ('unidentified', item.get('reference_id') or number)
        else:
            key = (metadata.get('organization_id'), metadata.get('project_id'),
                   metadata.get('source_type'), str(identifier))
        groups.setdefault(key, []).append(item)

    for group in groups.values():
        metadata = group[0].get('metadata') or {}
        title = metadata.get('title') or 'Untitled source'
        with st.expander(title):
            st.write(f"**Source:** {human_label(metadata.get('source_type', 'unknown'))}")
            if metadata.get('work_item_id'):
                st.write(f"**Related work item:** {metadata['work_item_id']}")
            seen_content = set()
            for item in group:
                content = item.get('content', '')
                if content in seen_content:
                    continue
                if seen_content:
                    st.divider()
                seen_content.add(content)
                st.write(f"**Why it was retrieved:** {item.get('reason', 'Related to the selected story')}")
                if item.get('score') is not None:
                    st.caption(f"Relevance score: {item['score']:.4f}")
                st.write(content)
            link = metadata.get('source_link', '')
            if urlsplit(link).scheme in ('http', 'https'):
                st.link_button('Open source', link)


def show_sync(result, label='Knowledge'):
    st.success(f"{label}: {result['changed']} added or updated, {result['unchanged']} unchanged, {result['deleted']} removed.")


def show_results(results):
    rows = [{'Test case': item.get('title', ''), 'Azure DevOps ID': item.get('case_id', ''),
             'Status': item.get('status', '')} for item in results]
    if rows:
        st.success(f'{len(rows)} test case(s) uploaded successfully.')
        st.dataframe(rows, hide_index=True, width='stretch')


def edit_draft(workflow, scope_key):
    """Edit case fields and steps, returning a structured draft for backend validation."""
    draft = copy.deepcopy(workflow['draft'])
    criteria = criteria_for(workflow['story'])
    sources = {item['reference_id']: item.get('metadata', {}).get('title', 'Knowledge source')
               for item in workflow['context']}
    kept = []
    for number, case in enumerate(draft['test_cases'], 1):
        prefix = f"case_{scope_key}_{workflow['id']}_{workflow['revision']}_{number}"
        with st.expander(f"Test case {number}: {case['title']}", expanded=number == 1):
            title = st.text_input('Title', case['title'], key=prefix + '_title')
            description = st.text_area('Description', case['description'], key=prefix + '_description')
            preconditions = st.text_area('Preconditions (one per line)', '\n'.join(case['preconditions']), key=prefix + '_preconditions')
            col1, col2 = st.columns(2)
            with col1:
                priority = st.selectbox('Priority', [1, 2, 3, 4], index=[1, 2, 3, 4].index(case['priority']), key=prefix + '_priority')
            with col2:
                category = st.text_input('Category', case['category'], key=prefix + '_category')
            # Preserve the existing automation feasibility value in the backend draft.
            references = st.multiselect('Acceptance criteria covered', list(criteria),
                                        default=case['acceptance_criteria_reference'],
                                        format_func=lambda identifier: f'{identifier}: {criteria[identifier]}', key=prefix + '_criteria')
            related = st.multiselect('Related knowledge sources', list(sources),
                                     default=case['rag_source_references'],
                                     format_func=lambda identifier: sources[identifier], key=prefix + '_sources')
            st.markdown('**Test steps** - add or remove rows as needed')
            rows = [{'Action': step['action'], 'Test data': step['test_data'], 'Expected result': step['expected_result']}
                    for step in case['steps']]
            edited_rows = st.data_editor(rows, num_rows='dynamic', hide_index=True, width='stretch', key=prefix + '_steps',
                column_config={'Action': st.column_config.TextColumn('Action', required=True),
                               'Test data': st.column_config.TextColumn('Test data'),
                               'Expected result': st.column_config.TextColumn('Expected result', required=True)})
            if hasattr(edited_rows, 'to_dict'):
                edited_rows = edited_rows.to_dict('records')
            steps = [{'step_number': i, 'action': text_value(row.get('Action')),
                      'test_data': text_value(row.get('Test data')), 'expected_result': text_value(row.get('Expected result'))}
                     for i, row in enumerate(edited_rows, 1)]
            remove = st.checkbox('Remove this test case', key=prefix + '_remove')
            case.update(title=title.strip(), description=description.strip(),
                        preconditions=[line.strip() for line in preconditions.splitlines() if line.strip()],
                        steps=steps, priority=priority, category=category.strip(),
                        acceptance_criteria_reference=references, rag_source_references=related)
            if not remove:
                kept.append(case)
    draft['test_cases'] = kept
    for key, label in [('gaps', 'Gaps'), ('ambiguities', 'Ambiguities'), ('assumptions', 'Assumptions to verify')]:
        if draft.get(key):
            st.markdown(f'**{label}**')
            for item in draft[key]:
                st.write(f'- {item}')
    return draft


def validate_editor_fields(draft):
    if not draft['test_cases']:
        raise ValueError('Keep at least one test case.')
    for number, case in enumerate(draft['test_cases'], 1):
        if not case['title'] or not case['description'] or not case['category']:
            raise ValueError(f'Test case {number} needs a title, description and category.')
        if not case['steps'] or any(not step['action'] or not step['expected_result'] for step in case['steps']):
            raise ValueError(f'Test case {number} needs at least one step, with an action and expected result for each step.')


# def edit_test_data(value, key_prefix, label='Test data'):
#     """Edit structured test data as labeled fields while preserving its types."""
#     if isinstance(value, dict):
#         if not value:
#             st.caption(f'{label}: empty')
#         return {key: edit_test_data(item, key_prefix, f'{label} / {key}') for key, item in value.items()}
#     if isinstance(value, list):
#         if not value:
#             st.caption(f'{label}: empty')
#         return [edit_test_data(item, key_prefix, f'{label} / {index + 1}') for index, item in enumerate(value)]
#     key = key_prefix + '_' + sha256(label.encode()).hexdigest()
#     if isinstance(value, bool):
#         return st.checkbox(label, value=value, key=key)
#     if isinstance(value, int):
#         return st.number_input(label, value=value, step=1, key=key)
#     if isinstance(value, float):
#         return st.number_input(label, value=value, key=key)
#     if value is None:
#         result = st.text_input(label, value='', key=key)
#         return result if result else None
#     if '\n' in str(value):
#         return st.text_area(label, value=str(value), key=key)
#     return st.text_input(label, value=str(value), key=key)


# def editable_proposal(proposal, key_prefix):
#     """Review generated files as code or structured fields, with JSON kept internal."""
#     result = copy.deepcopy(proposal)
#     result['summary'] = st.text_area('Automation summary', result['summary'], key=key_prefix + '_summary')
#     for name, label in [('assumptions', 'Assumptions'), ('locator_evidence', 'Locator evidence')]:
#         lines = st.text_area(label + ' (one per line)', '\n'.join(result[name]), key=key_prefix + '_' + name)
#         result[name] = [line.strip() for line in lines.splitlines() if line.strip()]
#     for number, file in enumerate(result['files']):
#         with st.expander(file['path']):
#             if file['path'].lower().endswith('.json'):
#                 try:
#                     data = json.loads(file['content'])
#                 except ValueError:
#                     st.error('The generated test data is invalid. Generate the proposal again.')
#                     continue
#                 st.caption('Generated test data')
#                 edited = edit_test_data(data, f'{key_prefix}_data_{number}')
#                 file['content'] = json.dumps(edited, indent=2)
#             else:
#                 file['content'] = st.text_area('File content', file['content'], height=240, key=f'{key_prefix}_file_{number}')
#     return result

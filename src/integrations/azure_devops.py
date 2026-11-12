"""Azure DevOps REST 7.1 connector; PAT stays inside ServiceHTTP."""
import re
from urllib.parse import quote, urlsplit

from pydantic import SecretStr

from src.integrations.http import ServiceHTTP, IntegrationError


def segment(value):
    return quote(str(value), safe='')


def wiql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


class AzureDevOps:
    def __init__(self, organization_url, pat: SecretStr, api_version='7.1', transport=None):
        api_version = api_version.strip() or '7.1'
        if not re.fullmatch(r'\d+\.\d+(?:-preview(?:\.\d+)?)?', api_version):
            raise ValueError('AZURE_DEVOPS_API_VERSION must be a version such as 7.1')
        url = urlsplit(organization_url)
        if not ((url.hostname == 'dev.azure.com' and len(url.path.strip('/').split('/')) == 1 and url.path.strip('/'))
                or (url.hostname and url.hostname.endswith('.visualstudio.com') and not url.path.strip('/'))):
            raise ValueError('Use https://dev.azure.com/organization or https://organization.visualstudio.com')
        self.http = ServiceHTTP(organization_url, '', pat, transport)
        self._accounts_http = ServiceHTTP('https://app.vssps.visualstudio.com', '', pat, transport)
        self.organization_id = self.http.base.lower()
        self.version = api_version

    def close(self):
        self.http.close()
        self._accounts_http.close()

    def accounts(self):
        """Best effort: PAT/account policies may deny profile/account discovery."""
        profile, _ = self._accounts_http.request('GET', '/_apis/profile/profiles/me', params={'api-version': '7.1-preview.3'})
        body, _ = self._accounts_http.request('GET', '/_apis/accounts', params={'memberId': profile['id'], 'api-version': '7.1-preview.1'})
        return body.get('value', [])

    def request(self, method, path, **kwargs):
        params = dict(kwargs.pop('params', {}))
        params.setdefault('api-version', self.version)
        return self.http.request(method, path, params=params, **kwargs)

    def values(self, path, **params):
        values, seen = [], set()
        while True:
            body, headers = self.request('GET', path, params=params)
            values.extend(body.get('value', []))
            token = headers.get('x-ms-continuationtoken')
            if not token:
                return values
            if token in seen:
                raise IntegrationError('Service repeated a pagination token')
            seen.add(token)
            params['continuationToken'] = token

    def validate(self):
        # Location/connectionData is preview even when Core/Projects is stable.
        # Match Microsoft's LocationClient instead of applying the stable default.
        version = self.version.split('-')[0] + '-preview.1'
        try:
            body, _ = self.request('GET', '/_apis/connectionData', params={'api-version': version})
        except IntegrationError as error:
            raise IntegrationError(f'Azure DevOps identity validation failed: {error}') from None
        if not body.get('authenticatedUser', {}).get('id'):
            raise IntegrationError('No authenticated Azure DevOps identity')
        return body['authenticatedUser']

    def projects(self):
        try:
            return self.values('/_apis/projects', **{'$top': 100})
        except IntegrationError as error:
            raise IntegrationError(f'Azure DevOps project discovery failed: {error}') from None

    def authorize_project(self, project):
        body, _ = self.request('GET', f'/_apis/projects/{segment(project)}')
        return body

    def teams(self, project):
        # Teams uses skip/top pagination rather than continuation tokens.
        result, skip = [], 0
        while True:
            body, _ = self.request('GET', f'/_apis/projects/{segment(project)}/teams', params={'$top': 100, '$skip': skip})
            page = body.get('value', [])
            result.extend(page)
            if len(page) < 100:
                return result
            skip += len(page)

    def iterations(self, project, team):
        return self.values(f'/{segment(project)}/{segment(team)}/_apis/work/teamsettings/iterations')

    def query_ids(self, project, iteration=None, types=None):
        where = f'[System.TeamProject] = {wiql_literal(project)}'
        if iteration:
            where += f' AND [System.IterationPath] = {wiql_literal(iteration)}'
        if types:
            where += ' AND [System.WorkItemType] IN (' + ','.join(wiql_literal(t) for t in types) + ')'
        result, last = [], 0
        while True:
            body, _ = self.request('POST', f'/{segment(project)}/_apis/wit/wiql', params={'$top': 2000},
                                   json={'query': f'SELECT [System.Id] FROM WorkItems WHERE {where} AND [System.Id] > {last} ORDER BY [System.Id]'})
            ids = [item['id'] for item in body.get('workItems', [])]
            if not ids:
                return result
            if min(ids) <= last:
                raise IntegrationError('Work item pagination did not advance')
            result.extend(ids)
            if len(ids) < 2000:
                return result
            last = max(ids)

    def work_items(self, project, ids):
        result = []
        for offset in range(0, len(ids), 200):
            body, _ = self.request('POST', f'/{segment(project)}/_apis/wit/workitemsbatch',
                                   json={'ids': ids[offset:offset + 200], '$expand': 'Relations', 'errorPolicy': 'Fail'})
            result.extend(body.get('value', []))
        return result

    def story(self, project, work_item_id):
        items = self.work_items(project, [int(work_item_id)])
        if not items:
            raise IntegrationError('Work item was not found')
        item = items[0]
        # Project routes do not guarantee that arbitrary IDs belong to that project.
        info = self.authorize_project(project)
        if item['fields'].get('System.TeamProject', '').casefold() != info['name'].casefold():
            raise IntegrationError('Work item is outside the selected project')
        return item

    def comments(self, project, work_item_id):
        result, continuation = [], None
        while True:
            params = {'api-version': '7.1-preview.4', '$top': 200}
            if continuation:
                params['continuationToken'] = continuation
            body, _ = self.http.request('GET', f'/{segment(project)}/_apis/wit/workItems/{int(work_item_id)}/comments', params=params)
            result.extend(body.get('comments', []))
            token = body.get('continuationToken')
            if not token:
                return result
            if token == continuation:
                raise IntegrationError('Repeated comments continuation token')
            continuation = token

    def plans(self, project):
        return self.values(f'/{segment(project)}/_apis/testplan/plans')

    def suites(self, project, plan):
        return self.values(f'/{segment(project)}/_apis/testplan/Plans/{int(plan)}/suites')

    def create_plan(self, project, name):
        if not name.strip():
            raise ValueError('Plan name is required')
        return self.request('POST', f'/{segment(project)}/_apis/testplan/plans', json={'name': name})[0]

    def create_suite(self, project, plan, parent, name):
        if not name.strip():
            raise ValueError('Suite name is required')
        return self.request('POST', f'/{segment(project)}/_apis/testplan/Plans/{int(plan)}/suites',
                            json={'name': name, 'suiteType': 'staticTestSuite', 'parentSuite': {'id': int(parent)}})[0]


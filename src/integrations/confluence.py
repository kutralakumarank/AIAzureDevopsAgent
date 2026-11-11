"""Confluence Cloud v2 connector with same-origin cursor traversal."""
from urllib.parse import urlsplit

from src.integrations.http import ServiceHTTP, IntegrationError


class Confluence:
    def __init__(self, site_url, email, token, transport=None):
        # Accept site root or site/wiki, canonicalize once.
        self.http = ServiceHTTP(site_url.rstrip('/').removesuffix('/wiki'), email, token, transport)

    def close(self):
        self.http.close()

    def pages(self, space_id=None):
        path, params, seen = '/wiki/api/v2/pages', {'body-format': 'storage', 'limit': 100}, set()
        if space_id:
            params['space-id'] = str(space_id)
        while path:
            body, _ = self.http.request('GET', path, params=params)
            yield from body.get('results', [])
            next_url = body.get('_links', {}).get('next')
            if not next_url:
                break
            parsed = urlsplit(next_url)
            base = urlsplit(self.http.base)
            if parsed.netloc and (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
                raise IntegrationError('Rejected cross-origin pagination link')
            path = parsed.path
            if path.startswith('/api/v2/'):
                path = '/wiki' + path
            if not path.startswith('/wiki/api/v2/pages') or next_url in seen:
                raise IntegrationError('Invalid Confluence pagination link')
            seen.add(next_url)
            path += ('?' + parsed.query) if parsed.query else ''
            params = {}

    def spaces(self):
        path, params, result, seen = '/wiki/api/v2/spaces', {'limit': 250}, [], set()
        while path:
            body, _ = self.http.request('GET', path, params=params)
            result.extend(body.get('results', []))
            next_url = body.get('_links', {}).get('next')
            if not next_url:
                return result
            parsed, base = urlsplit(next_url), urlsplit(self.http.base)
            if parsed.netloc and (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
                raise IntegrationError('Rejected cross-origin pagination link')
            path = parsed.path
            if path.startswith('/api/v2/'):
                path = '/wiki' + path
            if not path.startswith('/wiki/api/v2/spaces') or next_url in seen:
                raise IntegrationError('Invalid Confluence pagination link')
            seen.add(next_url)
            path += '?' + parsed.query if parsed.query else ''
            params = {}
        return result

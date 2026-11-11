"""Small credential-contained transport. Never propagate remote response bodies."""
import base64
from urllib.parse import urlsplit

import httpx


class IntegrationError(RuntimeError):
    pass


def https_base(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Supply an HTTPS service URL without credentials, query or fragment')
    return value.rstrip('/')


class ServiceHTTP:
    def __init__(self, base, username, token, transport=None):
        self.base = https_base(base)
        secret = token.get_secret_value()
        if not secret.strip() or any(c in secret for c in '\r\n'):
            raise ValueError('A nonempty token is required')
        encoded = base64.b64encode(f'{username}:{secret}'.encode()).decode()
        self._client = httpx.Client(headers={'Authorization': f'Basic {encoded}'},
                                    transport=transport, timeout=30, follow_redirects=False)

    def close(self):
        self._client.close()

    def request(self, method, path, **kwargs):
        # Paths are relative; neither pagination nor caller data may redirect credentials.
        if not path.startswith('/') or path.startswith('//') or '://' in path:
            raise ValueError('Expected a service-relative API path')
        if not kwargs.get('params'):
            kwargs.pop('params', None)  # httpx params={} would discard a cursor already in path.
        try:
            response = self._client.request(method, self.base + path, **kwargs)
        except httpx.HTTPError:
            raise IntegrationError('Service connection failed; check network and service URL') from None
        if response.status_code in (401, 403):
            raise IntegrationError('Authentication or permission denied; check token scopes and project access')
        if not response.is_success:
            raise IntegrationError(f'Service request failed (HTTP {response.status_code}); retry or check permissions')
        try:
            return response.json(), response.headers
        except ValueError:
            raise IntegrationError('Service returned an invalid JSON response') from None


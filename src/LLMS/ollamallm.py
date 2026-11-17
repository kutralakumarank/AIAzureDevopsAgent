"""Discover and connect to a locally running Ollama service."""
from urllib.parse import urlsplit

import httpx
from langchain_ollama import ChatOllama


def local_url(value):
    try:
        parsed = urlsplit(value.strip())
        valid = (parsed.scheme in ('http', 'https') and
                 parsed.hostname in ('localhost', '127.0.0.1', '::1') and
                 not parsed.username and not parsed.password and
                 parsed.path in ('', '/') and not parsed.query and not parsed.fragment)
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('OLLAMA_BASE_URL must point to a local Ollama server, such as http://127.0.0.1:11434.')
    return value.strip().rstrip('/')


def list_ollama_models(base_url):
    url = local_url(base_url)
    try:
        response = httpx.get(f'{url}/api/tags', timeout=3, trust_env=False, follow_redirects=False)
        response.raise_for_status()
        data = response.json()
        models = data['models']
        if not isinstance(models, list):
            raise ValueError
        return sorted({item['name'] for item in models if isinstance(item, dict)
                       and isinstance(item.get('name'), str) and item['name'].strip()})
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise ValueError('Could not load local Ollama models. Start Ollama and select Refresh local models.') from None


def ollama_llm(model, base_url):
    if not model or not model.strip():
        raise ValueError('Select an installed Ollama model first.')
    return ChatOllama(model=model, base_url=local_url(base_url), temperature=0,
                      client_kwargs={'trust_env': False})

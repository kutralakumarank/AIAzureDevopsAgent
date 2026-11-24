"""OpenRouter model discovery and generation client."""
import os

import httpx
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


OPENROUTER_URL = 'https://openrouter.ai/api/v1'
FREE_ROUTER = 'openrouter/free'


def list_openrouter_models():
    """List text models that advertise native structured-output support."""
    try:
        response = httpx.get(f'{OPENROUTER_URL}/models', timeout=8, follow_redirects=False)
        response.raise_for_status()
        models = response.json()['data']
        if not isinstance(models, list):
            raise ValueError
        identifiers = set()
        for model in models:
            if not isinstance(model, dict):
                continue
            identifier = model.get('id')
            params = model.get('supported_parameters') or []
            architecture = model.get('architecture') or {}
            if (isinstance(identifier, str) and identifier.strip() and isinstance(params, list)
                    and isinstance(architecture, dict) and 'structured_outputs' in params
                    and 'text' in (architecture.get('input_modalities') or [])
                    and 'text' in (architecture.get('output_modalities') or [])):
                identifiers.add(identifier)
        identifiers.discard(FREE_ROUTER)
        return [FREE_ROUTER, *sorted(identifiers, key=lambda item: (not item.endswith(':free'), item.lower()))]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise ValueError('Could not load OpenRouter models. Check your connection and refresh the model list.') from None


def openrouter_llm(model, api_key=''):
    model = (model or '').strip()
    key = (api_key or '').strip() or os.getenv('OPENROUTER_API_KEY', '').strip()
    if not model:
        raise ValueError('Select or enter an OpenRouter model first.')
    if not key:
        raise ValueError('Enter an OpenRouter API key or configure OPENROUTER_API_KEY in .env.')
    return ChatOpenAI(model=model, base_url=OPENROUTER_URL, api_key=SecretStr(key),
                      extra_body={'provider': {'require_parameters': True}})

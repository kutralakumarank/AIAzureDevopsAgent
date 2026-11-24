"""Gemini generation client; credentials remain scoped to this client."""
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr


GEMINI_MODELS = ('gemini-3.5-flash-lite', 'gemini-3.6-flash', 'gemini-3.1-pro-preview')
GEMINI_THINKING_LEVELS = {'gemini-3.5-flash-lite': 'minimal', 'gemini-3.6-flash': 'minimal',
                          'gemini-3.1-pro-preview': 'low'}


def gemini_llm(model, api_key=''):
    model = (model or '').strip()
    key = (api_key or '').strip() or os.getenv('GOOGLE_API_KEY', '').strip() or os.getenv('GEMINI_API_KEY', '').strip()
    if not model:
        raise ValueError('Select or enter a Gemini model first.')
    if not key:
        raise ValueError('Enter a Gemini API key or configure GOOGLE_API_KEY in .env.')
    # One SDK attempt per request; the application applies its own bounded
    # retry policy for 503s, avoiding multiplied retries.
    options = {'thinking_level': GEMINI_THINKING_LEVELS[model]} if model in GEMINI_THINKING_LEVELS else {}
    return ChatGoogleGenerativeAI(model=model, api_key=SecretStr(key), vertexai=False, max_retries=1, **options)

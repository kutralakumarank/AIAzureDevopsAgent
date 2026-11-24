"""Safe provider diagnostics; never display provider response bodies or prompts."""
from groq import APIStatusError
from openai import APIStatusError as OpenAIStatusError, APIConnectionError as OpenAIConnectionError
from huggingface_hub.errors import HfHubHTTPError
from ollama import ResponseError
import httpx
from langchain_core.exceptions import OutputParserException
from google.genai.errors import APIError as GoogleAPIError
from langchain_google_genai._common import GoogleGenerativeAIError


class StructuredOutputError(Exception):
    """The model failed to produce a valid response after a bounded retry."""


def groq_error_code(error):
    body = getattr(error, 'body', None)
    if isinstance(body, dict) and isinstance(body.get('error'), dict):
        body = body['error']
    code = body.get('code') if isinstance(body, dict) else None
    return code if isinstance(code, str) else None


def is_groq_tool_failure(error):
    body = getattr(error, 'body', None)
    if isinstance(body, dict) and isinstance(body.get('error'), dict):
        body = body['error']
    code = groq_error_code(error)
    return code == 'tool_use_failed' or (
        code is None and isinstance(body, dict) and 'failed_generation' in body)


def provider_error_message(error):
    if isinstance(error, (StructuredOutputError, OutputParserException)):
        return ('The model returned an invalid test-case response. Try again, or select a model '
                'that supports structured output. If using Ollama, try a larger or different local model.')
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
        return 'Could not connect to the model service. If using Ollama, make sure it is running and refresh the model list.'
    if isinstance(error, OpenAIConnectionError):
        return 'Could not connect to OpenRouter. Check your network connection and retry.'
    if isinstance(error, OpenAIStatusError):
        messages = {
            400: 'OpenRouter rejected the request (HTTP 400). Check model availability and structured-output support.',
            401: 'OpenRouter rejected the API key (HTTP 401). Enter a valid OpenRouter key.',
            402: 'OpenRouter could not process the request (HTTP 402). Check your account credits and the selected model pricing.',
            403: 'OpenRouter denied access (HTTP 403). Check your account and model permissions.',
            404: 'The selected OpenRouter model is unavailable (HTTP 404). Refresh the model list or choose another model.',
            429: 'OpenRouter rate limit or quota reached (HTTP 429). Retry later or choose another model.',
            503: 'OpenRouter or its model provider is unavailable (HTTP 503). Retry later or choose another model.',
        }
        return messages.get(error.status_code, f'OpenRouter request failed (HTTP {error.status_code}). Check model access and service status.')
    if isinstance(error, GoogleGenerativeAIError):
        cause = error.__cause__
        if isinstance(cause, GoogleAPIError):
            return provider_error_message(cause)
        return 'Google could not process the model request. Check the configured model, credentials, and input.'
    if isinstance(error, GoogleAPIError):
        messages = {
            400: 'Google rejected the request (HTTP 400). Check the selected model, input size, and structured-output support.',
            401: 'Google rejected the API key (HTTP 401). Check your Gemini API key.',
            403: 'Google denied access (HTTP 403). Check your API key, account permissions, and model access.',
            404: 'The selected Google model is unavailable (HTTP 404). Choose an available model.',
            429: 'Google quota or rate limit reached (HTTP 429). Check your account limits or retry later.',
            503: 'Google is temporarily unavailable (HTTP 503), even after retries. Retry shortly; for generation, you can also select another model.',
        }
        return messages.get(error.code, 'Google could not complete the model request. Check model access and service status.')
    if isinstance(error, ResponseError):
        if error.status_code == 404:
            return 'The selected Ollama model is unavailable. Pull it locally and refresh the model list.'
        return 'Ollama could not process the request. Check that the model supports structured output and any supplied images.'
    if isinstance(error, HfHubHTTPError):
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        messages = {
            401: 'Hugging Face rejected the token. Check HUGGINGFACEHUB_API_TOKEN in .env and restart the app.',
            402: 'Hugging Face embeddings failed (HTTP 402). Check your Inference Providers credits and billing, then retry.',
            403: 'Hugging Face denied access (HTTP 403). Check token permissions and accept the EmbeddingGemma model license.',
            404: 'Hugging Face could not find the embedding model or an available inference provider (HTTP 404).',
            429: 'Hugging Face rate limit reached (HTTP 429). Wait and retry.',
            503: 'The Hugging Face embedding service is unavailable (HTTP 503). Wait and retry.',
        }
        return messages.get(status, f'Hugging Face embedding request failed (HTTP {status}). Check the model and inference provider settings.' if status else 'Hugging Face embedding request failed. Check the model and inference provider settings.')
    if isinstance(error, APIStatusError):
        status = error.status_code
        code = groq_error_code(error)
        if status == 400 and (is_groq_tool_failure(error) or code == 'json_validate_failed'):
            return 'Groq could not produce a valid structured response. Retry or select a different model.'
        if status == 400 and code in {'model_decommissioned', 'model_not_found'}:
            return 'The selected Groq model is unavailable. Select another generation model.'
        messages = {
            400: 'Groq rejected the request (HTTP 400). Check the selected model and input size, or try another model.',
            401: 'Groq rejected the API key (HTTP 401). Enter a valid Groq API key.',
            403: 'Groq denied access (HTTP 403). Check model access and API key permissions.',
            404: 'The selected Groq model is unavailable (HTTP 404). Select another model.',
            413: 'The request to Groq is too large (HTTP 413). Reduce the provided context or screenshots.',
            429: 'Groq rate limit or quota reached (HTTP 429). Wait or check your account limits.',
            503: 'The Groq service is unavailable (HTTP 503). Wait and retry.',
        }
        return messages.get(status, f'Groq request failed (HTTP {status}). Check model access and service status.')
    return None

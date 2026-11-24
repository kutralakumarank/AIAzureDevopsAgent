"""Validate generated responses and retry malformed structured output once."""
import json

from groq import BadRequestError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from src.LLMS.provider_errors import StructuredOutputError, is_groq_tool_failure
from src.LLMS.google_retry import retry_google_unavailable


def invoke_structured(llm, schema, inputs):
    def invoke(messages, *, json_mode=False):
        options = {'method': 'json_mode'} if json_mode else {}
        call = lambda: llm.with_structured_output(schema, **options).invoke(messages)
        result = retry_google_unavailable(call) if isinstance(llm, ChatGoogleGenerativeAI) else call()
        return schema.model_validate(result)

    try:
        return invoke(inputs)
    except BadRequestError as error:
        if not is_groq_tool_failure(error):
            raise
        use_json_mode = True
    except (OutputParserException, ValidationError):
        # Ollama retains its schema-constrained mode; Groq can use JSON mode.
        use_json_mode = isinstance(llm, ChatGroq)

    # Do not include the invalid response or error details in the next prompt.
    messages = inputs.to_messages() if hasattr(inputs, 'to_messages') else list(inputs)
    schema_instruction = SystemMessage(content=(
        'The previous response could not be validated. Return one complete JSON object '
        'matching this JSON schema exactly, including required fields and valid field types. '
        'Do not include Markdown or extra fields. JSON schema: ' + json.dumps(schema.model_json_schema())))
    try:
        return invoke([schema_instruction, *messages], json_mode=use_json_mode)
    except (OutputParserException, ValidationError):
        raise StructuredOutputError('The model returned invalid structured output after retry.') from None

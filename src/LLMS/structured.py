"""Validated structured calls with a narrow fallback for failed Groq tool use."""
import json

from groq import BadRequestError
from langchain_core.messages import SystemMessage

from src.LLMS.provider_errors import is_groq_tool_failure


def invoke_structured(llm, schema, inputs):
    try:
        return llm.with_structured_output(schema).invoke(inputs)
    except BadRequestError as error:
        if not is_groq_tool_failure(error):
            raise
    # JSON mode still uses LangChain's Pydantic parser. Only retry the known
    # tool-generation failure, never authentication, permissions or arbitrary 400s.
    messages = inputs.to_messages() if hasattr(inputs, 'to_messages') else list(inputs)
    schema_instruction = SystemMessage(content=(
        'Return one JSON object matching this JSON schema. Treat the schema as the '
        'output contract and do not include Markdown. Schema: ' + json.dumps(schema.model_json_schema())))
    return llm.with_structured_output(schema, method='json_mode').invoke([schema_instruction, *messages])

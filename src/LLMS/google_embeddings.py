"""Google embeddings with bounded retries for temporary HTTP 503 failures."""
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.LLMS.google_retry import retry_google_unavailable


class RetryingGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts, **kwargs):
        return retry_google_unavailable(lambda: super(RetryingGoogleEmbeddings, self).embed_documents(texts, **kwargs))

    def embed_query(self, text, **kwargs):
        return retry_google_unavailable(lambda: super(RetryingGoogleEmbeddings, self).embed_query(text, **kwargs))

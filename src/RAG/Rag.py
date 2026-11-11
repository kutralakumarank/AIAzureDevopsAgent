"""Shared document splitting and embedding configuration for knowledge ingestion."""
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Rag:
    def __init__(self, embeddings=None):
        load_dotenv()
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def split_documents(self, documents):
        return self.text_splitter.split_documents(documents)

    def create_embeddings(self):
        if self.embeddings is None:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings

            self.embeddings = HuggingFaceEndpointEmbeddings(model='google/embeddinggemma-300m')
        return self.embeddings

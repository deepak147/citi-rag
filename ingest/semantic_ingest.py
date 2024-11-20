from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import TextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from typing import List, Any
from dotenv import load_dotenv
import re
import os

load_dotenv()


class GPTSplitter(TextSplitter):

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = 600,
        overlap: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self.model = ChatOpenAI(model=model_name)
        self.prompt = ChatPromptTemplate.from_template(
            "You are an expert in identifying semantic meaning of text. "
            "You wrap each chunk in <<<>>>.\n\n"
            "Example:\n"
            'Text: "Always log out of your account after completing your session, especially when using a shared or public device.'
            "Regularly update your account passwords and choose strong combinations of letters, numbers, and symbols."
            'Never click on suspicious links claiming to be from Citi Bank without verifying their authenticity."\n'
            "Wrapped:\n"
            "<<<Always log out of your account after completing your session, especially when using a shared or public device.>>>"
            "<<<Regularly update your account passwords and choose strong combinations of letters, numbers, and symbols.>>>"
            "<<<Never click on suspicious links claiming to be from Citi Bank without verifying their authenticity.>>>\n\n"
            'Text: " Review your account statements regularly to identify any unauthorized transactions.'
            "Enable two-factor authentication for an added layer of security on your account."
            'Contact Citi Bank’s fraud prevention team if you receive unexpected calls or emails requesting sensitive information."\n'
            "Wrapped:"
            "<<<Review your account statements regularly to identify any unauthorized transactions.>>>"
            "<<<Enable two-factor authentication for an added layer of security on your account.>>>"
            "<<<Contact Citi Bank’s fraud prevention team if you receive unexpected calls or emails requesting sensitive information.>>>\n\n"
            "Now, process the following text:\n\n"
            "{text}"
        )

        self.output_parser = StrOutputParser()
        self.chain = (
            {"text": RunnablePassthrough()}
            | self.prompt
            | self.model
            | self.output_parser
        )

        self._chunk_size = chunk_size
        self._overlap = overlap

    def splitt_text(self, text: str) -> List[str]:
        response = self.chain.invoke({"text": text})
        chunks = re.findall(r"<<<(.*?)>>>", response, re.DOTALL)
        return [chunk.strip() for chunk in chunks]

    def split_text(self, text: str) -> List[str]:
        """splits text into chunks with specified chunk_size"""

        chunks = self._split_with_overlap(text)
        wrapped_chunks = []
        for chunk in chunks:
            response = self.chain.invoke({"text": chunk})
            wrapped_chunks += re.findall("<<<(.*?)>>>", response, re.DOTALL)

        return [chunk.strip() for chunk in chunks]

    def _split_with_overlap(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self._chunk_size
            chunk = " ".join(words[start:end])

            # Ensure overlap for context
            if end < len(words):
                overlap_start = max(0, end - self._overlap)
                overlap = " ".join(words[overlap_start:end])
                chunk += f" {overlap}"

            chunks.append(chunk)
            start += self._chunk_size - self._overlap

        return chunks


try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception as e:
    print(f"Error initializing embeddings: {str(e)}")


def ingest():

    loader = PyPDFLoader("Citi_Marketplace.pdf")
    documents = loader.load()
    splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"))

    chunks = splitter.split_documents(documents)

    print(f"Number of chunks: {len(chunks)}")
    for chunk in chunks[50:55]:  # Print first 5 chunks for inspection
        print(chunk, "\n\n")

    """index_name = os.environ["INDEX_NAME"]

    vector_store = PineconeVectorStore(
        embedding=embeddings, index_name=index_name
    )

    vector_store.add_documents(documents=chunks)

    print("Ingestion done")"""


if __name__ == "__main__":

    ingest()

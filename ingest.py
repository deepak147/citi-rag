import os
import logging

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()
logging.basicConfig(filename="Ingestion.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    logger.debug("Embeddings initialized successfully")
except Exception as e:
    logger.error(f"Error initializing embeddings: {str(e)}")


def ingest():

    loader = PyPDFLoader("Citi_Marketplace.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")

    index_name = os.environ["INDEX_NAME"]

    vector_store = PineconeVectorStore(
        documents=chunks, embedding=embeddings, index_name=index_name
    )

    logger.info("Ingestion done")


if __name__ == "__main__":

    ingest()

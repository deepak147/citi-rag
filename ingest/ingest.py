from unstructured.partition.pdf import partition_pdf
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv

import os

load_dotenv()

file_path = "Citi_Marketplace.pdf"

try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception as e:
    print(f"Error initializing embeddings: {str(e)}")

chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True,
    strategy="hi_res",
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True,
    chunking_strategy="by_title",
    max_characters=2500,
    combine_text_under_n_chars=500,
    new_after_n_chars=500,
)

print(len(chunks))
for chunk in chunks[:30]:
    print(chunk.text)
    print(chunk, "\n\n-----------------------------------------\n\n")

documents = [
    Document(page_content=chunk.text, metadata={"source": file_path})
    for chunk in chunks
]

index_name = os.environ["INDEX_NAME"]
vector_store = PineconeVectorStore(embedding=embeddings, index_name=index_name)
vector_store.add_documents(documents=documents)

print("Ingestion done")

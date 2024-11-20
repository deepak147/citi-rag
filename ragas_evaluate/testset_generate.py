from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from ragas.testset.synthesizers.generate import TestsetGenerator

file_path = "Citi_Marketplace.pdf"
loader = PyPDFLoader(file_path=file_path)
docs = loader.load()

for document in docs:
    document.metadata["filename"] = document.metadata["source"]

generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

testset = generator.generate_with_langchain_docs(docs, test_size=40)

testset.to_pandas()

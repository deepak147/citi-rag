import os
import re
import asyncio
from typing import List

from dotenv import load_dotenv
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.documents import Document

load_dotenv()

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

chat_model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = os.getenv("INDEX_NAME")
pinecone_vs = PineconeVectorStore.from_existing_index(
    embedding=embeddings, index_name=index_name
)
retriever = pinecone_vs.as_retriever()


def split_and_clean_text(input_text: str) -> List[str]:
    return [item for item in re.split("<<|>>", input_text) if item.strip()]


def flatten_docs(multi_query_docs: List[List[Document]]) -> List[Document]:
    seen = set()
    unique_docs = []
    for doc in (doc for sublist in multi_query_docs for doc in sublist):
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs


async def rerank_docs_async(query: str, docs: List[Document]) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "document"],
        template=(
            "Rate the relevance of the following document to the query on a scale of 0 to 10:\n\n"
            "Query: {query}\n\nDocument: {document}\n\nRelevance Score (0-10):"
        ),
    )
    rerank_chain = prompt_template | chat_model | StrOutputParser()

    async def rerank(document):
        result = await rerank_chain.ainvoke(
            {"query": query, "document": document.page_content},
            config={"callbacks": [langfuse_handler]},
        )
        try:
            return float(result), document
        except ValueError:
            return 0, document

    scores = await asyncio.gather(*(rerank(doc) for doc in docs))
    return [doc for _, doc in sorted(scores, key=lambda x: x[0], reverse=True)[:2]]


async def evaluate_docs_async(
    question: str, documents: List[Document]
) -> List[Document]:
    doc_eval_prompt = PromptTemplate(
        input_variables=["document", "question"],
        template="""You are an AI language model assistant that answers questions based on CITI bank client manual. Your task is to evaluate the provided document to determine if it is suited to answer the given user question. Assess the document for its relevance to the question, the completeness of information, and the accuracy of the content.

        Original question: {question}
        Document for Evaluation: {document}
        Evaluation Result: <<'True' if the document is suited to answer the question, 'False' if it is not>>

        Note: Conclude with a 'True' or 'False' based on your analysis of the document's relevance, completeness, and accuracy in relation to the question.""",
    )
    eval_chain = doc_eval_prompt | chat_model | StrOutputParser()

    async def evaluate(doc):
        eval_result = await eval_chain.ainvoke(
            {"document": doc.page_content, "question": question},
            config={"callbacks": [langfuse_handler]},
        )
        return eval_result.strip() == "True"

    results = await asyncio.gather(*(evaluate(doc) for doc in documents))
    return [doc for doc, is_valid in zip(documents, results) if is_valid]


async def retrieve(query: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["input"],
        template="""
            You are a helpful AI Banking support assistant. 
            {input}
        """,
    )
    initial_chain = prompt | llm | StrOutputParser()

    config = RailsConfig.from_path("config/")
    input_rails = RunnableRails(config)
    chain_with_guardrails = input_rails | initial_chain

    validate_input_result = await chain_with_guardrails.ainvoke(
        {"input": query}, config={"callbacks": [langfuse_handler]}
    )
    if (
        isinstance(validate_input_result, dict)
        and validate_input_result.get("output") == "False"
    ):
        response = "I am sorry, I am not allowed to answer about this topic."
        yield response
        return

    multi_query_prompt = PromptTemplate(
        input_variables=["query"],
        template="""You are an AI banking assistant that answers queries based on a CITI bank Client Module. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspective on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative question like this:
        <<question1>>
        <<question2>>
        Only provide the query, no numbering.
        Original question: {query}""",
    )
    multi_query_chain = (
        multi_query_prompt
        | chat_model
        | StrOutputParser()
        | RunnableLambda(split_and_clean_text)
    )
    multi_queries = await multi_query_chain.ainvoke(
        {"query": query}, config={"callbacks": [langfuse_handler]}
    )

    retrieval_tasks = [retriever.ainvoke(q) for q in multi_queries]
    docs = await asyncio.gather(*retrieval_tasks)
    flattened_docs = flatten_docs(docs)

    reranked_docs = await rerank_docs_async(query, flattened_docs)
    filtered_docs = await evaluate_docs_async(query, reranked_docs)

    context = "\n".join(doc.page_content for doc in filtered_docs)

    retrieval_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        Answer the following customer query in one or two bried paragraphs. Be clear, concise, and provide only essential information. The response should focus on general Citibank services, policies, or guidelines and not involve any sensitive customer data or specific account details.

        # Customer Query: {question}

        # Response:
        1. Provide factual, direct information relevant to the question.
        2. Avoid unnecessary elaboration or promotional language.
        3. Use a professional yet approachable tone that aligns with Citibank's standards.
        Example Query and Response:

        Query: 'What are the fees for international wire transfers?'

        Response: 'Citibank charges an international wire transfer fee that varies depending on the account type. Additional fees may apply from intermediary or recipient banks. For exact details, refer to your account’s fee schedule.'

        # Retrieved Context:
        {context}

        # Response:
        """,
    )
    final_rag_chain = retrieval_prompt | chat_model | StrOutputParser()

    async for chunk in final_rag_chain.astream(
        {"question": query, "context": context},
        config={"callbacks": [langfuse_handler]},
    ):
        yield chunk


async def main():
    query = input("Query: ")
    async for chunk in retrieve(query):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())

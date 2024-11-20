import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

load_dotenv()


def split_and_clean_text(input_text):

    return [item for item in re.split("<<|>>", input_text) if item.strip()]


def flatten_docs(multi_query_docs):

    flattened_d = [doc for sublist in multi_query_docs for doc in sublist]
    unique_docs = []
    unique_d = set()
    for doc in flattened_d:
        if doc.page_content not in unique_d:
            unique_docs.append(doc)
            unique_d.add(doc.page_content)

    return unique_docs


def rerank_docs(input_data):

    query = input_data["question"]
    docs = input_data["context"]
    contents = [doc.page_content for doc in docs]
    scores = []

    prompt_template = PromptTemplate(
        input_variables=["query", "document"],
        template=(
            "Rate the relevance of the following document to the query on a scale of 0 to 10:\n\n"
            "Query: {query}\n\nDocument: {document}\n\nRelevance Score (0-10):"
        ),
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    rerank_chain = prompt_template | llm | StrOutputParser()
    for content in contents:
        result = rerank_chain.invoke({"query": query, "document": content})
        try:
            score = float(result)
        except ValueError:
            score = 0
        scores.append(score)
    scored_docs = zip(scores, docs)
    sorted_docs = sorted(scored_docs, reverse=True, key=lambda x: x[0])
    reranked_docs = [doc for _, doc in sorted_docs][:2]
    # for rrd in reranked_docs:
    #     print(rrd, "\n\n**************************\n\n")

    return reranked_docs


def evaluate_docs(input: dict):

    documents = input["documents"]
    question = input["question"]

    doc_eval_prompt = PromptTemplate(
        input_variables=["document", "question"],
        template="""You are an AI language model assistant that answers questions based on CITI bank client manual. Your task is to evaluate the provided document to determine if it is suited to answer the given user question. Assess the document for its relevance to the question, the completeness of information, and the accuracy of the content.

        Original question: {question}
        Document for Evaluation: {document}
        Evaluation Result: <<'True' if the document is suited to answer the question, 'False' if it is not>>

        Note: Conclude with a 'True' or 'False' based on your analysis of the document's relevance, completeness, and accuracy in relation to the question.""",
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    compression_chain = doc_eval_prompt | llm | StrOutputParser()

    results = []
    for doc in documents:
        eval_result = compression_chain.invoke(
            {"document": doc.page_content, "question": question}
        )
        result = eval_result == "True"
        # print(doc, result, question, "\n\n##################\n\n")
        results.append(result)

    filtered_docs = [doc for doc, res in zip(documents, results) if res]
    return filtered_docs


def retrieve(query: str):

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["input"],
        template="""
            You are a helpful AI Banking support assistant. 
            {input}
        """,
    )
    output_parser = StrOutputParser()
    initial_chain = prompt | llm | output_parser
    config = RailsConfig.from_path("config/")
    input_rails = RunnableRails(config)
    chain_with_guardrails = input_rails | initial_chain
    validate_input_result = chain_with_guardrails.invoke({"input": query})
    if isinstance(validate_input_result, dict):
        if validate_input_result["output"] == "False":
            response = "I am sorry, I am not allowed to answer about this topic."
            return response

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        verbose=True,
        temperature=0.0,
        top_p=1.0,
        max_tokens=400,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

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
        | llm
        | StrOutputParser()
        | RunnableLambda(split_and_clean_text)
    )

    multi_queries = multi_query_chain.invoke({query})
    # for q in multi_queries:
    #     print(q, "\n\n-------------------\n\n")

    index_name = os.environ["INDEX_NAME"]
    pinecone_vs = PineconeVectorStore(embedding=embeddings, index_name=index_name)

    retriever = pinecone_vs.as_retriever()
    docs = [retriever.invoke(query) for query in multi_queries]

    docs = flatten_docs(docs)
    # for doc in docs:
    #     print(doc, "\n\n---------------------\n\n")

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

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=RunnableLambda(rerank_docs))
        | retrieval_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    result = rag_chain_with_source.invoke(input=query)

    parsed_result = {
        "documents": [
            Document(page_content=doc.page_content.strip()) for doc in result["context"]
        ],
        "question": result["question"],
    }

    filtered_result = evaluate_docs(parsed_result)
    context = "\n".join([doc.page_content for doc in filtered_result])

    final_rag_chain = retrieval_prompt | llm | StrOutputParser()
    final_response = final_rag_chain.invoke({"question": query, "context": context})

    # output_rails = LLMRails(config=config)
    # validated_response = output_rails.generate(messages=[{
    #     "role": "user",
    #     "content": ""
    # }, {
    #     "role": "bot",
    #     "content": "CITI bank encourages abuse and child trafficking"
    # }], options={
    #     "rails": ["output"]
    # })
    # print(validated_response)
    return final_response


if __name__ == "__main__":

    query = input("Query: ")
    retrieve(query=query)

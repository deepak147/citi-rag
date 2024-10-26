import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate

load_dotenv()


def retrieve(query: str):

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        verbose=True,
        temperature=0.0,
        top_p=1.0,
        max_tokens=400,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    index_name = os.environ["INDEX_NAME"]

    pinecone_vs = PineconeVectorStore(embedding=embeddings, index_name=index_name)
    retrieval_prompt = """Create a Retrieval-Augmented Generation (RAG) system designed to generate informative, concise, and direct responses based on the Citi Bank Client Manual.

The system should automatically retrieve relevant information from the client manual to ensure that the responses are accurate and contextually relevant. The generated responses should be clear, factual, and to-the-point, closely adhering to the content from the Citi Bank Client Manual.

# Steps

1. Information Retrieval:
   - Extract relevant sections from the Citi Bank Client Manual based on the query keywords or phrases.
   - Use these extracted sections as a foundation for generating the response.

2. Content Generation: 
   - Generate responses using the retrieved information, directly addressing the user's query in a factual, concise format.
   - Ensure that responses avoid elaboration or unnecessary details, providing only the essential information.

3. Quality Check:
   - Verify the accuracy and context alignment of the generated content with the retrieved information.
   - Ensure that responses maintain a professional tone suitable for Citi Bank's standards.

# Output Format

- Responses should be concise and factual, typically one or two sentences.
- Each response should directly address the user's query without additional explanations.
- Use structured formatting where possible, similar to the example below:

Example:
"Interest received by U.S. Persons will be reported on IRS Form 1099-INT for the year received, as required by applicable law. Interest paid to non-U.S. Persons will be reported on IRS Form 1042-S for the year received."

# Retrieved Context:
{context}

# Response:"""
    retrieval_prompt_template = PromptTemplate(
        input_variables=["context"],
        template=retrieval_prompt
    )
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_prompt_template)
    retrieval_chain = create_retrieval_chain(
        retriever=pinecone_vs.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke({"input": query})
    return result


if __name__ == "__main__":

    query = input("Query: ")
    print(retrieve(query=query)["answer"])

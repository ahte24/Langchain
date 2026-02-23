import os
from dotenv import load_dotenv

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

print("Initializing components...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriver = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt_template = ChatPromptTemplate.from_template(
    """Answer the equestion based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer:
    """
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL
    Manually retrieves documents, formats them, generates a response

    Limitations:
    - Manual step-by-step execution
    - No build-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relavent documents
    docs = retriver.invoke(query)

    # step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content


def create_reterival_with_lcel():
    runnable_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriver | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return runnable_chain

if __name__ == "__main__":
    print("Retrieving...")

    query = "What is Pinecone in machine learning"

    # Without RAG
    print("\n" + "=" * 70)
    print("IMPLEMENTING 0: raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # With RAG
    print("\n" + "=" * 70)
    print("IMPLEMENTING 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # With RAG
    print("\n" + "=" * 70)
    print("IMPLEMENTING 2: With LCEL")
    print("=" * 70)
    lcel_retreval = create_reterival_with_lcel()
    result = lcel_retreval.invoke({"question": query})
    print("\nAnswer:")
    print(result)

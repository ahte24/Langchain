import os
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

if __name__ == "__main__":
    print("Ingestion....")
    loader = TextLoader("mediumblog1.txt", encoding="UTF-8")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    print(f"craeted {len(texts)} chunks of the document.")

    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    print("Ingesting....")

    PineconeVectorStore.from_documents(
        texts, embedding, index_name=os.environ.get("INDEX_NAME")
    )

    print("finish")

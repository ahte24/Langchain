import certifi
import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_huggingface import HuggingFaceEmbeddings
from logger import Colors, log_error, log_header, log_info, log_success, log_warning

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # use CPU (torch not compiled with CUDA)
    encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
)

# vector_store = PineconeVectorStore(
#     index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
# )
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_document_async(documents: List[Document], batch_size: int = 50):
    """Process document in batch asynchronously."""

    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"VectoreStore Indexing: Preparing to add {len(documents)} documents to vectore store",
        Colors.DARKCYAN,
    )

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    async def add_batch(batch: List[Document], batch_number: int):
        try:
            # HuggingFace embeddings are sync, so run in a thread to avoid blocking
            await asyncio.to_thread(vector_store.add_documents, batch)
            log_success(
                f"VectoreStore Indexing: Successfully added batch {batch_number}/ {len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_number} - {e}")
            return False
        return True

    # Process batches Concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful/len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully."
        )


async def main():
    """Main sync function to orchestrate the entire process."""
    log_header("DOCUMENT INGESTION PIPELINE.")
    log_info(
        "TavilyCrawl: Starting to crawl documentaion from https://python.langchain.com/",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke(
        {
            "url": "https://docs.langchain.com/oss/python/",
            "max_depth": 5,
            "max_breadth": 50,
            "limit": 450,
            "extract_depth": "advanced",
            "select_paths": ["/oss/python/.*"],
        }
    )
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    )

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)

    log_success(
        f"Text Splitter: Created {len(texts)} chunks from {len(all_docs)} documents"
    )

    await index_document_async(texts, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("Documentation ingestion pipeline finished successfully")
    log_info("Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(texts)}")


if __name__ == "__main__":
    asyncio.run(main())

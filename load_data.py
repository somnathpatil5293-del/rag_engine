from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import cosmosdb_vector_store
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load(urls: List[str], create_container: bool = True) -> None:
    """Load documents from URLs into Azure Cosmos DB vector store."""

    print("Uploading documents to Azure Cosmos DB", urls)

    try:
        # Load documents from web
        loader = WebBaseLoader(urls)
        documents = loader.load()

        if not documents:
            raise ValueError("No documents were loaded from the provided URLs")

        # Split documents into chunks
        markdown_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = markdown_splitter.split_documents(documents)

        if not split_docs:
            raise ValueError("No document chunks were created after splitting")

        # Get vector store instance and add documents
        store = cosmosdb_vector_store.get_instance(create_container)
        store.add_documents(split_docs)

        print(
            f"Loading {len(split_docs)} document chunks from {len(documents)} documents"
        )
        print("Data loaded into Azure Cosmos DB")

    except Exception as e:
        logger.error(f"Error during data loading: {str(e)}")
        raise


if __name__ == "__main__":
    doc_urls = [
        "https://raw.githubusercontent.com/MicrosoftDocs/azure-databases-docs/refs/heads/main/articles/cosmos-db/nosql/vector-search.md",
        "https://raw.githubusercontent.com/MicrosoftDocs/azure-databases-docs/refs/heads/main/articles/cosmos-db/nosql/multi-tenancy-vector-search.md",
    ]

    load(urls=doc_urls)

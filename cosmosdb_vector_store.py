from langchain_ollama import OllamaEmbeddings
from azure.cosmos import CosmosClient, PartitionKey
from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
)
from azure.identity import DefaultAzureCredential
import os
import logging
import urllib3

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Validate required environment variables
required_env_vars = ["DATABASE_NAME", "CONTAINER_NAME", "EMBEDDINGS_MODEL"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

# Check for COSMOS_DB_URL only if not using emulator
use_emulator = os.environ.get("USE_EMULATOR", "false").lower() == "true"
if not use_emulator and not os.environ.get("COSMOS_DB_URL"):
    raise ValueError(
        "Missing required environment variable: COSMOS_DB_URL (or set USE_EMULATOR=true)"
    )

database_name = os.environ["DATABASE_NAME"]
container_name = os.environ["CONTAINER_NAME"]
partition_key = PartitionKey(path="/id")
# ollama embedding models - https://ollama.com/search?c=embedding
embeddings_model_name = os.environ["EMBEDDINGS_MODEL"]
# Get embedding dimensions from environment variable with default
embedding_dimensions = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))

cosmos_container_properties = {"partition_key": partition_key, "offer_throughput": 1000}

indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
}

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": embedding_dimensions,
        }
    ]
}

text_key = "text"
embedding_key = "embedding"
metadata_key = "metadata"


def get_instance(create_container: bool = False) -> AzureCosmosDBNoSqlVectorSearch:
    logger.info(f"Using database: {database_name}, container: {container_name}")
    logger.info(
        f"Using embedding model: {embeddings_model_name} with dimensions: {embedding_dimensions}"
    )

    try:
        if use_emulator:
            logger.info("Using Cosmos DB Emulator")
            # Only disable SSL warnings for emulator connections
            urllib3.disable_warnings()

            cosmos_client = CosmosClient(
                "https://localhost:8081/",
                "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==",
                connection_verify=False,
            )
        else:
            cosmos_db_url = os.environ["COSMOS_DB_URL"]
            cosmos_client = CosmosClient(
                cosmos_db_url, credential=DefaultAzureCredential()
            )

        embeddings = OllamaEmbeddings(model=embeddings_model_name)

        store = AzureCosmosDBNoSqlVectorSearch(
            database_name=database_name,
            container_name=container_name,
            embedding=embeddings,
            cosmos_client=cosmos_client,
            vector_search_fields={
                "text_field": text_key,
                "embedding_field": embedding_key,
            },
            metadata_key=metadata_key,
            create_container=create_container,
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties={},
            full_text_search_enabled=False,
        )

        logger.info("Successfully created instance of AzureCosmosDBNoSqlVectorSearch")
        return store

    except Exception as e:
        logger.error(
            f"Failed to create AzureCosmosDBNoSqlVectorSearch instance: {str(e)}"
        )
        raise

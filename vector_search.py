import cosmosdb_vector_store
import sys
import logging
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def search_vectors(query: str, top_k: int = 5) -> List[Tuple]:
    """Perform vector similarity search."""
    try:
        print(f'Searching top {top_k} results for query: "{query}"\n')

        store = cosmosdb_vector_store.get_instance()
        results = store.similarity_search_with_score(query=query, k=top_k)

        if not results:
            print("No results found for the query.")
            return []

        for result in results:
            print(f"Score: {result[1]}")
            print(f"Content: {result[0].page_content}")
            print("=" * 70)

        return results

    except Exception as e:
        logger.error(f"Error during vector search: {str(e)}")
        raise


def main():
    """Main function to handle command line arguments and execute search."""
    if len(sys.argv) < 2:
        print("Usage: python vector_search.py <query> [top_k]")
        print("Example: python vector_search.py 'How does a vector store work?' 10")
        sys.exit(1)

    try:
        query = sys.argv[1]

        # Optional second argument for top_k
        top_k = 5  # default
        if len(sys.argv) > 2:
            try:
                top_k = int(sys.argv[2])
                if top_k <= 0:
                    raise ValueError("top_k must be a positive integer")
            except ValueError as e:
                print(f"Invalid top_k value: {sys.argv[2]}. Using default value of 5.")
                top_k = 5

        search_vectors(query, top_k)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

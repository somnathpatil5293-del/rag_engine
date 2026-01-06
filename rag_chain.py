from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import cosmosdb_vector_store
import os
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Validate required environment variables
required_env_vars = ["CHAT_MODEL"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

system_prompt = (
    "You are a friendly assistant for question-answering tasks. Use the following retrieved context to answer the question. "
    "Do not start the answer with 'According to the provided context'. "
    "Consider the previous conversation when relevant, but ensure your answer is primarily based on the retrieved context. "
    "If the answer is not present in the provided context, just say so. Ensure that the answer is strictly based on the context given, "
    "without inferring or making assumptions. Be helpful but concise. Do not be rude. While answering, you don't need to repeat that "
    "you are answering based on the context.\n\n"
    "Previous conversation:\n{chat_history}\n\n"
    "Retrieved context:\n{context}"
)

chat_model = os.environ["CHAT_MODEL"]
# Get top_k from environment variable with default
top_k = int(os.environ.get("TOP_K", "5"))

# Simple chat history storage
chat_history: List[Dict[str, str]] = []


def format_chat_history(history: List[Dict[str, str]], max_turns: int = 5) -> str:
    if not history:
        return "No previous conversation."

    # Keep only the last max_turns conversations
    recent_history = history[-max_turns:]

    formatted = []
    for turn in recent_history:
        formatted.append(f"Human: {turn['human']}")
        formatted.append(f"Assistant: {turn['assistant']}")

    return "\n".join(formatted)


def add_to_history(human_message: str, assistant_message: str) -> None:
    """Add conversation to history."""
    chat_history.append({"human": human_message, "assistant": assistant_message})


def clear_history() -> None:
    """Clear chat history."""
    global chat_history
    chat_history = []


def build():
    """Build and return the RAG chain."""

    print(f"Building RAG chain. Using model: {chat_model}")
    print(f"Using retriever with k={top_k}")

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        retriever = cosmosdb_vector_store.get_instance(False).as_retriever(k=top_k)
        llm = ChatOllama(model=chat_model)

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    except Exception as e:
        logger.error(f"Failed to build RAG chain: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        rag_chain = build()

        print(
            "Enter your questions below. Type 'exit' to quit, 'clear' to clear chat history, 'history' to view chat history."
        )

        while True:
            try:
                query = input("[User]: ").strip()
                if query.lower() == "exit":
                    break
                elif query.lower() == "clear":
                    clear_history()
                    print("Chat history cleared.")
                    continue
                elif query.lower() == "history":
                    if chat_history:
                        print("\n--- Chat History ---")
                        for i, turn in enumerate(chat_history, 1):
                            print(f"{i}. [User]: {turn['human']}")
                            print(f"{i}. [Assistant]: {turn['assistant']}")
                        print("--- End History ---\n")
                    else:
                        print("No chat history available.\n")
                    continue
                elif not query:
                    continue

                response = ""
                sources = []

                print("[Assistant]: ", end="")

                # Include chat history in the input
                formatted_history = format_chat_history(chat_history)

                for chunk in rag_chain.stream(
                    input={"input": query, "chat_history": formatted_history}
                ):
                    if "answer" in chunk:
                        print(chunk["answer"], end="")
                        response = response + chunk["answer"]
                    if "context" in chunk:
                        for source_doc in chunk["context"]:
                            sources.append(source_doc.metadata.get("source"))

                print("\n")

                # Add this conversation to history
                add_to_history(query, response)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error during conversation: {str(e)}")
                print(f"Sorry, an error occurred: {str(e)}")

    except Exception as e:
        logger.error(f"Failed to start RAG application: {str(e)}")
        print(f"Failed to start application: {str(e)}")
        exit(1)

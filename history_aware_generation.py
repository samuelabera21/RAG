import os
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from answergeneration import generate_answer


load_dotenv()


def create_retriever(
    persist_directory: str = "db/chroma_db",
    k: int = 5,
):
    """Load Chroma using the same Jina model used during ingestion."""

    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        raise RuntimeError("JINA_API_KEY is missing from the .env file.")

    embedding_model = JinaEmbeddings(
        jina_api_key=jina_api_key,
        model_name="jina-embeddings-v5-text-small",
    )

    database = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return database.as_retriever(search_kwargs={"k": k})


def rewrite_question(
    user_question: str,
    chat_history: Sequence[HumanMessage | AIMessage],
) -> str:
    """Rewrite a follow-up question as a standalone search query."""

    if not chat_history:
        return user_question.strip()

    messages = [
        SystemMessage(
            content=(
                "Rewrite the latest user question as one standalone, searchable "
                "question using the conversation history. Return only the rewritten "
                "question, with no explanation."
            )
        ),
        *chat_history,
        HumanMessage(content=f"New question: {user_question}"),
    ]

    model = ChatOpenRouter(
        model="openrouter/free",
        temperature=0,
    )
    result = model.invoke(messages)

    if not isinstance(result.content, str):
        raise TypeError("The chat model returned non-text content.")

    search_question = result.content.strip()
    if not search_question:
        raise RuntimeError("The model returned an empty search question.")

    return search_question


def ask_question(
    user_question: str,
    retriever,
    chat_history: list[HumanMessage | AIMessage] | None = None,
) -> tuple[str, list[HumanMessage | AIMessage]]:
    """Retrieve context, answer a question, and return updated chat history."""

    if not user_question.strip():
        raise ValueError("The question cannot be empty.")

    history = list(chat_history or [])
    search_question = rewrite_question(user_question, history)
    relevant_docs: list[Document] = list(retriever.invoke(search_question))

    answer = generate_answer(
        query=user_question,
        relevant_docs=relevant_docs,
        chat_history=history,
    )

    history.extend(
        [
            HumanMessage(content=user_question),
            AIMessage(content=answer),
        ]
    )
    return answer, history


def start_chat() -> None:
    """Run a terminal chat using the history-aware RAG pipeline."""

    retriever = create_retriever()
    chat_history: list[HumanMessage | AIMessage] = []

    print("Ask questions about your documents. Type 'quit' to exit.")

    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in {"quit", "exit"}:
            break

        answer, chat_history = ask_question(
            user_question=user_question,
            retriever=retriever,
            chat_history=chat_history,
        )
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    start_chat()

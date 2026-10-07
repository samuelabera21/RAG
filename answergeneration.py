
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

load_dotenv()


def generate_answer(
    query: str,
    relevant_docs: Sequence[Document],
    chat_history=None,
) -> str:
    """Generate an answer using only the retrieved documents."""

    if not relevant_docs:
        return "I don't have enough information to answer this question."

    context = "\n\n".join(
        f"Document {index}:\n{document.page_content}"
        for index, document in enumerate(relevant_docs, start=1)
    )

    prompt = f"""Answer the question using only the context below.

Question:
{query}

Context:
{context}

If the answer is not present in the context, say:
"I don't have enough information to answer this question."
"""

    messages = [
        SystemMessage(
            content=(
                "You are a helpful RAG assistant. "
                "Use only the supplied context. Do not invent facts."
            )
        ),
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append(HumanMessage(content=prompt))

    model = ChatOpenRouter(
        model="openrouter/free",
        temperature=0,
    )

    result = model.invoke(messages)

    if not isinstance(result.content, str):
        raise TypeError("The chat model returned non-text content.")

    return result.content.strip()

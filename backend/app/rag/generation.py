"""Grounded answer generation."""
from langchain_core.prompts import ChatPromptTemplate

from app.services.generator import get_chat_model


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an enterprise business-intelligence assistant.

Answer ONLY from the supplied context. If the context does not contain
enough information, say exactly: "The provided documents do not contain
enough information to answer this question."

Do not invent facts, numbers, sources, or citations.

Context:
{context}
""",
        ),
        ("human", "{question}"),
    ]
)


def _format_context(documents) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("filename", "unknown")
        page = metadata.get("page", metadata.get("row_index", ""))
        blocks.append(
            f"[Source {i}: {source}"
            + (f", page/row {page}" if page != "" else "")
            + f"]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, documents) -> str:
    context = _format_context(documents)
    messages = PROMPT.format_messages(context=context, question=question)
    response = get_chat_model().invoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)

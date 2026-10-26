from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an assistant that answers questions about the policies of the travel agency Rumbo Sur Viajes. "
                "Answer just with the context information, don't use your own knowledge. "
                "If the answer is not present in the context, answer exactly 'No lo sé' even if you know the answer from your own knowledge "
                "and the 'sources' list empty. In 'sources' put just the files that you actually used to find the answer. "
                "\n\n{format_instructions}"
            ),
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

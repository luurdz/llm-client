from langchain_core.prompts import ChatPromptTemplate

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a senior software engineer who analyzes software architecture "
                "descriptions and error logs. Extract the technical information from the "
                "text provided by the user. Only use information that appears in the text "
                "and never invent technologies. If the text is ambiguous or has little "
                "technical detail, extract only what is explicitly mentioned. "
                "If the text does not mention any specific technology, return an empty technologies list."
            ),
        ),
        ("human", "Text to analyze:\n{text}"),
    ]
)

from langchain_core.prompts import ChatPromptTemplate

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ("You are a senior software engineer who analyzes software architecture "
            "descriptions and error logs. Extract the technical information from the "
            "text provided by the user. Only use information that appears in the text "
            "and never invent technologies. If the text is ambiguous or has little "
            "technical detail, extract only what is explicitly mentioned and choose "
            "the lowest criticality level that the text supports."),
        ),
        ("human", "Text to analyze:\n{text}"),
    ]
)

if __name__ == "__main__":
    print(EXTRACTION_PROMPT.input_variables)
    prompt_value = EXTRACTION_PROMPT.invoke({"text": "FastAPI service with Redis cache is timing out."})
    print(prompt_value.to_string())
from src.llm_manager import LanguageModelFactory
from src.configs import ModelProvider, GroqModelName
from langchain_core.messages import HumanMessage


def generate_chat_title(user_message: str) -> str:
    """
    Generate a title for the conversation based on the user's message.
    """
    model = LanguageModelFactory.create_model(
        provider=ModelProvider.GROQ,
        model_name=GroqModelName.LLAMA_3_2_1B
    )

    user_message = HumanMessage(content='Generate one title based on the following message from a user in 3 to 5 words max: \n' + user_message)
    
    ai_response = model.generate_response([user_message])

    if ai_response.content:
        #strip the title of any leading or trailing whitespace, and remove any double quotes
        return ai_response.content.strip().replace('"', '')
    else:
        return "New Chat"

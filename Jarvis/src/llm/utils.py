from src.llm.llm_manager import LanguageModelFactory
from src.configs import ModelProvider, GroqModelName
from langchain_core.messages import HumanMessage


async def generate_chat_title(user_message: str) -> str:
    """
    Generate a title for the conversation based on the user's message.
    
    This function uses a language model to process the user's message and generate a title.
    The title is then processed and returned as a string.

    Args:
        user_message (str): The message from the user that will be used to generate the chat title.

    Returns:
        str: A title for the conversation in 3 to 5 words.

    Raises:
        Exception: If the language model fails to generate a response.

    Examples:
        >>> await generate_chat_title("Hello, how are you?")
        'Hello World'
    """

    
    # Create a language model instance
    model = await LanguageModelFactory.create_model(
        provider=ModelProvider.GROQ,
        model_name=GroqModelName.LLAMA_3_2_1B
    )

    # Prepare the user message for the model
    user_input = HumanMessage(content=f"Generate one title based on the following message from a user in 3 to 5 words max: \n{user_message}")

    # Generate a response from the model
    ai_response = await model.generate_response([user_input])

    # Extract and process the response content
    title = ai_response.content.strip().replace('"', '') if ai_response.content else "New Chat"

    # Return the processed title
    return title

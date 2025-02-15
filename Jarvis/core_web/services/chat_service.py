from core_web.models import Conversation
from src.chat import BotBuilder
from src.configs import WorkflowType, ModelProvider, BaseModelName
from core_web.django_storage import ChatStorageType
import logging
import traceback


logger = logging.getLogger(__name__)

class CreateChatbotError(Exception):
    """Exception raised when a chatbot instance cannot be created."""
    pass

# Global cache to store chatbot instances
CHATBOT_CACHE = {}

async def get_chatbot_instance(model_provider: ModelProvider, model_name: BaseModelName, temperature: float):
    """
    Reuse or create a chatbot instance based on configuration.
    """

    global CHATBOT_CACHE

    # Generate a key for caching
    config_key = f"{model_provider.value}:{model_name.value}:{temperature}"

    # Reuse instance if it exists
    if config_key in CHATBOT_CACHE:
        return CHATBOT_CACHE[config_key]

    # Otherwise, create a new chatbot instance
    builder = BotBuilder()
    
    builder = await builder.with_model(provider=model_provider, model_name=model_name)
    builder = await builder.with_storage(storage_type=ChatStorageType.DJANGO)
    builder = await builder.with_workflow(workflow_type=WorkflowType.CHATBOT)
    builder = await builder.with_temperature(temperature)
    
    chatbot = await builder.build()

    if chatbot:
        # Store in cache
        CHATBOT_CACHE[config_key] = chatbot
        return chatbot
    else:
        logger.error(f"Failed to create chatbot instance: {traceback.format_exc()}")
        raise CreateChatbotError("Failed to create chatbot instance")



from django.db import transaction
from asgiref.sync import sync_to_async

class ChatService:
    @staticmethod
    async def create_or_get_empty_chat(user):
        """
        Creates a new chat or returns an existing empty conversation.
        
        Args:
            user: The user object
            
        Returns:
            tuple: (conversation, message)
            - conversation: Conversation object
            - message: Status message string
        """
        try:
            # Get the latest empty conversation (sync ORM wrapped with async)
            latest_conversation = await sync_to_async(
                lambda: Conversation.objects.filter(user=user).order_by('-created_at').first()
            )()

            # Check if the latest conversation is empty
            if latest_conversation and not await sync_to_async(latest_conversation.message_pairs.exists)():
                return latest_conversation, 'Using existing empty conversation'

            # Create a new conversation (inside a transaction for safety)
            @sync_to_async
            def create_conversation():
                with transaction.atomic():
                    return Conversation.objects.create(user=user, title="New Chat")

            conversation = await create_conversation()

            return conversation, 'New conversation created'

        except Exception as e:
            logger.error(f"Error in create_or_get_empty_chat: {str(e)}")
            raise






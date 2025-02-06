from core_web.models import Conversation
import logging

logger = logging.getLogger(__name__)

class ChatService:
    @staticmethod
    def create_or_get_empty_chat(user):
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
            # Check if user is already in an empty conversation
            user_conversations = Conversation.objects.filter(user=user)
            
            # Get the most recent conversation
            latest_conversation = user_conversations.order_by('-created_at').first()
            
            if latest_conversation and not latest_conversation.message_pairs.exists():
                return latest_conversation, 'Using existing empty conversation'
            
            # If no empty conversation exists, create a new one
            conversation = Conversation.objects.create(
                user=user,
                title="New Chat"
            )
            
            return conversation, 'New conversation created'

        except Exception as e:
            logger.error(f"Error in create_or_get_empty_chat: {str(e)}")
            raise 
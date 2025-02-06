from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
from core_web.models import Conversation, MessagePair
from typing import List, Dict
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.configs import ChatStorageType


@dataclass
class MessageData:
    """Data class to represent a message pair"""
    user_message: str
    ai_message: str
    summary: Optional[str] = None
    tokens_used: Dict = None
    model_version: str = ""
    status: str = "completed"
    processing_time: Optional[float] = None
    error_message: str = ""


class ChatStorageInterface(ABC):
    """Abstract interface for chat storage operations"""
    
    @abstractmethod
    def create_conversation(self, user_id: int, title: str = "") -> str:
        """Create a new conversation and return its ID"""
        pass
    
    @abstractmethod
    def save_message(self, conversation_id: str, message_data: MessageData) -> bool:
        """Save a message pair to the conversation"""
        pass
    
    @abstractmethod
    def load_conversation(self, conversation_id: str, limit: Optional[int] = None) -> List[MessageData]:
        """
        Load messages from a conversation
        Args:
            conversation_id: The ID of the conversation
            limit: Optional number of latest messages to return. If None, returns all messages.
        Returns:
            List of MessageData ordered by newest first
        """
        pass
    
    @abstractmethod
    def get_user_conversations(self, user_id: int) -> List[Dict]:
        """Get all conversations for a user"""
        pass

class DjangoStorage(ChatStorageInterface):
    """Django implementation of chat storage"""
    
    def create_conversation(self, user_id: int, title: str = "") -> str:
        """Create a new conversation and return its ID"""
        User = get_user_model()
        user = User.objects.get(id=user_id)
        conversation = Conversation.objects.create(
            user=user,
            title=title
        )
        return str(conversation.conversation_id)
    
    def save_message(self, conversation_id: str, message_data: MessageData) -> bool:
        """Save a message pair to the conversation"""
        try:
            conversation = Conversation.objects.get(conversation_id=conversation_id)
            
            # Create message pair
            message_pair = MessagePair.objects.create(
                conversation=conversation,
                user_message=message_data.user_message,
                ai_message=message_data.ai_message,
                summary=message_data.summary,
                tokens_used=message_data.tokens_used or {},
                model_version=message_data.model_version,
                status=message_data.status,
                processing_time=message_data.processing_time,
                error_message=message_data.error_message
            )
            return True
        except ObjectDoesNotExist:
            return False
    
    def load_conversation(self, conversation_id: str, limit: Optional[int] = None) -> List[MessageData]:
        """
        Load messages from a conversation
        Args:
            conversation_id: The ID of the conversation
            limit: Optional number of latest messages to return. If None, returns all messages.
        Returns:
            List of MessageData ordered by newest first
        """
        try:
            conversation = Conversation.objects.get(conversation_id=conversation_id)
            messages = []
            
            # Get message pairs ordered by creation time, newest first
            message_pairs = conversation.message_pairs.all().order_by('-created_at')
            
            # Apply limit if specified
            if limit is not None:
                message_pairs = message_pairs[:limit]
            
            for pair in message_pairs:
                messages.append(MessageData(
                    user_message=pair.user_message,
                    ai_message=pair.ai_message,
                    summary=pair.summary,
                    tokens_used=pair.tokens_used,
                    model_version=pair.model_version,
                    status=pair.status,
                    processing_time=pair.processing_time,
                    error_message=pair.error_message
                ))
            
            # Reverse the list to maintain chronological order (oldest to newest)
            return list(reversed(messages))
            
        except ObjectDoesNotExist:
            return []
    
    def get_user_conversations(self, user_id: int) -> List[Dict]:
        """Get all conversations for a user"""
        User = get_user_model()
        try:
            user = User.objects.get(id=user_id)
            return [
                {
                    'id': str(conv.id),
                    'title': conv.title or 'Untitled',
                    'created_at': conv.created_at,
                    'updated_at': conv.updated_at,
                    'message_count': conv.message_pairs.count()
                }
                for conv in user.conversations.all()
            ]
        except ObjectDoesNotExist:
            return []
        
class StorageManager:
    """Interface to manage chat storage operations"""
    
    def __init__(self, storage_type: ChatStorageType = ChatStorageType.DJANGO):
        if storage_type == ChatStorageType.DJANGO:
            self.storage = DjangoStorage()
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def create_conversation(self, user_id: int, title: str = "") -> str:
        """Create a new conversation"""
        return self.storage.create_conversation(user_id, title)
    
    def save_message(self, conversation_id: str, message_data: MessageData) -> bool:
        """Save a message pair to the conversation"""
        return self.storage.save_message(conversation_id, message_data)
    
    def load_conversation(self, conversation_id: str, limit: Optional[int] = None) -> List[MessageData]:
        """
        Load messages from a conversation
        Args:
            conversation_id: The ID of the conversation
            limit: Optional number of latest messages to return. If None, returns all messages.
        Returns:
            List of MessageData ordered by newest first
        """
        return self.storage.load_conversation(conversation_id, limit)
    
    def get_user_conversations(self, user_id: int) -> List[Dict]:
        """Get all conversations for a user"""
        return self.storage.get_user_conversations(user_id)
    
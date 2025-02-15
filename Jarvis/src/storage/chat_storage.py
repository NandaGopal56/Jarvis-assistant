from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
from asgiref.sync import sync_to_async
from typing import List, Dict
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.configs import ChatStorageType
from core_web.models import Conversation, MessagePair


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
    async def create_conversation(self, user_id: int, title: str = "") -> str:
        """Create a new conversation and return its ID"""
        pass
    
    @abstractmethod
    async def save_message(self, conversation_id: str, message_data: MessageData) -> bool:
        """Save a message pair to the conversation"""
        pass
    
    @abstractmethod
    async def load_conversation(self, conversation_id: str, limit: Optional[int] = None) -> List[MessageData]:
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
    async def get_user_conversations(self, user_id: int) -> List[Dict]:
        """Get all conversations for a user"""
        pass

class DjangoStorage(ChatStorageInterface):
    """Django implementation of chat storage"""
    
    async def create_conversation(self, user_id: int, title: str = "") -> str:
        """Create a new conversation and return its ID"""
        User = get_user_model()
        user = await User.objects.aget(id=user_id)
        conversation = await sync_to_async(Conversation.objects.create)(
            user=user,
            title=title
        )
        return str(conversation.conversation_id)

    async def save_message(self, conversation_id: str, message_data: MessageData) -> bool:
        """Save a message pair to the conversation"""
        try:
            conversation = await Conversation.objects.aget(conversation_id=conversation_id)
            
            # Create message pair
            await sync_to_async(MessagePair.objects.create)(
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
    
    async def load_conversation(self, conversation_id: str, limit: Optional[int] = None) -> List[MessageData]:
        """
        Load messages from a conversation
        Args:
            conversation_id: The ID of the conversation
            limit: Optional number of latest messages to return. If None, returns all messages.
        Returns:
            List of MessageData ordered by newest first
        """
        try:
            conversation = await Conversation.objects.aget(conversation_id=conversation_id)
            messages = []

            # Get message pairs ordered by creation time, newest first
            message_pairs = await sync_to_async(list)(conversation.message_pairs.all().order_by('-created_at'))

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
        
    async def get_user_conversations(self, user_id: int) -> List[Dict]:
        """Get all conversations for a user"""
        User = get_user_model()
        try:
            user = await User.objects.aget(id=user_id)
            return [
                {
                    'id': str(conv.id),
                    'title': conv.title or 'Untitled',
                    'created_at': conv.created_at,
                    'updated_at': conv.updated_at,
                    'message_count': await conv.message_pairs.count()
                }
                for conv in await user.conversations.all()
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
    
    async def create_conversation(self, user_id: int, title: str = "") -> str:
        """Create a new conversation"""
        return await self.storage.create_conversation(user_id, title)
    
    async def save_message(self, conversation_id: str, message_data: MessageData) -> bool:
        """Save a message pair to the conversation"""
        return await self.storage.save_message(conversation_id, message_data)
    
    async def load_conversation(self, conversation_id: str, limit: Optional[int] = None) -> List[MessageData]:
        """
        Load messages from a conversation
        Args:
            conversation_id: The ID of the conversation
            limit: Optional number of latest messages to return. If None, returns all messages.
        Returns:
            List of MessageData ordered by newest first
        """
        return await self.storage.load_conversation(conversation_id, limit)
    
    async def get_user_conversations(self, user_id: int) -> List[Dict]:
        """Get all conversations for a user"""
        return await self.storage.get_user_conversations(user_id)
    
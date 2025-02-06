import json
import logging
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from src.llm_manager import GroqModelName
from core_web.django_storage import ChatStorageType
from rest_framework.decorators import api_view
from src.chat import BotBuilder
from src.configs import WorkflowType, ModelProvider
from core_web.models import Conversation, AIChatMessageStatus
from core_web.services.chat_service import ChatService
from src.utils import generate_chat_title

logger = logging.getLogger(__name__)

# Home Page
def home_view(request):
    return render(request, 'home.html')


@login_required
def chat_home(request):

    # Get or create an empty conversation
    conversation, message = ChatService.create_or_get_empty_chat(request.user)

    # Redirect to the chat view with the empty conversation ID
    return redirect('chat_with_id', conversation.conversation_id)

# Chat with LLM
@login_required
def chat_view(request, conversation_id):
    """
    Single view to handle both new chats and existing conversations
    """
    # Get all conversations for the sidebar
    conversations = Conversation.objects.filter(user=request.user).order_by('-updated_at')
    
    current_conversation_set = conversations.filter(conversation_id=conversation_id)
    if current_conversation_set:
        current_conversation = current_conversation_set[0]
    else:
        current_conversation = None
        
    # If invalid conversation_id, redirect to main chat page
    if not current_conversation:
        return redirect('chat_home')
    
    return render(request, 'chat.html', {
        'conversations': conversations,
        'current_conversation': current_conversation
    })

@login_required
@csrf_protect
@require_http_methods(["POST"])
def create_new_chat(request):
    """
    API endpoint that creates a new conversation or returns existing empty conversation.
    Returns JSON with conversation ID and redirect URL.
    """
    try:
        conversation, message = ChatService.create_or_get_empty_chat(request.user)
        
        return JsonResponse({
            'status': 'success',
            'message': message,
            'thread_id': conversation.conversation_id,
        })

    except Exception as e:
        logger.error(f"Error creating new chat: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


# Initialize chatbot with config
chatbot = BotBuilder() \
            .with_model(provider=ModelProvider.GROQ, model_name=GroqModelName.LLAMA_3_3_70B) \
            .with_storage(storage_type = ChatStorageType.DJANGO) \
            .with_workflow(workflow_type = WorkflowType.CHATBOT) \
            .with_temperature(0.0) \
            .build()

    
@csrf_protect
@require_http_methods(["POST"])
def chat_api(request):
    """
    API endpoint to handle chat interactions
    
    Expected POST data:
    {
        "message": "user message here",
        "thread_id": "unique_conversation_id",
        "model_config": {  # optional
            "model_provider": "groq",
            "model_name": "llama-3-2-1b-preview",
            "temperature": 0
        }
    }
    """
    try:
        # Read the request body once and store it
        if not hasattr(request, '_cached_body'):
            request._cached_body = request.body.decode('utf-8')
        
        data = json.loads(request._cached_body)
        
        # Check for required fields with specific error messages
        if 'message' not in data:
            return JsonResponse({
                'error': 'Message field is missing in the request body'
            }, status=400)
            
        if 'thread_id' not in data:
            return JsonResponse({
                'error': 'thread_id field is missing in the request body'
            }, status=400)
            
        
        user_message = data['message']
        thread_id = data['thread_id']

        print(f"Received message: {user_message} with thread_id: {thread_id}")

        global chatbot

        # Get response from chatbot
        response = chatbot.chat(user_message, thread_id)

        conversation = Conversation.objects.get(conversation_id=thread_id)

        if conversation.title == "New Chat":
            conversation.title = generate_chat_title(user_message)
            conversation.save()

        return JsonResponse({
            'status': 'success',
            'thread_id': thread_id,
            'conversation_title': conversation.title,
            'response': response
        })

    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        return JsonResponse({
            'error': 'Invalid JSON in request body',
            'thread_id': thread_id,
        }, status=400)
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return JsonResponse({
            'error': f'Error processing request: {str(e)}',
            'thread_id': thread_id,
        }, status=500)


@login_required
@require_http_methods(["GET"])
def get_conversation_history(request, conversation_id):
    """Get the history of a specific conversation"""
    try:
        global chatbot  # Using default config
        messages = chatbot.storage.load_conversation(conversation_id)
        
        return JsonResponse({
            'status': 'success',
            'thread_id': conversation_id,
            'messages': [
                {
                    'user_message': msg.user_message,
                    'ai_message': msg.ai_message,
                    'timestamp': msg.created_at if hasattr(msg, 'created_at') else None
                } for msg in messages
            ]
        })
    except Exception as e:
        logger.error(f"Error fetching conversation history: {str(e)}")
        return JsonResponse({
            'error': f'Error fetching conversation history: {str(e)}'
        }, status=500)
    



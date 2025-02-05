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
from core_web.models import Conversation


logger = logging.getLogger(__name__)

# Home Page
def home_view(request):
    return render(request, 'home.html')

# Chat with LLM
@login_required
def chat_view(request, conversation_id=None):
    """
    Single view to handle both new chats and existing conversations
    """
    # Get all conversations for the sidebar
    conversations = Conversation.objects.filter(user=request.user).order_by('-updated_at')
    
    # If conversation_id is provided, get that specific conversation
    current_conversation = None
    if conversation_id:
        current_conversation = Conversation.objects.filter(
            conversation_id=conversation_id,
            user=request.user
        ).first()
        
        # If invalid conversation_id, redirect to main chat page
        if not current_conversation:
            return redirect('chat')
    
    return render(request, 'chat.html', {
        'conversations': conversations,
        'current_conversation': current_conversation
    })

@login_required
def create_new_chat(request):
    """
    Creates a new conversation and redirects to its chat view
    """
    conversation = Conversation.objects.create(
        user=request.user,
        title="New Chat"
    )
    
    return redirect('chat_with_id', conversation_id=conversation.conversation_id)

# Search with LLM
@login_required
def search_with_llm_view(request):
    return render(request, 'search_with_llm.html')

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

        return JsonResponse({
            'status': 'success',
            'thread_id': thread_id,
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

# Optional: Endpoint to get conversation history
@login_required
@require_http_methods(["GET"])
def get_conversation_history(request, thread_id):
    """Get the history of a specific conversation"""
    try:
        chatbot = ChatBot()  # Using default config
        messages = chatbot.storage.load_conversation(thread_id)
        
        return JsonResponse({
            'status': 'success',
            'thread_id': thread_id,
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
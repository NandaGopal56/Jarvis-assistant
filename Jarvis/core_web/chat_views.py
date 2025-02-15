import json
import logging
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from asgiref.sync import sync_to_async
from django.conf import settings
from core_web.models import Conversation
from core_web.services.chat_service import ChatService
from src.llm.llm_manager import GroqModelName
from src.configs import ModelProvider
from src.llm.utils import generate_chat_title
from core_web.services.chat_service import get_chatbot_instance
import traceback

logger = logging.getLogger(__name__)

# Home Page
def home_view(request):
    return render(request, 'home.html')

@login_required
async def chat_home(request, conversation_id=None):


    if conversation_id is None:
        # Get or create an empty conversation
        conversation, message = await ChatService.create_or_get_empty_chat(request.user)
        conversation_id = conversation.conversation_id  
        return redirect('chat_with_id', conversation_id)

    return render(request, 'chat.html', {'conversation_id': conversation_id})
    


@login_required
async def get_conversations(request, conversation_id):
    """
    API endpoint to get all conversations and the current conversation.
    """
    # Accessing request.user in an async context requires sync_to_async
    user = await sync_to_async(lambda: request.user, thread_sensitive=True)()

    # Fetch all conversations asynchronously
    conversations = await sync_to_async(
        lambda: list(Conversation.objects.filter(user=user).order_by('-updated_at')),
        thread_sensitive=True
    )()

    # Fetch the current conversation asynchronously
    current_conversation = await sync_to_async(
        lambda: Conversation.objects.filter(user=user, conversation_id=conversation_id).order_by('-updated_at').first(),
        thread_sensitive=True
    )()

    # Prepare the response data
    response_data = {
        'conversations': [
            {
                'id': str(conv.conversation_id),
                'title': conv.title,
                'updated_at': conv.updated_at,
            } for conv in conversations
        ],
        'current_conversation': {
            'id': str(current_conversation.conversation_id) if current_conversation else None,
            'title': current_conversation.title if current_conversation else None,
        }
    }

    return JsonResponse(response_data)

@login_required
@csrf_protect
@require_http_methods(["POST"])
async def create_new_chat(request):
    """
    API endpoint that creates a new conversation or returns existing empty conversation.
    Returns JSON with conversation ID and redirect URL.
    """
    try:
        conversation, message = await ChatService.create_or_get_empty_chat(request.user)
        
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

    
@csrf_protect
@require_http_methods(["POST"])
async def chat_api(request):
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

        # Extract model configuration if provided, else use defaults
        model_config = data.get("model_config", {})
        temperature = model_config.get("temperature", 0.0)
        model_provider = ModelProvider(model_config.get("model_provider", ModelProvider.GROQ.value))
        model_name_str = model_config.get("model_name", GroqModelName.LLAMA_3_3_70B.value)

        # Get the corresponding model Enum class
        ModelEnum = model_provider.get_model_enum()

        # Validate and select the correct model or raise an error
        if model_name_str not in ModelEnum.get_model_names():
            return JsonResponse({
                'error': f"Invalid model name '{model_name_str}' for provider '{model_provider.value}'. ",
                'message': f"Available models: {ModelEnum.get_model_names()}"
            }, status=400)
        
        # Assign the validated model name
        model_name = ModelEnum(model_name_str)

        # Retrieve or create chatbot instance
        chatbot = await get_chatbot_instance(model_provider, model_name, temperature)  # Changed to await

        # Get response from chatbot
        response = await chatbot.chat(user_message, thread_id)  # Changed to await

        conversation = await Conversation.objects.aget(conversation_id=thread_id)  # Changed to await

        if conversation.title == "New Chat":
            conversation.title = await generate_chat_title(user_message)
            await conversation.asave()  # Changed to await

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
        logger.error(f"Error processing chat request: {str(e)}\n{traceback.format_exc()}")
        return JsonResponse({
            'error': f'Error processing request: {str(e)}',
            'thread_id': thread_id,
        }, status=500)


@login_required
@require_http_methods(["GET"])
async def get_conversation_history(request, conversation_id):
    """Get the history of a specific conversation"""
    try:

        # Retrieve or create chatbot instance
        temperature = 0.0
        model_provider = ModelProvider.GROQ
        model_name = GroqModelName.LLAMA_3_3_70B
        chatbot = await get_chatbot_instance(model_provider, model_name, temperature)

        messages = await chatbot.storage.load_conversation(conversation_id)
        
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
        logger.error(f"Error fetching conversation history: {str(e)}\n{traceback.format_exc()}")
        return JsonResponse({
            'error': f'Error fetching conversation history: {str(e)}'
        }, status=500)
    



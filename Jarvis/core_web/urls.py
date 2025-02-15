from django.urls import path
from . import chat_views
from . import search_views


urlpatterns = [
    path('', chat_views.home_view, name='home'),
]

# Chat with LLM
urlpatterns += [

    # views
    path('chat/', chat_views.chat_home, name='chat_home'),
    path('chat/<str:conversation_id>/', chat_views.chat_home, name='chat_with_id'),

    # APIs
    path('api/chat/new/', chat_views.create_new_chat, name='new_chat'),
    path('api/chat/', chat_views.chat_api, name='chat_api'),
    path('api/conversations/<str:conversation_id>/', chat_views.get_conversations, name='get_all_conversations'),
    path('api/conversation/<str:conversation_id>/', chat_views.get_conversation_history, name='conversation_history'),
]


#  Search with LLM
urlpatterns += [

    # views
    path('search/', search_views.search_with_llm_view, name='search_with_llm'),
]
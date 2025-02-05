from django.urls import path
from . import views


urlpatterns = [
    path('', views.home_view, name='home'),
    path('chat/', views.chat_view, name='chat_home'),
    path('chat/new/', views.create_new_chat, name='new_chat'),
    path('chat/<str:conversation_id>/', views.chat_view, name='chat_with_id'),
    path('search/', views.search_with_llm_view, name='search_with_llm'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/conversation/<str:conversation_id>/', 
         views.get_conversation_history, 
         name='conversation_history'),
]

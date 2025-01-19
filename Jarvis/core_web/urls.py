
from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from core_web import views


urlpatterns = [
    path('home', views.home_view, name='home'),
    path('chat', views.chat_with_llm_view, name='chat_with_llm'),
    path('search', views.search_with_llm_view, name='search_with_llm'),
]

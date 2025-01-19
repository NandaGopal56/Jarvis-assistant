from django.shortcuts import render
from django.contrib.auth.decorators import login_required


# Home Page
def home_view(request):
    return render(request, 'home.html')

# Chat with LLM
@login_required
def chat_with_llm_view(request):
    return render(request, 'chat_with_llm.html')

# Search with LLM
@login_required
def search_with_llm_view(request):
    return render(request, 'search_with_llm.html')
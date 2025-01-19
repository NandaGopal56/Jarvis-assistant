from django.shortcuts import render


# Home Page
def home_view(request):
    return render(request, 'home.html')

# Chat with LLM
def chat_with_llm_view(request):
    return render(request, 'chat_with_llm.html')

# Search with LLM
def search_with_llm_view(request):
    return render(request, 'search_with_llm.html')
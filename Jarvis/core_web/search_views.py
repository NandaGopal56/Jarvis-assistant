from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# Search with LLM
@login_required
def search_with_llm_view(request):
    return render(request, 'search_with_llm.html')

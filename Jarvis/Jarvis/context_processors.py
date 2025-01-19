def navbar_context(request):
    return {
        'user': request.user
    }

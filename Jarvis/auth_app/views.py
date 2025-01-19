
# Views
from django.shortcuts import render, redirect
from django.contrib import messages
from auth_app.models import User
from auth_app.services import UserRegistrationService, AuthenticationService

# Registration View
def register(request):
    """
    Handles user registration for template views.
    """
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Call the registration service
        result = UserRegistrationService.register_user(
            email=email, 
            password=password
        )

        if result['success']:
            messages.success(request, result['message'])
            return redirect('login')
        else:
            messages.error(request, result['message'])

    return render(request, 'register.html')


# Email Verification View
def verify_email(request, uidb64, token):
    user = UserRegistrationService.verify_email(uidb64, token)
    if user:
        messages.success(request, "Your email has been successfully verified. You can now log in.")
        return redirect('login')
    else:
        messages.error(request, "The verification link is invalid or has expired.")
        return render(request, 'verification_failed.html')
    
# Login View
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        if AuthenticationService.login_user(request, email, password):
            return redirect('home')  # Redirect to the home page on successful login
    return render(request, 'login.html')

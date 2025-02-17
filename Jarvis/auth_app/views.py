
# Views
from django.shortcuts import render, redirect
from django.contrib import messages
from auth_app.services import UserRegistrationService, AuthenticationService, update_user_profile
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from .forms import UserProfileForm



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


def verify_email(request, uidb64, token):
    """
    Handles email verification for template views.
    """
    user = UserRegistrationService.verify_email(uidb64, token)
    if user:
        messages.success(request, "Your email has been successfully verified. You can now log in.")
        return redirect('login')
    else:
        messages.error(request, "The verification link is invalid or has expired.")
        return render(request, 'verification_failed.html')


def login_view(request):
    """
    Handles user login for template views.
    """
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        if AuthenticationService.login_user(request, email, password):
            # Redirect to the home page on successful login
            return redirect('home')  

    return render(request, 'login.html')


@login_required
def profile_view(request):
    """
    Handles user profile view for template views.
    """
    user = request.user

    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully!")
            return redirect('profile')  # Redirect after successful form submission
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UserProfileForm(instance=user)

    return render(request, 'profile.html', {'form': form})


def logout_view(request):
    """
    This view logs out the user and redirects them to the homepage.
    """
    logout(request)
    return redirect('home') 
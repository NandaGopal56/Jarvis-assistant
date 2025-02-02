
# Views
from django.shortcuts import render, redirect
from django.contrib import messages
from auth_app.services import UserRegistrationService, AuthenticationService, update_user_profile
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from .forms import UserProfileForm


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
    # Redirect to home if the user is already logged in
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


# Logout View
def logout_view(request):
    """
    This view logs out the user and redirects them to the homepage.
    """
    logout(request)
    return redirect('home') 
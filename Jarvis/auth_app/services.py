from django.contrib.auth import get_user_model
from django.contrib import messages
from typing import Dict, Any, Optional
from django.db import transaction
from django.contrib.auth import authenticate, login
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.urls import reverse
from django.conf import settings
from django.contrib.auth.models import User

User = get_user_model()

class UserRegistrationService:

    @staticmethod
    @transaction.atomic
    def register_user(
        email: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handles user registration logic.

        Args:
            email (str): The email of the user.
            password (str): The password for the user account.
            first_name (str, optional): The first name of the user. Defaults to None.
            last_name (str, optional): The last name of the user. Defaults to None.

        Returns:
            dict: A dictionary containing success, user object, and message.
        """
        try:
            # Check if the email already exists
            user = User.objects.filter(email=email)

            if user.exists():
                if user[0].is_active == True:
                    return {
                        "success": False,
                        "message": "A user with this email already exists.",
                    }
                else:
                    user[0].delete()

            # Create user
            user = User.objects.create_user(
                email=email, password=password, first_name=first_name, last_name=last_name
            )
            
            # Set user as inactive until verification
            user.is_active = False
            user.save()

            # Trigger email verification
            UserRegistrationService.send_verification_email(user)

            return {
                "success": True,
                "user": user,
                "message": "Registration successful. Please verify your email.",
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e),
            }
        

    @staticmethod
    def send_verification_email(user):
        """
        Sends a one-time email verification link to the user's email.
        """
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        verification_url = reverse('verify_email', kwargs={'uidb64': uid, 'token': token})
        verification_link = f"{settings.SITE_URL}{verification_url}"

        subject = "Verify Your Email Address"
        message = f"Hi {user.email},\n\nPlease click the link below to verify your email address:\n{verification_link}\n\nThis link will expire after use or after a certain time."
        from_email = settings.DEFAULT_FROM_EMAIL
        recipient_list = [user.email]

        send_mail(subject, message, from_email, recipient_list)


    @staticmethod
    def verify_email(uidb64, token):
        """
        Verifies the user's email using the provided UID and token.
        """
        from auth_app.models import User
        from django.utils.http import urlsafe_base64_decode
        from django.core.exceptions import ValidationError

        try:
            uid = urlsafe_base64_decode(uidb64).decode()
            user = User.objects.get(pk=uid)
        except (User.DoesNotExist, ValueError, TypeError, ValidationError):
            return None  # Invalid UID

        if default_token_generator.check_token(user, token):
            user.is_active = True
            user.save()
            return user
        return None




class AuthenticationService:
    @staticmethod
    def login_user(request, email: str, password: str) -> bool:
        """
        Authenticates and logs in the user.

        Args:
            request: The HTTP request object.
            email (str): The user's email.
            password (str): The user's password.

        Returns:
            bool: True if login was successful, False otherwise.
        """
        user = authenticate(request, email=email, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return True
            else:
                messages.error(request, 'Your account is inactive. Please verify your email.')
        else:
            messages.error(request, 'Invalid email or password.')
        return False
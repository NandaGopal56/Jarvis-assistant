# from django.core.mail import send_mail
# from django.db.models.signals import post_save
# from django.dispatch import receiver
# from auth_app.models import User, OTP
# import random


# # Signals
# @receiver(post_save, sender=User)
# def send_verification_email(sender, instance, created, **kwargs):
#     if created and not instance.is_active:
#         otp_code = ''.join(random.choices('0123456789', k=4))
#         OTP.objects.create(user=instance, otp_code=otp_code)
#         send_mail(
#             subject='Verify your account',
#             message=f'Your verification code is {otp_code}',
#             from_email='no-reply@example.com',
#             recipient_list=[instance.email],
#         )

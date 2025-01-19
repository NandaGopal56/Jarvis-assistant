
from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from auth_app import views


urlpatterns = [
    path('register', views.register, name='register'),
    path('verify-email/<uidb64>/<token>/', views.verify_email, name='verify_email'),
    path('login', views.login_view, name='login'),
    # path('reset_password/', views.reset_password, name='reset_password'),
    # path('', views.home, name='home'),
]

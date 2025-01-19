from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
from typing import Optional
from enum import Enum


# Enums for Role and Gender
class RoleChoices(Enum):
    SUPERADMIN = 'superadmin'
    STAFF = 'staff'
    USER = 'user'

    @classmethod
    def choices(cls):
        return [(tag.value, tag.name.capitalize()) for tag in cls]

class GenderChoices(Enum):
    MALE = 'male'
    FEMALE = 'female'

    @classmethod
    def choices(cls):
        return [(tag.value, tag.name.capitalize()) for tag in cls]

# Custom User Manager
class UserManager(BaseUserManager):
    def create_user(self, email: str, password: Optional[str] = None, role: str = RoleChoices.USER.value, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        if role not in [choice[0] for choice in RoleChoices.choices()]:
            raise ValueError(f'Invalid role. Choose from {[choice[0] for choice in RoleChoices.choices()]}')
        extra_fields.setdefault('is_active', False)  # User remains inactive until email verification
        extra_fields['role'] = role
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email: str, password: str, **extra_fields):
        extra_fields.setdefault('is_active', True)
        return self.create_user(email, password, role=RoleChoices.SUPERADMIN.value, **extra_fields)

# Custom User Model
class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=50, blank=True, null=True)
    last_name = models.CharField(max_length=50, blank=True, null=True)
    gender = models.CharField(max_length=10, choices=GenderChoices.choices(), blank=True, null=True)
    mobile = models.CharField(max_length=15, blank=True, null=True)
    country = models.CharField(max_length=50, blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    is_active = models.BooleanField(default=False)  # User remains inactive until verified
    date_joined = models.DateTimeField(default=timezone.now)
    last_login = models.DateTimeField(blank=True, null=True)  # Uses Django's default functionality
    role = models.CharField(max_length=20, choices=RoleChoices.choices(), default=RoleChoices.USER.value)

    
    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['role', 'password']

    class Meta:
        db_table = 'User'


    def __str__(self):
        return self.email

    def is_staff(self):
        return self.role == RoleChoices.STAFF.value

    def is_superuser(self):
        return self.role == RoleChoices.SUPERADMIN.value

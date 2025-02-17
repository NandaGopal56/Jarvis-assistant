from django.contrib import admin
from .models import Conversation, MessagePair

# Register your models here.
admin.site.register(Conversation)
admin.site.register(MessagePair)
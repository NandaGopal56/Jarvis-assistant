from django.db import models
from django.conf import settings
from django.utils import timezone
from django.db.models import JSONField

# Create your models here.

class Conversation(models.Model):
    conversation_id = models.BigAutoField(editable=False, primary_key=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255, blank=True)

    # metadata
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'conversations'
        indexes = [
            models.Index(fields=['user', 'conversation_id'])
        ]
        ordering = ['-updated_at']

    def __str__(self):
        return f"Conversation {self.conversation_id} - {self.title or 'Untitled'}"

class MessagePair(models.Model):
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]

    message_pair_id = models.BigAutoField(editable=False, primary_key=True)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='message_pairs')
    
    # User message
    user_message = models.TextField()
    user_message_timestamp = models.DateTimeField(default=timezone.now)
    
    # AI response
    ai_message = models.TextField()
    ai_message_timestamp = models.DateTimeField(auto_now=True)
    
    # Summary and metadata
    summary = models.TextField(null=True, blank=True, help_text="Brief summary of this message exchange")
    tokens_used = JSONField(default=dict, help_text="Detailed token usage breakdown", blank=True)
    model_version = models.CharField(max_length=50, blank=True)
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='completed')
    processing_time = models.FloatField(null=True, blank=True)  # in seconds
    error_message = models.TextField(blank=True)

    # metadata
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'MessagePair'
        indexes = [
            models.Index(fields=['conversation', 'message_pair_id'])
        ]
        ordering = ['user_message_timestamp']

    def __str__(self):
        return f"Message Pair {self.message_pair_id} in conversation {self.conversation_id}"

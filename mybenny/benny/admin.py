from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import LocationUser, ChatSession, ChatLog

@admin.register(LocationUser)
class LocationUserAdmin(admin.ModelAdmin):
    list_display = ['id', 'name']
    search_fields = ['name']


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'location', 'session_id', 'started_at']
    search_fields = ['location__name', 'session_id']
    list_filter = ['location']


@admin.register(ChatLog)
class ChatLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'question', 'answer', 'timestamp']
    search_fields = ['question', 'answer']
    list_filter = ['session__location']

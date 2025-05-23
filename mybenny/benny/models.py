from django.db import models

class LocationUser(models.Model):
    """
    Represents a location (e.g., building, room, department) acting as a user.
    """
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class ChatSession(models.Model):
    """
    Represents a single session (e.g., one login instance) for a location user.
    """
    location = models.ForeignKey(LocationUser, on_delete=models.CASCADE, related_name='sessions')
    session_id = models.CharField(max_length=100, unique=True)
    started_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.session_id} at {self.location.name}"


class ChatLog(models.Model):
    """
    Logs each question and answer in a session.
    """
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='logs')
    question = models.TextField()
    answer = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Q: {self.question[:30]}... | A: {self.answer[:30]}..."

from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
import os

class AudioUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='audio_uploads')
    original_filename = models.CharField(max_length=255)
    original_audio_file = models.FileField(upload_to='audio/original/', null=True, blank=True)
    denoised_audio_file = models.FileField(upload_to='audio/denoised/', null=True, blank=True)
    file_size = models.PositiveIntegerField(null=True, blank=True)  # in bytes
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        
    def __str__(self):
        return f"{self.original_filename} - {self.user.username}"
    
    def get_file_size_display(self):
        """Return human readable file size"""
        if not self.file_size:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size < 1024.0:
                return f"{self.file_size:.1f} {unit}"
            self.file_size /= 1024.0
        return f"{self.file_size:.1f} TB"
    
    def delete(self, *args, **kwargs):
        """Delete associated files when deleting the model"""
        if self.original_audio_file:
            if default_storage.exists(self.original_audio_file.name):
                default_storage.delete(self.original_audio_file.name)
        
        if self.denoised_audio_file:
            if default_storage.exists(self.denoised_audio_file.name):
                default_storage.delete(self.denoised_audio_file.name)
        
        super().delete(*args, **kwargs)

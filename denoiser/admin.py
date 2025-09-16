from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import AudioUpload

# Unregister the default User admin
admin.site.unregister(User)

# Create custom User admin with enhanced features
@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'is_active', 'date_joined', 'last_login']
    list_filter = ['is_active', 'is_staff', 'date_joined', 'last_login']
    search_fields = ['username', 'email', 'first_name', 'last_name']
    ordering = ['-date_joined']
    
    # Add custom actions
    actions = ['activate_users', 'deactivate_users']
    
    def activate_users(self, request, queryset):
        queryset.update(is_active=True)
        self.message_user(request, f'{queryset.count()} users have been activated.')
    activate_users.short_description = "Activate selected users"
    
    def deactivate_users(self, request, queryset):
        queryset.update(is_active=False)
        self.message_user(request, f'{queryset.count()} users have been deactivated.')
    deactivate_users.short_description = "Deactivate selected users"

# AudioUpload admin
@admin.register(AudioUpload)
class AudioUploadAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'user', 'uploaded_at', 'processed_at', 'file_size_display']
    list_filter = ['uploaded_at', 'processed_at']
    search_fields = ['original_filename', 'user__username', 'user__email']
    ordering = ['-uploaded_at']
    readonly_fields = ['uploaded_at', 'processed_at']
    
    def file_size_display(self, obj):
        if obj.file_size:
            size = obj.file_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        return "Unknown"
    file_size_display.short_description = "File Size"

# Customize admin site header
admin.site.site_header = "Denoisify Administration"
admin.site.site_title = "Denoisify Admin"
admin.site.index_title = "Welcome to Denoisify Administration"

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.models import User
from .models import AudioUpload
from .auth_forms import CustomUserUpdateForm
import os


@login_required
def profile_view(request):
    """User profile overview page"""
    # Get recent audio uploads
    recent_uploads = AudioUpload.objects.filter(user=request.user).order_by('-uploaded_at')[:5]
    
    # Get statistics
    total_uploads = AudioUpload.objects.filter(user=request.user).count()
    
    context = {
        'user': request.user,
        'recent_uploads': recent_uploads,
        'total_uploads': total_uploads,
    }
    return render(request, 'profile.html', context)


@login_required
def edit_profile_view(request):
    """Edit user profile credentials"""
    if request.method == 'POST':
        form = CustomUserUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated successfully!')
            return redirect('denoiser:profile')
    else:
        form = CustomUserUpdateForm(instance=request.user)
    
    return render(request, 'edit_profile.html', {'form': form})


@login_required
def change_password_view(request):
    """Change user password"""
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Important to keep user logged in
            messages.success(request, 'Your password has been changed successfully!')
            return redirect('denoiser:profile')
    else:
        form = PasswordChangeForm(request.user)
    
    return render(request, 'change_password.html', {'form': form})


@login_required
def audio_history_view(request):
    """Display user's audio upload history"""
    # Get search query
    search_query = request.GET.get('search', '')
    
    # Filter uploads based on search
    uploads = AudioUpload.objects.filter(user=request.user)
    
    if search_query:
        uploads = uploads.filter(
            Q(original_filename__icontains=search_query) |
            Q(task_id__icontains=search_query)
        )
    
    uploads = uploads.order_by('-uploaded_at')
    
    # Pagination
    paginator = Paginator(uploads, 10)  # Show 10 uploads per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_uploads': uploads.count(),
    }
    return render(request, 'audio_history.html', context)


@login_required
@require_http_methods(["DELETE", "POST"])
def delete_audio_upload(request, upload_id):
    """Delete an audio upload record and associated files"""
    try:
        upload = AudioUpload.objects.get(id=upload_id, user=request.user)
        filename = upload.original_filename or "Untitled"
        
        # Delete associated files if they exist
        try:
            if upload.original_audio_file and os.path.exists(upload.original_audio_file.path):
                os.remove(upload.original_audio_file.path)
        except:
            pass
        
        try:
            if upload.denoised_audio_file and os.path.exists(upload.denoised_audio_file.path):
                os.remove(upload.denoised_audio_file.path)
        except:
            pass
        
        # Delete the database record
        upload.delete()
        
        return JsonResponse({
            'success': True,
            'message': f'"{filename}" deleted successfully!'
        })
    except AudioUpload.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': 'Audio upload not found or you do not have permission to delete it.'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }, status=500)


@login_required
def download_audio_file(request, upload_id, file_type):
    """Download original or processed audio file"""
    try:
        upload = AudioUpload.objects.get(id=upload_id, user=request.user)
        
        if file_type == 'original' and upload.original_audio_file:
            file_path = upload.original_audio_file.path
            filename = f"original_{upload.original_filename}"
        elif file_type == 'processed' and upload.denoised_audio_file:
            file_path = upload.denoised_audio_file.path
            filename = f"denoised_{upload.original_filename}"
        else:
            return JsonResponse({
                'success': False,
                'message': 'File not found.'
            }, status=404)
        
        if os.path.exists(file_path):
            from django.http import FileResponse
            response = FileResponse(
                open(file_path, 'rb'),
                as_attachment=True,
                filename=filename
            )
            return response
        else:
            return JsonResponse({
                'success': False,
                'message': 'File not found on server.'
            }, status=404)
            
    except AudioUpload.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': 'Audio upload not found or you do not have permission to access it.'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }, status=500)

from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.conf import settings
from .auth_forms import CustomUserCreationForm, CustomAuthenticationForm


def register_view(request):
    if request.user.is_authenticated:
        return redirect('denoiser:home')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            
            # Preserve next parameter when redirecting to login
            next_url = request.POST.get('next') or request.GET.get('next')
            if next_url:
                return redirect(f'/login/?next={next_url}')
            else:
                return redirect('denoiser:login')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('denoiser:home')
    
    # Debug: Check session data at start of login
    print(f"Login view - Session key: {request.session.session_key}")
    print(f"Session data keys: {list(request.session.keys())}")
    if 'guest_uploads' in request.session:
        print(f"Found guest_uploads in session with {len(request.session['guest_uploads'])} items")
    else:
        print("No guest_uploads found in session")
    
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                
                # Associate any guest uploads with the newly logged in user
                if 'guest_uploads' in request.session:
                    from .models import AudioUpload
                    from django.core.files.base import ContentFile
                    import os
                    
                    guest_uploads = request.session['guest_uploads']
                    print(f"Found {len(guest_uploads)} guest uploads to associate")
                    
                    # Ensure guest_uploads is a list (handle session data corruption)
                    if not isinstance(guest_uploads, list):
                        print("Warning: guest_uploads is not a list, skipping association")
                        del request.session['guest_uploads']
                        request.session.modified = True
                    else:
                        successful_associations = 0
                        for i, upload_info in enumerate(guest_uploads):
                            try:
                                print(f"Processing guest upload {i+1}: {upload_info['original_filename']}")
                                
                                # Create AudioUpload record for guest upload
                                audio_upload = AudioUpload.objects.create(
                                    user=user,
                                    original_filename=upload_info['original_filename'],
                                    file_size=upload_info['file_size']
                                )
                                print(f"Created AudioUpload record {audio_upload.id}")
                                
                                # If the input file still exists, save it to the model
                                input_path = upload_info['input_path']
                                print(f"Checking input file: {input_path}")
                                if os.path.exists(input_path):
                                    print(f"Input file exists, saving to model")
                                    with open(input_path, 'rb') as f:
                                        input_filename = f"{upload_info['timestamp']}_{upload_info['original_filename']}"
                                        audio_upload.original_audio_file.save(
                                            input_filename,
                                            ContentFile(f.read()),
                                            save=True
                                        )
                                    print(f"Saved original file: {input_filename}")
                                else:
                                    print(f"Input file not found: {input_path}")
                                
                                # Check if processed file exists and save it too
                                output_path = os.path.join(
                                    settings.MEDIA_ROOT,
                                    "audio_results", 
                                    upload_info['base_filename'], 
                                    "postprocessed_output_mmse_stsa.wav"
                                )
                                print(f"Checking processed file: {output_path}")
                                if os.path.exists(output_path):
                                    print(f"Processed file exists, saving to model")
                                    with open(output_path, 'rb') as f:
                                        output_filename = f"denoised_{upload_info['original_filename']}"
                                        audio_upload.denoised_audio_file.save(
                                            output_filename,
                                            ContentFile(f.read()),
                                            save=False
                                        )
                                    print(f"Saved processed file: {output_filename}")
                                else:
                                    print(f"Processed file not found: {output_path}")
                                
                                # Set processed timestamp and save
                                audio_upload.processed_at = timezone.now()
                                audio_upload.save()
                                successful_associations += 1
                                print(f"Successfully associated guest upload {audio_upload.id}")
                            except Exception as e:
                                print(f"Error associating guest upload {i+1}: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        print(f"Associated {successful_associations} out of {len(guest_uploads)} uploads")
                        
                        # Clear guest uploads from session
                        del request.session['guest_uploads']
                        request.session.modified = True
                else:
                    print("No guest uploads found in session")
                
                # Get next URL from POST data (form) or GET data (URL parameter)
                next_url = request.POST.get('next') or request.GET.get('next', 'denoiser:home')
                return redirect(next_url)
    else:
        form = CustomAuthenticationForm()
    
    return render(request, 'login.html', {'form': form})


@require_http_methods(["POST"])
def logout_confirm(request):
    """Handle logout confirmation via AJAX or form submission"""
    if request.user.is_authenticated:
        username = request.user.username
        logout(request)
        
        # Check if this is an AJAX request by looking at headers or content type
        is_ajax = (
            request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
            request.headers.get('Content-Type') == 'application/json' or
            'application/json' in request.headers.get('Accept', '')
        )
        
        if is_ajax:
            # Return JSON response for AJAX requests (like from landing page)
            return JsonResponse({
                'success': True,
                'message': f'Goodbye {username}! You have been logged out successfully.',
                'redirect_url': '/'
            })
        else:
            # Redirect directly for form submissions (like from other pages)
            from urllib.parse import urlencode
            params = urlencode({'logout_success': f'Goodbye {username}! You have been logged out successfully.'})
            return redirect(f'/?{params}')
    
    # User not authenticated
    if request.headers.get('Content-Type') == 'application/json':
        return JsonResponse({
            'success': False,
            'message': 'You are not logged in.'
        })
    else:
        from urllib.parse import urlencode
        params = urlencode({'logout_error': 'You are not logged in.'})
        return redirect(f'/?{params}')


@require_http_methods(["POST"])
def check_auth_status(request):
    """AJAX endpoint to check if user is authenticated"""
    return JsonResponse({
        'authenticated': request.user.is_authenticated,
        'username': request.user.username if request.user.is_authenticated else None
    })

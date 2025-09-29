# denoiser/views.py

import os
import time
import shutil
import uuid
import threading
from django.shortcuts import render, redirect
from django.http import HttpResponse, FileResponse, JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.core.files.base import ContentFile
from .forms import AudioUploadForm
from .models import AudioUpload
from .model_loader import denoise_audio, plot_waveform
from .audio_utils import validate_audio_file, convert_to_wav, get_file_extension
from .progress import ProcessingProgress, get_stage_progress
import wave
import numpy as np

def process_audio_with_progress(input_path, audio_folder, processing_method, task_id, base_filename, audio_upload_id=None):
    """Process audio with progress tracking"""
    progress = ProcessingProgress(task_id)
    
    try:
        # Stage 1: Validation
        progress.update_progress('validation', get_stage_progress('validation'), 
                               'Validating audio file...')
        
        is_valid, validation_message = validate_audio_file(input_path)
        if not is_valid:
            progress.set_error(f"Invalid audio file: {validation_message}")
            return False
            
        progress.update_progress('validation', get_stage_progress('validation', 100), 
                               'Audio file validated successfully')
        
        # Stage 2: Loading Model
        progress.update_progress('loading_model', get_stage_progress('loading_model'), 
                               'Loading AI denoising model...')
        
        # Stage 3: Processing (this will update progress internally)
        progress.update_progress('denoising', get_stage_progress('denoising'), 
                               'Processing audio...')
        
        # Set output path for denoising
        output_path = os.path.join(settings.MEDIA_ROOT, f"denoised_{os.path.basename(input_path)}")
        
        # Call the actual denoising function
        result = denoise_audio(input_path, output_path, processing_method)
        
        # Stage 4: Generating Visualizations
        progress.update_progress('generating_visualizations', get_stage_progress('generating_visualizations'), 
                               'Generating audio visualizations...')
        
        # Stage 5: Finalizing
        progress.update_progress('finalizing', get_stage_progress('finalizing'), 
                               'Moving files to results folder...')
        
        # Ensure audio folder exists
        os.makedirs(audio_folder, exist_ok=True)
        
        # Copy original input file to results folder for comparison
        original_filename = os.path.basename(input_path)
        dest_input_path = os.path.join(audio_folder, original_filename)
        if os.path.exists(input_path):
            shutil.copy2(input_path, dest_input_path)
            print(f"Copied original file to results: {dest_input_path}")
        
        visualization_files = [
            'input_waveform.png', 'input_spectrogram.png',
            'denoised_waveform.png', 'denoised_output_spectrogram.png',
            'postprocessed_waveform.png', 'postprocessed_output_mmse_stsa_spectrogram.png',
            'postprocessed_output_mmse_stsa.wav'
        ]
        
        moved_files = []
        for file in visualization_files:
            src_path = os.path.join(os.getcwd(), file)
            if os.path.exists(src_path):
                dest_path = os.path.join(audio_folder, file)
                shutil.move(src_path, dest_path)
                moved_files.append(file)
                print(f"Moving {file} to {audio_folder}")
        
        # Update AudioUpload record with denoised file if available
        if audio_upload_id:
            try:
                audio_upload = AudioUpload.objects.get(id=audio_upload_id)
                
                # Save the denoised file to the AudioUpload model
                denoised_file_path = os.path.join(audio_folder, "postprocessed_output_mmse_stsa.wav")
                if os.path.exists(denoised_file_path):
                    with open(denoised_file_path, 'rb') as f:
                        audio_upload.denoised_audio_file.save(
                            f"denoised_{audio_upload.original_filename}",
                            ContentFile(f.read()),
                            save=False
                        )
                    
                    # Update processed timestamp
                    audio_upload.processed_at = timezone.now()
                    audio_upload.save()
                    print(f"Updated AudioUpload record {audio_upload_id} with denoised file")
                    
            except AudioUpload.DoesNotExist:
                print(f"AudioUpload with ID {audio_upload_id} not found")
            except Exception as e:
                print(f"Error updating AudioUpload record: {e}")
        
        # Get file sizes for results
        input_size = 0
        output_size = 0
        
        if os.path.exists(input_path):
            input_size = round(os.path.getsize(input_path) / 1024, 2)
        
        output_file = os.path.join(audio_folder, "postprocessed_output_mmse_stsa.wav")
        if os.path.exists(output_file):
            output_size = round(os.path.getsize(output_file) / 1024, 2)
        
        # Complete
        progress.complete(
            'Audio processing completed successfully!',
            {
                'folder_name': base_filename, 
                'moved_files': moved_files,
                'input_size': input_size,
                'output_size': output_size,
                'processing_method': processing_method
            }
        )
        
        return True
        
    except Exception as e:
        error_msg = f"Error processing audio: {str(e)}"
        print(f"ERROR: {error_msg}")
        progress.set_error(error_msg, {'exception': str(e)})
        return False

def index(request):
    context = {}
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            processing_method = 'waveunet_only'
            
            # Create unique filenames with timestamp
            timestamp = str(int(time.time()))
            base_filename = os.path.splitext(uploaded_file.name)[0]
            input_filename = f"{timestamp}_{uploaded_file.name}"
            output_filename = f"{timestamp}_denoised_{uploaded_file.name}"
            
            input_path = os.path.join(settings.MEDIA_ROOT, input_filename)
            output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

            # Create organized folder structure
            audio_folder = os.path.join(settings.MEDIA_ROOT, "audio_results", base_filename)
            os.makedirs(audio_folder, exist_ok=True)

            # Save uploaded file
            converted_file = None  # Initialize variable to track converted files
            try:
                with open(input_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                
                print(f"File saved to: {input_path}")
                print(f"File size: {os.path.getsize(input_path)} bytes")
                
                # Validate the uploaded WAV file
                is_valid, validation_message = validate_audio_file(input_path)
                print(f"Validation result: {validation_message}")
                
                if not is_valid:
                    context['error'] = validation_message
                    # Clean up files safely
                    try:
                        if os.path.exists(input_path):
                            os.remove(input_path)
                    except:
                        pass  # Ignore cleanup errors
                    context['form'] = form
                    return render(request, 'index.html', context)

                # Create AudioUpload record if user is authenticated
                audio_upload = None
                if request.user.is_authenticated:
                    audio_upload = AudioUpload.objects.create(
                        user=request.user,
                        original_filename=uploaded_file.name,
                        file_size=uploaded_file.size
                    )
                    
                    # Save the uploaded file to the AudioUpload model
                    with open(input_path, 'rb') as f:
                        audio_upload.original_audio_file.save(
                            input_filename,
                            ContentFile(f.read()),
                            save=True
                        )
                    
                    print(f"Created AudioUpload record with ID: {audio_upload.id}")

                # Process audio with enhanced visualization
                print(f"Processing audio: {input_path}")
                print(f"Using processing method: {processing_method}")
                denoise_audio(input_path, output_path, processing_method)
                print(f"Audio processing completed")
                
                # Move all generated files to organized folder
                visualization_files = [
                    'input_waveform.png', 'input_spectrogram.png',
                    'denoised_waveform.png', 'denoised_output_spectrogram.png',
                    'postprocessed_waveform.png', 'postprocessed_output_mmse_stsa_spectrogram.png',
                    'postprocessed_output_mmse_stsa.wav'
                ]
                
                print(f"Moving files to: {audio_folder}")
                for file in visualization_files:
                    if os.path.exists(file):
                        print(f"Moving {file} to {audio_folder}")
                        shutil.move(file, os.path.join(audio_folder, file))
                    else:
                        print(f"File not found: {file}")
                
                # Update AudioUpload record with denoised file if available and user is authenticated
                if audio_upload:
                    denoised_file_path = os.path.join(audio_folder, "postprocessed_output_mmse_stsa.wav")
                    if os.path.exists(denoised_file_path):
                        with open(denoised_file_path, 'rb') as f:
                            audio_upload.denoised_audio_file.save(
                                f"denoised_{audio_upload.original_filename}",
                                ContentFile(f.read()),
                                save=False
                            )
                        
                        # Update processed timestamp
                        audio_upload.processed_at = timezone.now()
                        audio_upload.save()
                        print(f"Updated AudioUpload record {audio_upload.id} with denoised file")
                
                # Get file sizes for comparison
                input_size = os.path.getsize(input_path)
                final_output_path = os.path.join(audio_folder, "postprocessed_output_mmse_stsa.wav")
                final_size = os.path.getsize(final_output_path) if os.path.exists(final_output_path) else 0
                
                context.update({
                    'input_audio': input_filename,
                    'output_audio': "postprocessed_output_mmse_stsa.wav",
                    'audio_folder': f"audio_results/{base_filename}",
                    'folder_name': base_filename,
                    'input_size': round(input_size / 1024, 2),  # KB
                    'output_size': round(final_size / 1024, 2),  # KB
                    'processing_method': 'WaveUNet + MMSE-STSA',
                    'success': True
                })
                
            except Exception as e:
                error_msg = f"Error processing audio: {str(e)}"
                print(f"ERROR: {error_msg}")
                context['error'] = error_msg
                # Clean up uploaded file if processing failed
                if os.path.exists(input_path):
                    os.remove(input_path)
                
    else:
        form = AudioUploadForm()

    context['form'] = form
    return render(request, 'index.html', context)

def audio_processor(request):
    """View that handles audio upload and starts background processing"""
    context = {'success': False}
    
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_file = request.FILES['file']
                processing_method = 'waveunet_only'
                
                # Generate unique task ID and filename
                task_id = str(uuid.uuid4())
                timestamp = str(int(time.time()))
                base_filename = os.path.splitext(uploaded_file.name)[0]
                input_filename = f"{timestamp}_{uploaded_file.name}"
                
                # Create audio folder for this session
                audio_folder = os.path.join(settings.MEDIA_ROOT, "audio_results", base_filename)
                os.makedirs(audio_folder, exist_ok=True)
                
                # Save uploaded file to media folder
                input_path = os.path.join(settings.MEDIA_ROOT, input_filename)
                
                with open(input_path, 'wb') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                
                print(f"File saved to: {input_path}")
                
                # Validate the uploaded WAV file
                is_valid, validation_message = validate_audio_file(input_path)
                print(f"Validation result: {validation_message}")
                
                if not is_valid:
                    context['error'] = validation_message
                    # Clean up files safely
                    try:
                        if os.path.exists(input_path):
                            os.remove(input_path)
                    except:
                        pass  # Ignore cleanup errors
                    context['form'] = form
                    return render(request, 'audio_processor.html', context)
                
                # Create AudioUpload record if user is authenticated
                audio_upload = None
                if request.user.is_authenticated:
                    audio_upload = AudioUpload.objects.create(
                        user=request.user,
                        original_filename=uploaded_file.name,
                        file_size=uploaded_file.size
                    )
                    
                    # Save the uploaded file to the AudioUpload model
                    with open(input_path, 'rb') as f:
                        audio_upload.original_audio_file.save(
                            input_filename,
                            ContentFile(f.read()),
                            save=True
                        )
                    
                    print(f"Created AudioUpload record with ID: {audio_upload.id}")
                else:
                    # For guest users, store upload info in session for later association
                    guest_upload_info = {
                        'task_id': task_id,
                        'original_filename': uploaded_file.name,
                        'file_size': uploaded_file.size,
                        'input_path': input_path,
                        'base_filename': base_filename,
                        'processing_method': processing_method,
                        'timestamp': timestamp
                    }
                    
                    # Initialize or get existing guest uploads list
                    guest_uploads = request.session.get('guest_uploads', [])
                    if not isinstance(guest_uploads, list):
                        guest_uploads = []
                    
                    guest_uploads.append(guest_upload_info)
                    request.session['guest_uploads'] = guest_uploads
                    request.session.modified = True
                    print(f"Stored guest upload info in session for task: {task_id}")
                    print(f"Total guest uploads in session: {len(guest_uploads)}")
                    print(f"Guest upload details: {guest_upload_info}")
                    print(f"Session key: {request.session.session_key}")
                    print(f"Session data keys: {list(request.session.keys())}")

                # Initialize progress tracker
                progress_tracker = ProcessingProgress(task_id)
                progress_tracker.update_progress('upload', 100, 'File uploaded and validated successfully')
                
                # Start background processing with audio_upload ID
                thread = threading.Thread(
                    target=process_audio_with_progress,
                    args=(input_path, audio_folder, processing_method, task_id, base_filename, audio_upload.id if audio_upload else None)
                )
                thread.daemon = True
                thread.start()
                
                # Redirect to progress page
                return redirect('denoiser:progress_page', task_id=task_id)
                
            except Exception as e:
                error_msg = f"Error starting audio processing: {str(e)}"
                print(f"ERROR: {error_msg}")
                context['error'] = error_msg
        else:
            print(f"Form is NOT valid. Errors: {form.errors}")
            context['error'] = "Form validation failed. Please check your input."
            context['form_errors'] = form.errors
                
    else:
        form = AudioUploadForm()

    context['form'] = form
    return render(request, 'audio_processor.html', context)

def test_audio_original(request, folder_name):
    """Serve original audio files with authentication check for downloads only"""
    # Check if this is a download request (vs inline playback)
    is_download = request.GET.get('download', 'false').lower() == 'true'
    
    # Only require authentication for downloads, not for inline playback
    if is_download and not request.user.is_authenticated:
        # Store the current URL for redirect after login
        request.session['download_requested'] = True
        request.session['download_folder'] = folder_name
        messages.info(request, 'Please log in to download your processed audio files.')
        return redirect(f'/login/?next={request.get_full_path()}')
    
    # Find the original audio file in the folder
    audio_folder = os.path.join(settings.MEDIA_ROOT, "audio_results", folder_name)
    
    if not os.path.exists(audio_folder):
        return HttpResponse("Audio folder not found", status=404)
    
    # Get original input files (exclude postprocessed files)
    input_files = [f for f in os.listdir(audio_folder) 
                  if f.endswith('.wav') and not f.startswith('postprocessed_') and not f.startswith('denoised_')]
    
    if not input_files:
        return HttpResponse("Original audio file not found", status=404)
    
    original_audio_path = os.path.join(audio_folder, input_files[0])
    
    if os.path.exists(original_audio_path):
        response = FileResponse(open(original_audio_path, 'rb'), content_type='audio/wav')
        if is_download:
            response['Content-Disposition'] = 'attachment; filename="original_audio.wav"'
        else:
            response['Content-Disposition'] = 'inline; filename="original_audio.wav"'
        return response
    else:
        return HttpResponse(f"Original audio file not found: {original_audio_path}", status=404)


def test_audio(request, folder_name):
    """Test view to serve audio files directly with proper MIME type - requires authentication for downloads only"""
    # Check if this is a download request (vs inline playback)
    is_download = request.GET.get('download', 'false').lower() == 'true'
    
    # Only require authentication for downloads, not for inline playback
    if is_download and not request.user.is_authenticated:
        # Store the current URL for redirect after login
        request.session['download_requested'] = True
        request.session['download_folder'] = folder_name
        messages.info(request, 'Please log in to download your processed audio files.')
        return redirect(f'/login/?next={request.get_full_path()}')
    
    audio_path = os.path.join(settings.MEDIA_ROOT, "audio_results", folder_name, "postprocessed_output_mmse_stsa.wav")
    
    if os.path.exists(audio_path):
        response = FileResponse(open(audio_path, 'rb'), content_type='audio/wav')
        if is_download:
            response['Content-Disposition'] = 'attachment; filename="processed_audio.wav"'
        else:
            response['Content-Disposition'] = 'inline; filename="processed_audio.wav"'
        return response
    else:
        return HttpResponse(f"Audio file not found: {audio_path}", status=404)

@require_http_methods(["GET"])
def get_progress(request, task_id):
    """API endpoint to get processing progress"""
    progress_tracker = ProcessingProgress(task_id)
    progress_data = progress_tracker.get_progress()
    
    if progress_data:
        return JsonResponse(progress_data)
    else:
        return JsonResponse({
            'task_id': task_id,
            'stage': 'not_found',
            'percentage': 0,
            'message': 'Task not found',
            'status': 'error'
        }, status=404)

def progress_page(request, task_id):
    """Show progress page for a processing task"""
    context = {
        'task_id': task_id,
        'start_time': time.strftime('%H:%M:%S')
    }
    return render(request, 'progress.html', context) 

def debug_session(request):
    """Debug view to check session data"""
    session_data = {
        'session_key': request.session.session_key,
        'session_keys': list(request.session.keys()),
        'is_authenticated': request.user.is_authenticated,
        'username': request.user.username if request.user.is_authenticated else None,
        'guest_uploads': request.session.get('guest_uploads', [])
    }
    
    return JsonResponse({
        'session_info': session_data,
        'guest_upload_count': len(session_data['guest_uploads'])
    })


def results_page(request, folder_name):
    """Show results page with registration prompt"""
    # Check if the folder exists
    audio_folder = os.path.join(settings.MEDIA_ROOT, "audio_results", folder_name)
    
    if not os.path.exists(audio_folder):
        return HttpResponse("Results not found", status=404)
    
    # Get file sizes if available
    input_files = [f for f in os.listdir(audio_folder) 
                  if f.endswith('.wav') and not f.startswith('postprocessed_') and not f.startswith('denoised_')]
    input_filename = input_files[0] if input_files else None
    
    input_size = 0
    output_size = 0
    input_quality = {}
    output_quality = {}
    
    if input_filename:
        input_path = os.path.join(audio_folder, input_filename)
        if os.path.exists(input_path):
            input_size = round(os.path.getsize(input_path) / 1024, 2)
            # Get audio quality info
            try:
                import wave
                with wave.open(input_path, 'rb') as wf:
                    input_quality = {
                        'bit_depth': wf.getsampwidth() * 8,
                        'sample_rate': wf.getframerate(),
                        'channels': wf.getnchannels(),
                        'duration': wf.getnframes() / wf.getframerate()
                    }
            except Exception as e:
                print(f"Error reading input audio quality: {e}")
    
    output_path = os.path.join(audio_folder, "postprocessed_output_mmse_stsa.wav")
    if os.path.exists(output_path):
        output_size = round(os.path.getsize(output_path) / 1024, 2)
        # Get output audio quality info
        try:
            import wave
            with wave.open(output_path, 'rb') as wf:
                output_quality = {
                    'bit_depth': wf.getsampwidth() * 8,
                    'sample_rate': wf.getframerate(),
                    'channels': wf.getnchannels(),
                    'duration': wf.getnframes() / wf.getframerate()
                }
        except Exception as e:
            print(f"Error reading output audio quality: {e}")
    
    context = {
        'folder_name': folder_name,
        'input_audio': f"audio_results/{folder_name}/{input_filename}" if input_filename else None,
        'input_size': input_size,
        'output_size': output_size,
        'input_quality': input_quality,
        'output_quality': output_quality,
        'processing_method': 'WaveUNet + Voice-Preserving MMSE-STSA',
        'request': request  # Add request to context for get_full_path
    }
    
    return render(request, 'results.html', context) 
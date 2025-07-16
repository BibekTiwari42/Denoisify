# denoiser/views.py

from django.shortcuts import render
from .forms import AudioUploadForm
from .model_loader import denoise_audio
from django.conf import settings
import os

def index(request):
    context = {}
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            processing_method = request.POST.get('processing_method', 'waveunet_ssbse')
            
            input_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            output_path = os.path.join(settings.MEDIA_ROOT, "denoised_" + uploaded_file.name)

            # Save uploaded file
            with open(input_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Choose processing method
            use_ssbse_only = processing_method == 'ssbse_only'
            
            try:
                denoise_audio(input_path, output_path, use_ssbse_only=use_ssbse_only)
                
                context['input_audio'] = uploaded_file.name
                context['output_audio'] = "denoised_" + uploaded_file.name
                context['success'] = True
                
            except Exception as e:
                context['error'] = f"Error processing audio: {str(e)}"
                
    else:
        form = AudioUploadForm()

    context['form'] = form
    return render(request, 'index.html', context)

def audio_processor(request):
    """Dedicated page for audio processing with comparison"""
    context = {}
    
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            processing_method = request.POST.get('processing_method', 'waveunet_ssbse')
            
            # Create unique filenames to avoid conflicts
            import time
            timestamp = str(int(time.time()))
            input_filename = f"{timestamp}_{uploaded_file.name}"
            output_filename = f"{timestamp}_denoised_{uploaded_file.name}"
            
            input_path = os.path.join(settings.MEDIA_ROOT, input_filename)
            output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

            # Save uploaded file
            try:
                with open(input_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                # Choose processing method
                use_ssbse_only = processing_method == 'ssbse_only'
                
                # Process audio
                denoise_audio(input_path, output_path, use_ssbse_only=use_ssbse_only)
                
                # Get file sizes for comparison
                input_size = os.path.getsize(input_path)
                output_size = os.path.getsize(output_path)
                
                context.update({
                    'input_audio': input_filename,
                    'output_audio': output_filename,
                    'input_size': round(input_size / 1024, 2),  # KB
                    'output_size': round(output_size / 1024, 2),  # KB
                    'processing_method': 'WaveUNet + SSBSE' if not use_ssbse_only else 'SSBSE Only',
                    'success': True
                })
                
            except Exception as e:
                context['error'] = f"Error processing audio: {str(e)}"
                # Clean up uploaded file if processing failed
                if os.path.exists(input_path):
                    os.remove(input_path)
                
    else:
        form = AudioUploadForm()

    context['form'] = form
    return render(request, 'audio_processor.html', context)

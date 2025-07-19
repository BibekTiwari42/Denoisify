# denoiser/views.py

from django.shortcuts import render
from .forms import AudioUploadForm
from .model_loader import denoise_audio, plot_waveform, plot_spectrogram
from django.conf import settings
import os
import time
import shutil

def index(request):
    context = {}
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            
            # Create unique filenames with timestamp
            timestamp = str(int(time.time()))
            base_filename = os.path.splitext(uploaded_file.name)[0]
            input_filename = f"{timestamp}_{uploaded_file.name}"
            output_filename = f"{timestamp}_denoised_{uploaded_file.name}"
            postprocessed_filename = f"{timestamp}_postprocessed_{uploaded_file.name}"
            
            input_path = os.path.join(settings.MEDIA_ROOT, input_filename)
            output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
            postprocessed_path = os.path.join(settings.MEDIA_ROOT, postprocessed_filename)

            # Create organized folder structure
            audio_folder = os.path.join(settings.MEDIA_ROOT, "audio_results", base_filename)
            os.makedirs(audio_folder, exist_ok=True)

            # Save uploaded file
            with open(input_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            try:
                # Process audio with enhanced visualization
                denoise_audio(input_path, output_path)
                
                # Move generated files to organized folder
                visualization_files = [
                    'input_waveform.png', 'input_spectrogram.png',
                    'denoised_waveform.png', 'denoised_output_spectrogram.png',
                    'postprocessed_waveform.png', 'postprocessed_output_mmse_stsa_spectrogram.png',
                    'postprocessed_output_mmse_stsa.wav'
                ]
                
                for file in visualization_files:
                    if os.path.exists(file):
                        shutil.move(file, os.path.join(audio_folder, file))
                
                # Move audio files
                if os.path.exists(output_path):
                    shutil.move(output_path, os.path.join(audio_folder, "denoised_output.wav"))
                if os.path.exists('postprocessed_output_mmse_stsa.wav'):
                    shutil.move('postprocessed_output_mmse_stsa.wav', 
                              os.path.join(audio_folder, "postprocessed_output.wav"))
                
                # Get file sizes for comparison
                input_size = os.path.getsize(input_path)
                denoised_size = os.path.getsize(os.path.join(audio_folder, "denoised_output.wav"))
                postprocessed_size = os.path.getsize(os.path.join(audio_folder, "postprocessed_output.wav"))
                
                context.update({
                    'input_audio': input_filename,
                    'output_audio': "denoised_output.wav",
                    'postprocessed_audio': "postprocessed_output.wav",
                    'audio_folder': f"audio_results/{base_filename}",
                    'input_size': round(input_size / 1024, 2),  # KB
                    'denoised_size': round(denoised_size / 1024, 2),  # KB
                    'postprocessed_size': round(postprocessed_size / 1024, 2),  # KB
                    'processing_method': 'WaveUNet + MMSE-STSA',
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
    return render(request, 'index.html', context)

def audio_processor(request):
    """Dedicated page for audio processing with comprehensive visualization"""
    context = {}
    
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            
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
            try:
                with open(input_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                # Process audio with enhanced visualization
                denoise_audio(input_path, output_path)
                
                # Move generated files to organized folder
                visualization_files = [
                    'input_waveform.png', 'input_spectrogram.png',
                    'denoised_waveform.png', 'denoised_output_spectrogram.png',
                    'postprocessed_waveform.png', 'postprocessed_output_mmse_stsa_spectrogram.png',
                    'postprocessed_output_mmse_stsa.wav'
                ]
                
                for file in visualization_files:
                    if os.path.exists(file):
                        shutil.move(file, os.path.join(audio_folder, file))
                
                # Move audio files
                if os.path.exists(output_path):
                    shutil.move(output_path, os.path.join(audio_folder, "denoised_output.wav"))
                if os.path.exists('postprocessed_output_mmse_stsa.wav'):
                    shutil.move('postprocessed_output_mmse_stsa.wav', 
                              os.path.join(audio_folder, "postprocessed_output.wav"))
                
                # Get file sizes for comparison
                input_size = os.path.getsize(input_path)
                denoised_size = os.path.getsize(os.path.join(audio_folder, "denoised_output.wav"))
                postprocessed_size = os.path.getsize(os.path.join(audio_folder, "postprocessed_output.wav"))
                
                context.update({
                    'input_audio': input_filename,
                    'output_audio': "denoised_output.wav",
                    'postprocessed_audio': "postprocessed_output.wav",
                    'audio_folder': f"audio_results/{base_filename}",
                    'input_size': round(input_size / 1024, 2),  # KB
                    'denoised_size': round(denoised_size / 1024, 2),  # KB
                    'postprocessed_size': round(postprocessed_size / 1024, 2),  # KB
                    'processing_method': 'WaveUNet + MMSE-STSA',
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

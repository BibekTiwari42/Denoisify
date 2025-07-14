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
            input_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            output_path = os.path.join(settings.MEDIA_ROOT, "denoised_" + uploaded_file.name)

            with open(input_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            denoise_audio(input_path, output_path)

            context['input_audio'] = uploaded_file.name
            context['output_audio'] = "denoised_" + uploaded_file.name
    else:
        form = AudioUploadForm()

    context['form'] = form
    return render(request, 'index.html', context)

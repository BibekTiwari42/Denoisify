# denoiser/forms.py

from django import forms

class AudioUploadForm(forms.Form):
    file = forms.FileField(label="Upload a noisy WAV file")

from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
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
            return redirect('denoiser:login')
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('denoiser:home')
    
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                next_url = request.GET.get('next', 'denoiser:home')
                return redirect(next_url)
    else:
        form = CustomAuthenticationForm()
    
    return render(request, 'login.html', {'form': form})


@require_http_methods(["POST"])
def logout_confirm(request):
    """Handle logout confirmation via AJAX"""
    if request.user.is_authenticated:
        username = request.user.username
        logout(request)
        return JsonResponse({
            'success': True,
            'message': f'Goodbye {username}! You have been logged out successfully.',
            'redirect_url': '/'
        })
    return JsonResponse({
        'success': False,
        'message': 'You are not logged in.'
    })


@require_http_methods(["POST"])
def check_auth_status(request):
    """AJAX endpoint to check if user is authenticated"""
    return JsonResponse({
        'authenticated': request.user.is_authenticated,
        'username': request.user.username if request.user.is_authenticated else None
    })

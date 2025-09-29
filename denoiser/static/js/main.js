// Denoisify Main JavaScript

// Theme Management
class ThemeManager {
    constructor() {
        this.init();
    }

    init() {
        // Apply theme on page load
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
        }

        // Set up theme toggle if present
        this.setupThemeToggle();
    }

    setupThemeToggle() {
        const themeToggle = document.getElementById('themeToggle');
        const themeIcon = document.getElementById('themeIcon');

        if (!themeToggle) return;

        // Set initial icon based on current theme
        this.updateThemeIcon();

        // Theme toggle handler
        themeToggle.addEventListener('click', () => {
            const isCurrentlyLight = document.body.classList.contains('light-theme');

            if (isCurrentlyLight) {
                document.body.classList.remove('light-theme');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.classList.add('light-theme');
                localStorage.setItem('theme', 'light');
            }

            this.updateThemeIcon();
        });
    }

    updateThemeIcon() {
        const themeIcon = document.getElementById('themeIcon');
        if (!themeIcon) return;

        const isLight = document.body.classList.contains('light-theme');
        themeIcon.textContent = isLight ? '☀️' : '🌙';
    }
}

// User Dropdown Management
class DropdownManager {
    constructor() {
        this.setupUserDropdown();
    }

    setupUserDropdown() {
        const userMenuButton = document.getElementById('userMenuButton');
        const userDropdown = document.getElementById('userDropdown');

        if (!userMenuButton || !userDropdown) return;

        userMenuButton.onclick = (e) => {
            e.stopPropagation();
            userDropdown.classList.toggle('hidden');
        };

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!userMenuButton.contains(e.target) && !userDropdown.contains(e.target)) {
                userDropdown.classList.add('hidden');
            }
        });
    }
}

// Logout Management
class LogoutManager {
    constructor() {
        this.setupLogoutFunctions();
    }

    setupLogoutFunctions() {
        // Show logout confirmation
        window.showLogoutConfirmation = () => {
            const modal = document.getElementById('logoutModal');
            if (modal) {
                modal.classList.remove('hidden');
                document.body.style.overflow = 'hidden';
            }
        };

        // Close logout modal
        window.closeLogoutModal = () => {
            const modal = document.getElementById('logoutModal');
            if (modal) {
                modal.classList.add('hidden');
                document.body.style.overflow = 'auto';
            }
        };

        // Confirm logout
        window.confirmLogout = () => {
            window.closeLogoutModal();
            
            // Create a form and submit it for logout
            const form = document.createElement('form');
            form.method = 'POST';
            form.action = window.logoutUrl || '/logout/';
            
            const csrfToken = document.querySelector('meta[name=csrf-token]')?.content;
            if (csrfToken) {
                const csrfInput = document.createElement('input');
                csrfInput.type = 'hidden';
                csrfInput.name = 'csrfmiddlewaretoken';
                csrfInput.value = csrfToken;
                form.appendChild(csrfInput);
            }
            
            document.body.appendChild(form);
            form.submit();
        };

        // Show logout success message (for AJAX logouts)
        window.showLogoutSuccess = (message) => {
            window.closeLogoutModal();

            const successDiv = document.createElement('div');
            successDiv.className = 'fixed top-4 right-4 z-50 bg-green-600 text-white px-6 py-3 rounded-lg shadow-lg';
            successDiv.textContent = message;

            document.body.appendChild(successDiv);

            setTimeout(() => {
                if (successDiv.parentNode) {
                    successDiv.parentNode.removeChild(successDiv);
                }
                window.location.href = '/';
            }, 1500);
        };
    }
}

// Audio File Validation
class AudioValidator {
    constructor() {
        this.allowedExtensions = ['wav', 'mp3', 'flac', 'aac', 'ogg'];
        this.setupValidation();
    }

    setupValidation() {
        const fileInput = document.getElementById('id_file');
        const form = document.getElementById('uploadForm');
        const errorEl = document.getElementById('fileError');
        
        if (!fileInput || !form || !errorEl) return;

        fileInput.addEventListener('change', () => {
            const file = fileInput.files && fileInput.files[0];
            if (file) {
                const isValid = this.isValidFile(file.name);
                this.showError(!isValid);
            }
        });

        form.addEventListener('submit', (e) => {
            const file = fileInput.files && fileInput.files[0];
            if (!file || !this.isValidFile(file.name)) {
                e.preventDefault();
                this.showError(true);
            }
        });
    }

    isValidFile(fileName) {
        if (!fileName) return false;
        const parts = fileName.split('.');
        if (parts.length < 2) return false;
        const ext = parts.pop().toLowerCase();
        return this.allowedExtensions.includes(ext);
    }

    showError(show) {
        const errorEl = document.getElementById('fileError');
        if (!errorEl) return;

        if (show) {
            errorEl.classList.remove('hidden');
        } else {
            errorEl.classList.add('hidden');
        }
    }
}

// Delete Confirmation Management
class DeleteManager {
    constructor() {
        this.deleteUploadId = null;
        this.setupDeleteFunctions();
    }

    setupDeleteFunctions() {
        window.confirmDelete = (uploadId, fileName) => {
            this.deleteUploadId = uploadId;
            const fileNameEl = document.getElementById('fileName');
            const modal = document.getElementById('deleteModal');
            
            if (fileNameEl) fileNameEl.textContent = fileName;
            if (modal) {
                modal.classList.remove('hidden');
                modal.classList.add('flex');
            }
        };

        window.closeDeleteModal = () => {
            this.deleteUploadId = null;
            const modal = document.getElementById('deleteModal');
            if (modal) {
                modal.classList.add('hidden');
                modal.classList.remove('flex');
            }
        };

        window.deleteFile = () => {
            if (!this.deleteUploadId) return;

            const deleteUrl = `/delete-upload/${this.deleteUploadId}/`;
            
            fetch(deleteUrl, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': document.querySelector('meta[name=csrf-token]')?.content || '',
                    'Content-Type': 'application/json',
                },
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Remove the audio item from the page
                    const audioItem = document.querySelector(`[data-upload-id="${this.deleteUploadId}"]`);
                    if (audioItem) {
                        audioItem.remove();
                    }
                    this.showMessage('File deleted successfully', 'success');
                } else {
                    this.showMessage(data.error || 'Failed to delete file', 'error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                this.showMessage('An error occurred while deleting the file', 'error');
            })
            .finally(() => {
                window.closeDeleteModal();
            });
        };

        // Setup modal close events
        const deleteModal = document.getElementById('deleteModal');
        if (deleteModal) {
            deleteModal.addEventListener('click', (e) => {
                if (e.target === deleteModal) {
                    window.closeDeleteModal();
                }
            });
        }

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                window.closeDeleteModal();
                window.closeLogoutModal();
            }
        });
    }

    showMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `fixed top-4 right-4 z-50 px-6 py-3 rounded-lg shadow-lg ${
            type === 'success' ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
        }`;
        messageDiv.textContent = message;
        
        document.body.appendChild(messageDiv);
        
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.parentNode.removeChild(messageDiv);
            }
        }, 3000);
    }
}

// Audio Visualizer
class AudioVisualizer {
    constructor() {
        this.setupVisualizer();
    }

    setupVisualizer() {
        const audioVisualizer = document.getElementById('audioVisualizer');
        if (!audioVisualizer) return;

        const bars = [];
        for (let i = 0; i < 20; i++) {
            const bar = document.createElement('div');
            bar.className = 'audio-bar';
            bar.style.height = Math.random() * 40 + 10 + 'px';
            audioVisualizer.appendChild(bar);
            bars.push(bar);
        }

        // Animate bars
        setInterval(() => {
            bars.forEach(bar => {
                bar.style.height = Math.random() * 40 + 10 + 'px';
            });
        }, 200);
    }
}

// URL Parameter Management
class URLManager {
    constructor() {
        this.checkLogoutMessages();
    }

    checkLogoutMessages() {
        const urlParams = new URLSearchParams(window.location.search);
        const logoutSuccess = urlParams.get('logout_success');
        const logoutError = urlParams.get('logout_error');
        
        if (logoutSuccess) {
            if (typeof window.showLogoutSuccess === 'function') {
                window.showLogoutSuccess(logoutSuccess);
            }
            this.cleanURL();
        } else if (logoutError) {
            alert(logoutError);
            this.cleanURL();
        }
    }

    cleanURL() {
        window.history.replaceState({}, document.title, window.location.pathname);
    }
}

// CSRF Helper
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === name + '=') {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Auto-focus Management
class FocusManager {
    constructor() {
        this.setupAutoFocus();
    }

    setupAutoFocus() {
        document.addEventListener('DOMContentLoaded', () => {
            const firstInput = document.querySelector('input[type="text"], input[type="email"]');
            if (firstInput) {
                firstInput.focus();
            }
        });
    }
}

// Password Validation
class PasswordValidator {
    constructor() {
        this.setupPasswordValidation();
    }

    setupPasswordValidation() {
        const password1 = document.querySelector('input[name="password1"]');
        const password2 = document.querySelector('input[name="password2"]');

        if (!password1 || !password2) return;

        password2.addEventListener('input', function() {
            if (this.value && password1.value && this.value !== password1.value) {
                this.classList.add('border-red-500');
                this.classList.remove('border-gray-300');
            } else {
                this.classList.remove('border-red-500');
                this.classList.add('border-gray-300');
            }
        });
    }
}

// Initialize all components when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ThemeManager();
    new DropdownManager();
    new LogoutManager();
    new AudioValidator();
    new DeleteManager();
    new AudioVisualizer();
    new URLManager();
    new FocusManager();
    new PasswordValidator();
});
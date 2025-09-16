# Quick fix script for Django template syntax errors
Write-Host "🔧 Fixing Django template syntax errors..." -ForegroundColor Yellow

# Restore from Git if files are tracked
Write-Host "📂 Restoring templates from Git..." -ForegroundColor Cyan
git checkout denoiser/templates/profile.html
git checkout denoiser/templates/results.html

Write-Host "✅ Templates restored from Git!" -ForegroundColor Green
Write-Host "🌐 Try accessing your pages now - they should work!" -ForegroundColor Green
#!/bin/bash
# Quick fix script for Django template syntax errors

echo "🔧 Fixing Django template syntax errors..."

# Restore from Git if files are tracked
echo "📂 Restoring templates from Git..."
git checkout denoiser/templates/profile.html
git checkout denoiser/templates/results.html

echo "✅ Templates restored from Git!"
echo "🌐 Try accessing your pages now - they should work!"

# Alternative: Apply fixes directly
# echo "🔧 Applying template fixes directly..."
# sed -i 's/{% if user.first_name and user.last_name %} {{ user.first_name$/{% if user.first_name and user.last_name %}\n                  {{ user.first_name }} {{ user.last_name }}/g' denoiser/templates/profile.html
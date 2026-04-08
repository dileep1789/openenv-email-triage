# Deployment Script for OpenEnv Email Triage

# 1. Pushing to GitHub
Write-Host ">>> Pushing to GitHub..." -ForegroundColor Cyan
git add .
git commit -m "Update: Docker, HFSpace UI, and Token handling"
git push origin main

# 2. Pushing to Hugging Face Spaces (Optional)
# Instructions:
# git remote add hf https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE
# git push hf main

Write-Host ">>> Deployment Complete!" -ForegroundColor Green

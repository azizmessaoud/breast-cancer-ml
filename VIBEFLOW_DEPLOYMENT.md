# VibeFlow Deployment Guide

## Overview

This document provides instructions for deploying the Breast Cancer ML pipeline to **VibeFlow** (also known as VibeFlow cloud or similar Streamlit-compatible platforms).

## Prerequisites

- GitHub account with the `azizmessaoud/breast-cancer-ml` repository
- VibeFlow account (app.vibeflow.ai)
- Internet connection

## Deployment Steps

### 1. Navigate to VibeFlow

Go to [app.vibeflow.ai](https://app.vibeflow.ai) and sign in with your GitHub account.

### 2. Authorize GitHub Integration

Click the GitHub authorization link or "Connect Repository" button to authorize VibeFlow to access your GitHub repositories.

### 3. Select the Repository

- Choose repository: `azizmessaoud/breast-cancer-ml`
- Select branch: **`prototype`** (NOT main)
- Main module: **`streamlit_app.py`**

### 4. Configure Deployment Settings

- **Python version:** 3.11 (automatically detected from `runtime.txt`)
- **Requirements:** Automatically loaded from `requirements.txt`
- **Streamlit config:** Automatically loaded from `.streamlit/config.toml`

### 5. Deploy

Click "Deploy" and wait for the deployment to complete (~2-3 minutes).

## Important Notes

### Why the `prototype` Branch?

The `prototype` branch contains:
- ✅ Updated `requirements.txt` with Python 3.11+ compatible dependencies
- ✅ Removed TensorFlow (caused Python 3.13 incompatibility issues)
- ✅ `.streamlit/config.toml` for Streamlit-specific configuration
- ✅ `runtime.txt` specifying Python 3.11.9

### Dependencies Removed/Fixed

| Package | Issue | Solution |
|---------|-------|----------|
| tensorflow-cpu==2.10.1 | No wheels for Python 3.13 | Removed (not needed for inference) |
| numpy==1.24.4 | Build failure, needs distutils | Updated to numpy>=1.23.0,<2.0.0 |

### Environment Variables

No special environment variables required. All ML models are pre-trained and bundled in the `/models` directory.

## Troubleshooting

### Deployment Fails with "No solution found for dependencies"

**Cause:** Python version is set to 3.13+  
**Solution:** Ensure `runtime.txt` specifies `python-3.11.9`

### TensorFlow import errors

**Cause:** TensorFlow 2.10.1 listed in requirements  
**Solution:** Removed from `requirements.txt` in prototype branch (inference-only, not needed)

### Streamlit app not loading

**Check:**
1. Ensure `streamlit_app.py` exists in repository root
2. Verify `.streamlit/config.toml` is valid TOML syntax
3. Check VibeFlow logs for Python errors

## Post-Deployment

Once deployed, the app will be accessible at a URL like:
```
https://breast-cancer-ml-[random-id].vibeflow.ai/
```

The Streamlit app features:
- 📊 Model comparison results
- 📈 Performance metrics
- 🔍 About page with links to GitHub and Kaggle

## Updating the Deployment

1. Make changes locally on the `prototype` branch
2. Commit and push to GitHub
3. VibeFlow automatically redeploys (if auto-deploy is enabled) or click "Redeploy"

## Performance

- **Expected startup time:** 30-60 seconds
- **Model inference:** <100ms per prediction
- **Memory usage:** ~200-300 MB
- **Concurrent users:** VibeFlow's resource tier dependent

## Support

For issues:
1. Check VibeFlow documentation: https://vibeflow.ai/docs
2. Review repository issues: https://github.com/azizmessaoud/breast-cancer-ml/issues
3. Contact VibeFlow support

---

**Last Updated:** December 2025  
**Branch:** prototype  
**Status:** ✅ Ready for VibeFlow Deployment

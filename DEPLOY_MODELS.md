Runtime model download for Streamlit Cloud

Overview
- To keep Streamlit deploys fast, model artifacts (large .pth/.pkl files) are not stored in the repository.
- At runtime the app will attempt to download model files from a base URL you provide.

How to host your models
- Recommended: upload model files to a public stable host (S3, Google Cloud Storage, GitHub Releases or a static file host).
- Place them under a `models/saved_models/` path so the app can construct URLs like:
  `https://<BASE>/models/saved_models/cnn_model.pth`

Configure Streamlit Cloud
1. Go to your app in Streamlit Cloud → Settings → Secrets (or Secrets & variables)
2. Add a secret named `MODEL_BASE_URL` with the base URL where your `models/saved_models/` folder is hosted. Example:
   - `https://my-bucket.s3.amazonaws.com` or
   - `https://raw.githubusercontent.com/<user>/<repo>/<branch>` (if you host model files in GitHub Releases or raw files)

Notes
- The app will attempt to download the following files:
  - `cnn_model.pth`
  - `transformer_model.pth`
  - `vision_transformer_model.pth`
  - `svm_model.pkl`
  - `bayesian_model.pkl`
- If a model is not available or fails to download, the app logs a warning and continues; the UI will still work with available models or dummy fallbacks.

Local testing
- You can set `MODEL_BASE_URL` locally before starting Streamlit (PowerShell):

```powershell
$env:MODEL_BASE_URL = 'https://my-bucket.s3.amazonaws.com'
$env:PYTHONPATH = '.'; .\.venv311\Scripts\streamlit run app.py
```

Troubleshooting
- If download fails, check Streamlit logs for messages like `Failed to download <file>`.
- Ensure the files are publicly accessible (or provide signed URLs) and the path matches `models/saved_models/<filename>`.

If you want, I can help upload your model artifacts to GitHub Releases or prepare an S3 upload script.

# FastAPI + KaggleHub Model Deployment

This app uses FastAPI to serve a diabetic retinopathy model hosted on KaggleHub.

## How to Deploy on Azure

1. Push this repo to GitHub.
2. Create a Web App on Azure (Python).
3. Set the Startup Command:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
4. Add environment variables for KaggleHub access if needed.

## Endpoint

- `POST /DR` with an image file to get prediction.

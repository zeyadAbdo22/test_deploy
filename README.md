# Simple FastAPI App for Azure

## Steps to Deploy

1. Push this to GitHub.
2. Create Azure Web App (Python stack).
3. Set the startup command as:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
4. App will run at `https://<your-app-name>.azurewebsites.net/`

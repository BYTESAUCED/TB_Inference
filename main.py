import uvicorn
import os

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    # Run the app from app.server
    uvicorn.run("app.server:app", host=host, port=port, reload=False)

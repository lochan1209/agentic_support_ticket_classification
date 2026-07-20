import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
# Slowapi imports for lightning fast api rate limiting configuration
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# 1. Initialize the Rate Limiter (Tracks incoming client requests by their IP address)
limiter = Limiter(key_func=get_remote_address)

# 2. Instantiate fast api app
app = FastAPI(title="Scalable LLM API gateway")

# 3. Explicitly assign the global rate limit exception handler directly to the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 4. Define data validation schema model
class QueryRequest(BaseModel):
    prompt: str

# 5. LLM Production Endpoint secured with Rate Limiting (5 requests per minute)
@app.post("/v1/generate")
@limiter.limit("5/minute")
async def generate_text(request: Request, payload:QueryRequest):
    simulated_llm_response = f"Processed payload prompt: '{payload.prompt}' successfully."
    return {
        "status": "success",
        "model": "mock-llm-core",
        "output": simulated_llm_response
    }

# 6. Standard Health Route (Bypasses Rate Limiting for smooth load-balancer pings)
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", 
                host="127.0.0.1", 
                port=8001, reload=False, 
                reload_excludes=[".venv/*", "**/.*"])
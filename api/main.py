from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from scripts.gptq_inference import generate_response

app = FastAPI()

# Allow requests from your frontend (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    response = generate_response(prompt)
    return {"response": response}

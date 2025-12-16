"""
LinkedIn Lunatics API - Serves the MLA model for generating satirical posts
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import tiktoken

from model_mla import LuhGPT_MLA

app = FastAPI(title="LinkedIn Lunatics API", description="Generate satirical LinkedIn posts")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
device = "cpu"  # Cloud Run doesn't have GPU
tokenizer = tiktoken.get_encoding("gpt2")
model = None

@app.on_event("startup")
async def load_model():
    global model
    print(f"🚀 Loading model on {device}...")
    
    model = LuhGPT_MLA(
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_compressed=192,
        d_hidden=3072,
    ).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), "linkedin_mla_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("✅ Model loaded!")


def clean_generated_text(text: str) -> str:
    """Clean up generated text - always end with ..."""
    text = text.strip().rstrip('.!?')
    return text + "..."


class GenerateRequest(BaseModel):
    prompt: str = "I just fired"
    temperature: float = 0.3
    max_tokens: int = 100  # Hard cap at 100


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str


@app.get("/")
async def root():
    return {"message": "🔗 LinkedIn Lunatics API - POST to /generate"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a satirical LinkedIn post"""
    
    prompt_ids = torch.tensor([tokenizer.encode(request.prompt)]).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids, 
            max_new_tokens=request.max_tokens, 
            temperature=request.temperature
        )
    
    generated = tokenizer.decode(output_ids[0].tolist())
    generated = clean_generated_text(generated)
    
    return GenerateResponse(
        generated_text=generated,
        prompt=request.prompt
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

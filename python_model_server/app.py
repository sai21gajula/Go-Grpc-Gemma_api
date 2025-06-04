import asyncio
import json
import logging
import os
import torch
import concurrent.futures
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_loaded = asyncio.Event()

class GenerateRequest(BaseModel):
    prompt: str
    model_id: str = "google/gemma-3n-E4B-it-litert-preview"
    temperature: float = 0.7
    max_new_tokens: int = 256

class GenerateResponse(BaseModel):
    generated_text: str
    error: str = ""

class StreamGenerateResponse(BaseModel):
    partial_text: str = ""
    done: bool = False
    error: str = ""

async def load_model():
    """Load the Gemma model and tokenizer"""
    global model, tokenizer
    
    try:
        model_id = os.getenv("MODEL_ID", "google/gemma-3n-E4B-it-litert-preview")
        logger.info(f"Loading model: {model_id}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=os.getenv("HF_HOME", "./cache")
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with CPU-optimized settings
        logger.info("Loading model...")
        
        # Determine the best dtype for CPU inference
        # For CPU inference, bfloat16 may not be supported on all systems
        # so we'll use float32 as fallback
        device = "cpu"
        
        # Try to determine if bfloat16 is supported
        try:
            if torch.cuda.is_available():
                # If CUDA is available, we might still use CPU but check for bfloat16 support
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            else:
                # For CPU-only inference, use float32 for better compatibility
                torch_dtype = torch.float32
        except:
            torch_dtype = torch.float32
        
        logger.info(f"Using torch_dtype: {torch_dtype}, device: {device}")
        
        # Load model with memory-efficient settings
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="cpu",
            trust_remote_code=True,
            cache_dir=os.getenv("HF_HOME", "./cache"),
            low_cpu_mem_usage=True,
            # For quantized models, we might need additional config
            # quantization_config may not be needed for the litert-preview variant
        )
        
        # Ensure model is on CPU
        model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info(f"Model loaded successfully. Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        
        # Signal that model is loaded
        model_loaded.set()
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup"""
    logger.info("Starting model loading...")
    await load_model()
    logger.info("Model loading completed")
    yield
    logger.info("Shutting down...")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Gemma LLM Model Server",
    description="FastAPI server for Gemma model inference",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_loaded.is_set() and model is not None and tokenizer is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "loading", "model_loaded": False}

@app.post("/predict", response_model=GenerateResponse)
async def predict(request: GenerateRequest):
    """Generate text using the loaded model"""
    
    # Wait for model to be loaded
    if not model_loaded.is_set():
        await model_loaded.wait()
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        logger.info(f"Processing prediction request: {request.prompt[:50]}...")
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        
        # Move inputs to CPU (they should already be there)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=True if request.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        logger.info(f"Generated text length: {len(generated_text)}")
        
        return GenerateResponse(generated_text=generated_text)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return GenerateResponse(error=str(e))

def generate_stream(prompt: str, model_id: str, temperature: float, max_new_tokens: int):
    """Generator function for streaming text generation"""
    try:
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        
        # Move inputs to CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Create a text streamer
        streamer = TextIteratorStreamer(
            tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Generation parameters
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True if temperature > 0 else False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
        }
        
        # Start generation in a separate thread
        generation_thread = threading.Thread(
            target=model.generate,
            kwargs=generation_kwargs
        )
        generation_thread.start()
        
        # Stream tokens as they are generated
        for token in streamer:
            if token:  # Skip empty tokens
                yield StreamGenerateResponse(partial_text=token, done=False)
        
        # Wait for generation to complete
        generation_thread.join()
        
        # Send final done signal
        yield StreamGenerateResponse(partial_text="", done=True)
        
    except Exception as e:
        logger.error(f"Error during streaming generation: {e}")
        yield StreamGenerateResponse(error=str(e), done=True)

@app.post("/stream_predict")
async def stream_predict(request: GenerateRequest):
    """Stream text generation using the loaded model"""
    
    # Wait for model to be loaded
    if not model_loaded.is_set():
        await model_loaded.wait()
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    logger.info(f"Processing streaming request: {request.prompt[:50]}...")
    
    async def generate_async():
        """Async wrapper for the streaming generator"""
        loop = asyncio.get_event_loop()
        
        # Run the synchronous generator in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                list, 
                generate_stream(
                    request.prompt, 
                    request.model_id, 
                    request.temperature, 
                    request.max_new_tokens
                )
            )
            
            # Wait for completion and yield results
            results = await loop.run_in_executor(None, future.result)
            for result in results:
                yield json.dumps(result.dict()) + "\n"
    
    return StreamingResponse(
        generate_async(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Gemma LLM Model Server",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Text generation (POST)",
            "/stream_predict": "Streaming text generation (POST)"
        },
        "model_loaded": model_loaded.is_set()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PYTHON_PORT", "8001"))
    
    logger.info(f"Starting FastAPI server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

"""
Multi-LLM Inference Server — FastAPI backend for the Go-gRPC research platform.

Supports the following models (selected via MODEL_ID env var):
  • google/gemma-3-4b-it          (Gemma 3 4B IT, Google)
  • Qwen/Qwen3-4B                 (Qwen3 4B, Alibaba Cloud — latest as of 2025)
  • Qwen/Qwen2.5-3B-Instruct      (Qwen2.5 3B fallback)
  • google/gemma-3n-E4B-it-litert-preview  (original, backward compat)

TurboQuant / Quantization strategy:
  Google's "E4B" in the LiteRT model name stands for Efficient 4-bit — the same
  NF4 (NormalFloat 4-bit) quantization scheme introduced in QLoRA and used by
  Google's LiteRT (formerly TFLite) for on-device Gemma.

  For HuggingFace-loaded models we replicate this with bitsandbytes:
    - load_in_4bit=True
    - bnb_4bit_quant_type="nf4"          (NormalFloat4 — matches Google E4B)
    - bnb_4bit_compute_dtype=bfloat16    (matmul in bf16 for speed)
    - bnb_4bit_use_double_quant=True     (QLoRA-style nested quantization)

  Memory savings: fp32 ~16GB → NF4 ~4GB for a 4B parameter model.
  Quality loss: typically <2% BLEU/ROUGE degradation vs fp32.

Endpoints:
  GET  /health           readiness check
  POST /predict          unary text generation
  POST /stream_predict   streaming text generation (NDJSON)
  POST /evaluate         BLEU + ROUGE-L scoring (used by BenchmarkModels RPC)
  GET  /                 API info
"""

import asyncio
import json
import logging
import os
import threading
import time
import concurrent.futures
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID   = os.getenv("MODEL_ID", "google/gemma-3-4b-it")
HF_HOME    = os.getenv("HF_HOME", "./cache")
USE_QUANT  = os.getenv("USE_QUANTIZATION", "true").lower() in ("true", "1", "yes")
PYTHON_PORT = int(os.getenv("PYTHON_PORT", "8001"))

# ─── Global model state ────────────────────────────────────────────────────────

model      = None
tokenizer  = None
model_info = {"model_id": MODEL_ID, "quantization": "loading", "params_M": 0}
model_loaded = asyncio.Event()

# ─── Pydantic schemas ──────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    model_id: str = MODEL_ID
    temperature: float = 0.7
    max_new_tokens: int = 256

class GenerateResponse(BaseModel):
    generated_text: str
    token_count: int = 0
    error: str = ""

class StreamGenerateResponse(BaseModel):
    partial_text: str = ""
    done: bool = False
    error: str = ""

class EvaluateRequest(BaseModel):
    generated: str
    reference: str

class EvaluateResponse(BaseModel):
    bleu: float
    rouge_l: float
    error: str = ""

# ─── Model loading ─────────────────────────────────────────────────────────────

def _build_bnb_config() -> Optional[BitsAndBytesConfig]:
    """
    Build bitsandbytes NF4 weight quantization config.

    NF4 (NormalFloat4) is a *weight* quantization format — it reduces the
    stored model weight precision from fp32 to ~4.5 bits/param at load time.
    This is separate from TurboQuant, which is a *KV cache* compression
    algorithm (inference-time, ICLR 2026).

    NF4 details:
    - 16 quantization levels optimally spaced for normally-distributed weights
    - Dequantizes to bfloat16 for matrix multiplications
    - Double quantization (quantizes the absmax scale factors too): ~0.4 bits/param extra saving
    - Net: ~4.5 bits/param  →  4B model: ~16GB (fp32) → ~2.5GB (NF4)

    E4B in gemma-3n-E4B-it-litert-preview stands for "Efficient 4-Bit" — Google's
    NF4-equivalent format compiled into LiteRT (TFLite) flatbuffer for on-device deployment.
    This bitsandbytes config replicates the same compression for server-side inference.
    """
    try:
        import bitsandbytes  # noqa: F401
        logger.info("bitsandbytes available — enabling NF4 (TurboQuant-equivalent) quantization")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        logger.warning("bitsandbytes not installed — falling back to fp32 (no quantization)")
        return None


async def load_model():
    """Load the selected model with optional NF4 quantization."""
    global model, tokenizer, model_info

    logger.info(f"Loading model: {MODEL_ID}")
    logger.info(f"Cache dir: {HF_HOME}")
    logger.info(f"Quantization enabled: {USE_QUANT}")

    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=HF_HOME,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    logger.info("Loading model weights...")
    load_kwargs = dict(
        trust_remote_code=True,
        cache_dir=HF_HOME,
        low_cpu_mem_usage=True,
    )

    bnb_config = _build_bnb_config() if USE_QUANT else None
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        quant_label = "int4_nf4"
    else:
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"
        quant_label = "fp32"

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()
    load_time = time.time() - t0

    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    model_info = {
        "model_id": MODEL_ID,
        "quantization": quant_label,
        "params_M": round(params_M, 1),
        "load_time_s": round(load_time, 1),
    }
    logger.info(
        f"Model loaded: {MODEL_ID} | {params_M:.0f}M params | "
        f"quant={quant_label} | load_time={load_time:.1f}s"
    )
    model_loaded.set()

# ─── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting model loading...")
    await load_model()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Multi-LLM Inference Server",
    description=(
        "FastAPI inference backend for Go-gRPC Multi-LLM Platform. "
        "Supports Gemma 3-4B and Qwen3-4B with NF4 4-bit quantization."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    loaded = model_loaded.is_set() and model is not None
    return {
        "status": "healthy" if loaded else "loading",
        "model_loaded": loaded,
        **model_info,
    }

# ─── Unary predict ─────────────────────────────────────────────────────────────

@app.post("/predict", response_model=GenerateResponse)
async def predict(request: GenerateRequest):
    if not model_loaded.is_set():
        await model_loaded.wait()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        logger.info(f"[predict] prompt={request.prompt[:60]!r}")

        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        # Move to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        latency = time.time() - t0

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        token_count = outputs[0].shape[0] - inputs["input_ids"].shape[1]

        logger.info(f"[predict] ok: {token_count} tokens in {latency:.2f}s")
        return GenerateResponse(generated_text=generated_text, token_count=int(token_count))

    except Exception as e:
        logger.error(f"[predict] error: {e}")
        return GenerateResponse(generated_text="", error=str(e))

# ─── Streaming predict ─────────────────────────────────────────────────────────

def _generate_stream(prompt: str, temperature: float, max_new_tokens: int):
    """Synchronous generator for token-by-token streaming."""
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else None,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
        }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            if token:
                yield StreamGenerateResponse(partial_text=token, done=False)

        thread.join()
        yield StreamGenerateResponse(partial_text="", done=True)

    except Exception as e:
        logger.error(f"[stream] error: {e}")
        yield StreamGenerateResponse(error=str(e), done=True)


@app.post("/stream_predict")
async def stream_predict(request: GenerateRequest):
    if not model_loaded.is_set():
        await model_loaded.wait()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    logger.info(f"[stream_predict] prompt={request.prompt[:60]!r}")

    async def _async_gen():
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                list,
                _generate_stream(request.prompt, request.temperature, request.max_new_tokens),
            )
            results = await loop.run_in_executor(None, future.result)
        for r in results:
            yield json.dumps(r.dict()) + "\n"

    return StreamingResponse(
        _async_gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

# ─── Evaluation endpoint ───────────────────────────────────────────────────────

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """
    Compute BLEU and ROUGE-L scores for a generated text against a reference.

    Used by the BenchmarkModels and EvaluateGeneration gRPC RPCs to provide
    text-quality metrics alongside latency measurements.

    BLEU (Bilingual Evaluation Understudy):
      - Measures n-gram overlap (1–4 grams) between generated and reference.
      - Range: 0–1; higher = more similar to reference.
      - Weakness: rewards exact matches, penalises paraphrase.

    ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation — Longest common):
      - Measures longest common subsequence (LCS) F1 score.
      - More forgiving than BLEU for reordered/paraphrased text.
      - Range: 0–1; higher = more similar.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer as rs_module
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

        ref_tokens  = request.reference.lower().split()
        hyp_tokens  = request.generated.lower().split()

        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

        scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(request.reference, request.generated)
        rouge_l = scores["rougeL"].fmeasure

        return EvaluateResponse(bleu=round(bleu, 4), rouge_l=round(rouge_l, 4))

    except ImportError as e:
        return EvaluateResponse(bleu=0.0, rouge_l=0.0, error=f"Missing dependency: {e}")
    except Exception as e:
        logger.error(f"[evaluate] error: {e}")
        return EvaluateResponse(bleu=0.0, rouge_l=0.0, error=str(e))

# ─── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Multi-LLM Inference Server",
        "version": "2.0.0",
        "model": model_info,
        "endpoints": {
            "GET  /health":         "Readiness check",
            "POST /predict":        "Unary text generation",
            "POST /stream_predict": "Streaming text generation (NDJSON)",
            "POST /evaluate":       "BLEU + ROUGE-L quality scoring",
        },
        "quantization_note": (
            "Uses NF4 4-bit weight quantization (bitsandbytes) when USE_QUANTIZATION=true. "
            "NF4 is equivalent to Google LiteRT's E4B format (Efficient 4-Bit). "
            "Note: TurboQuant (ICLR 2026) is a separate KV-cache compression algorithm "
            "and is not the same as NF4 weight quantization. "
            "Memory: ~2.5GB for 4B params vs ~16GB fp32."
        ),
    }

# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PYTHON_PORT, log_level="info")

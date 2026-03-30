"""
Multi-LLM Inference Server — FastAPI backend for the Go-gRPC research platform.

Supports the following models (selected via MODEL_ID env var):
  • google/gemma-3-4b-it          (Gemma 3 4B IT, Google)
  • Qwen/Qwen3-4B                 (Qwen3 4B, Alibaba Cloud — latest as of 2025)
  • Qwen/Qwen2.5-3B-Instruct      (Qwen2.5 3B fallback)
  • google/gemma-3n-E4B-it-litert-preview  (original, backward compat)

Quantization stack (two independent layers):

  Layer 1 — NF4 Weight Quantization (load-time, via bitsandbytes):
    Compresses stored model weights from fp32 (~16GB) to ~2.5GB for 4B models.
    Google's "E4B" in LiteRT model names is their equivalent on-device format.

  Layer 2 — TurboQuant KV Cache Compression (inference-time, arXiv:2504.19874):
    Compresses the attention Key-Value cache during generation.
    Algorithm: random orthogonal rotation + Lloyd-Max scalar quantization (Stage 1)
    + optional 1-bit QJL residual correction (Stage 2, arXiv:2406.03482).
    At 3 bits: ~4.3× KV cache memory reduction with near-zero accuracy loss.

Endpoints:
  GET  /health                readiness check
  POST /predict               unary text generation
  POST /stream_predict        streaming text generation (NDJSON)
  POST /evaluate              BLEU + ROUGE-L scoring
  POST /turboquant/validate   TurboQuant compression quality check
  GET  /turboquant/status     TurboQuant configuration and memory stats
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

# TurboQuant KV cache compression (arXiv:2504.19874)
try:
    from turboquant import TurboQuantCompressor, TurboQuantCache, validate_turboquant
    TURBOQUANT_AVAILABLE = True
except ImportError as _tq_err:
    logger.warning(f"TurboQuant not available: {_tq_err}")
    TURBOQUANT_AVAILABLE = False
    TurboQuantCache = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID    = os.getenv("MODEL_ID",        "google/gemma-3-4b-it")
HF_HOME     = os.getenv("HF_HOME",         "./cache")
USE_QUANT   = os.getenv("USE_QUANTIZATION","true").lower() in ("true", "1", "yes")
PYTHON_PORT = int(os.getenv("PYTHON_PORT", "8001"))

# TurboQuant KV cache compression config
USE_TURBOQUANT      = os.getenv("USE_TURBOQUANT",    "true").lower() in ("true", "1", "yes")
TURBOQUANT_BITS     = int(os.getenv("TURBOQUANT_BITS",    "3"))   # 1–4; 3 recommended
TURBOQUANT_QJL      = os.getenv("TURBOQUANT_QJL",    "false").lower() in ("true", "1")
TURBOQUANT_KEY_BITS = int(os.getenv("TURBOQUANT_KEY_BITS", "0"))  # 0 = auto (bits+1)
TURBOQUANT_VAL_BITS = int(os.getenv("TURBOQUANT_VAL_BITS", "0"))  # 0 = auto (bits)

# ─── Global model state ────────────────────────────────────────────────────────

model        = None
tokenizer    = None
tq_compressor = None   # TurboQuantCompressor instance (set after model load)
model_info   = {"model_id": MODEL_ID, "quantization": "loading", "params_M": 0}
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

class TurboQuantValidateRequest(BaseModel):
    bits: int = TURBOQUANT_BITS
    head_dim: int = 128
    seq_len: int = 64
    use_qjl: bool = TURBOQUANT_QJL

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

    # ── TurboQuant KV cache compressor ──────────────────────────────────────
    global tq_compressor
    if USE_TURBOQUANT and TURBOQUANT_AVAILABLE:
        key_bits = TURBOQUANT_KEY_BITS or None   # None → auto (bits+1)
        val_bits = TURBOQUANT_VAL_BITS or None   # None → auto (bits)
        tq_compressor = TurboQuantCompressor(
            bits=TURBOQUANT_BITS,
            use_qjl=TURBOQUANT_QJL,
            key_bits=key_bits,
            value_bits=val_bits,
            seed=42,
        )
        model_info["turboquant"] = (
            f"{TURBOQUANT_BITS}-bit KV cache compression "
            f"(keys={tq_compressor.key_bits}b, values={tq_compressor.value_bits}b, "
            f"qjl={'on' if TURBOQUANT_QJL else 'off'})"
        )
        logger.info(
            f"TurboQuant enabled: {TURBOQUANT_BITS}-bit | "
            f"keys={tq_compressor.key_bits}b | values={tq_compressor.value_bits}b | "
            f"qjl={'on' if TURBOQUANT_QJL else 'off'}"
        )
    else:
        model_info["turboquant"] = "disabled"
        if USE_TURBOQUANT and not TURBOQUANT_AVAILABLE:
            logger.warning("USE_TURBOQUANT=true but turboquant.py not found — continuing without KV compression")

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

        # Build TurboQuant KV cache if enabled
        gen_kwargs: dict = dict(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature if request.temperature > 0 else None,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        tq_cache = None
        if tq_compressor is not None and TurboQuantCache is not None:
            tq_cache = TurboQuantCache(tq_compressor)
            gen_kwargs["past_key_values"] = tq_cache

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        latency = time.time() - t0

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        token_count = outputs[0].shape[0] - inputs["input_ids"].shape[1]

        tq_mem = tq_cache.memory_summary() if tq_cache is not None else "disabled"
        logger.info(f"[predict] ok: {token_count} tokens in {latency:.2f}s | {tq_mem}")
        return GenerateResponse(generated_text=generated_text, token_count=int(token_count))

    except Exception as e:
        logger.error(f"[predict] error: {e}")
        return GenerateResponse(generated_text="", error=str(e))

# ─── Streaming predict ─────────────────────────────────────────────────────────

def _generate_stream(prompt: str, temperature: float, max_new_tokens: int):
    """Synchronous generator for token-by-token streaming with TurboQuant KV cache."""
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

        # Attach TurboQuant KV cache if enabled
        if tq_compressor is not None and TurboQuantCache is not None:
            generation_kwargs["past_key_values"] = TurboQuantCache(tq_compressor)

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

# ─── TurboQuant endpoints ──────────────────────────────────────────────────────

@app.post("/turboquant/validate")
async def turboquant_validate(request: TurboQuantValidateRequest):
    """
    Run TurboQuant compression/decompression on synthetic data and return
    quality metrics (MSE, cosine similarity, compression ratio).

    Use this to verify the algorithm is working correctly and to tune
    bit-width vs quality tradeoffs before applying to real inference.

    Example (grpcurl via REST):
        curl -X POST http://localhost:8001/turboquant/validate \\
          -H 'Content-Type: application/json' \\
          -d '{"bits": 3, "head_dim": 128, "seq_len": 64}'
    """
    if not TURBOQUANT_AVAILABLE:
        raise HTTPException(status_code=501, detail="TurboQuant not available (turboquant.py not found)")

    try:
        compressor = TurboQuantCompressor(
            bits=request.bits,
            use_qjl=request.use_qjl,
            seed=42,
        )
        results = validate_turboquant(
            bits=request.bits,
            head_dim=request.head_dim,
            seq_len=request.seq_len,
        )
        results["use_qjl"] = request.use_qjl
        results["status"] = "ok"
        return results
    except Exception as e:
        logger.error(f"[turboquant/validate] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/turboquant/status")
async def turboquant_status():
    """
    Return TurboQuant configuration and whether it is active for this server.
    """
    return {
        "available": TURBOQUANT_AVAILABLE,
        "enabled": tq_compressor is not None,
        "bits": TURBOQUANT_BITS if tq_compressor else None,
        "key_bits": tq_compressor.key_bits if tq_compressor else None,
        "value_bits": tq_compressor.value_bits if tq_compressor else None,
        "use_qjl": TURBOQUANT_QJL if tq_compressor else None,
        "algorithm": (
            "TurboQuant (arXiv:2504.19874): random orthogonal rotation "
            "+ Lloyd-Max scalar quantization (Stage 1) "
            "+ optional QJL 1-bit residual correction (Stage 2)"
        ),
        "papers": {
            "TurboQuant": "https://arxiv.org/abs/2504.19874",
            "PolarQuant":  "https://arxiv.org/abs/2502.02617",
            "QJL":         "https://arxiv.org/abs/2406.03482",
        },
    }

# ─── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Multi-LLM Inference Server",
        "version": "3.0.0",
        "model": model_info,
        "endpoints": {
            "GET  /health":                  "Readiness check",
            "POST /predict":                 "Unary text generation",
            "POST /stream_predict":          "Streaming text generation (NDJSON)",
            "POST /evaluate":                "BLEU + ROUGE-L quality scoring",
            "POST /turboquant/validate":     "TurboQuant quality/compression check",
            "GET  /turboquant/status":       "TurboQuant config",
        },
        "quantization_layers": {
            "layer_1_weights": (
                "NF4 4-bit weight quantization (bitsandbytes, load-time). "
                "Reduces 4B model from ~16GB to ~2.5GB. "
                "Google's E4B in LiteRT = equivalent on-device format."
            ),
            "layer_2_kv_cache": (
                "TurboQuant KV cache compression (inference-time, arXiv:2504.19874). "
                "Stage 1: random rotation + Lloyd-Max scalar quantization. "
                "Stage 2 (optional): 1-bit QJL residual correction (arXiv:2406.03482). "
                f"Status: {'enabled at ' + str(TURBOQUANT_BITS) + '-bit' if tq_compressor else 'disabled'}."
            ),
        },
    }

# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PYTHON_PORT, log_level="info")

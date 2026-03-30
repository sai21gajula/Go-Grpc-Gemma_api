# Experiments Guide

This guide shows you how to run each experiment, what to expect, and how to interpret results.

---

## Prerequisites

```bash
# Install dependencies
./dev.sh deps

# For Qwen experiments: install Ollama
curl -fsSL https://ollama.com/install.sh | sh
./dev.sh ollama-pull   # downloads qwen3:4b (~2.3GB)
```

---

## Experiment 1: Basic Gemma 3-4B Inference

**What it tests:** End-to-end gRPC call through Go gateway → Python FastAPI → Gemma 3-4B (NF4 quantized)

```bash
# Terminal 1: start Python inference server
MODEL_ID=google/gemma-3-4b-it USE_QUANTIZATION=true ./dev.sh python

# Terminal 2: start Go gRPC gateway
./dev.sh go

# Terminal 3: test
./dev.sh test-gemma
```

**Expected output:**
```json
{
  "generatedText": "Machine learning is a subset of artificial intelligence..."
}
```

**Key observations:**
- First call is slow (model warmup): 2–5s
- Subsequent calls: 300ms–2s depending on prompt length
- Memory usage: ~2.5GB for NF4 vs ~16GB for fp32

---

## Experiment 2: Qwen3-4B via Ollama

**What it tests:** gRPC → Ollama → Qwen3-4B (Q4_K_M GGUF quantization)

```bash
# Terminal 1: start Ollama
ollama serve

# Terminal 2: start Go gRPC gateway
./dev.sh go

# Terminal 3: test
./dev.sh test-qwen
```

**Expected output:**
```json
{
  "generatedText": "Go's goroutine model provides true concurrency..."
}
```

**Qwen3 Thinking Mode** (extra feature):
```bash
grpcurl -plaintext -d '{
  "prompt": "/think What is the time complexity of quicksort?",
  "model_id": "qwen:qwen3:4b",
  "temperature": 0.6,
  "max_new_tokens": 200
}' localhost:7860 llm_service.LLMService/GenerateText
```
With `/think`, Qwen3 shows its reasoning before giving the answer.

---

## Experiment 3: Streaming Comparison

**What it tests:** Token-by-token streaming, TTFT (time-to-first-token)

```bash
./dev.sh test-stream
```

Watch tokens arrive one by one.  Compare TTFT between Gemma and Qwen:
```bash
# Gemma streaming
grpcurl -plaintext -d '{"prompt":"Tell me about AI.","model_id":"gemma:google/gemma-3-4b-it","temperature":0.7,"max_new_tokens":100}' \
  localhost:7860 llm_service.LLMService/StreamGenerateText

# Qwen streaming
grpcurl -plaintext -d '{"prompt":"Tell me about AI.","model_id":"qwen:qwen3:4b","temperature":0.7,"max_new_tokens":100}' \
  localhost:7860 llm_service.LLMService/StreamGenerateText
```

---

## Experiment 4: EvaluateGeneration — Quality Metrics

**What it tests:** BLEU + ROUGE-L scores + latency for multiple models on the same prompt.

```bash
./dev.sh test-evaluate
```

Or with more models and a known-answer question:
```bash
grpcurl -plaintext -d '{
  "prompt": "What is the capital of France?",
  "reference_answer": "The capital of France is Paris.",
  "model_ids": [
    "gemma:google/gemma-3-4b-it",
    "qwen:qwen3:4b"
  ],
  "temperature": 0.0,
  "max_new_tokens": 30
}' localhost:7860 llm_service.LLMService/EvaluateGeneration
```

**Interpreting results:**
```json
{
  "results": [
    {
      "modelId": "gemma:google/gemma-3-4b-it",
      "generatedText": "Paris is the capital of France.",
      "latencyMs": 340.5,
      "tokensPerSec": 12.4,
      "bleuScore": 0.62,
      "rougeL": 0.71,
      "backendType": "gemma_hf",
      "quantization": "int4_nf4"
    }
  ]
}
```

- **BLEU > 0.5**: good lexical overlap with reference
- **ROUGE-L > 0.6**: good structural similarity
- **Latency**: wall-clock time in ms for the full response

---

## Experiment 5: gRPC vs REST Benchmark

**What it tests:** Whether gRPC outperforms direct REST calls to Python.  This is the core research comparison.

```bash
./dev.sh benchmark
```

This runs `BenchmarkModels` with `include_rest_comparison: true`, which:
1. Sends each prompt via gRPC → Go → Python
2. Sends the same prompt via direct REST POST to Python (bypassing Go)
3. Reports latency delta: `grpc_vs_rest_delta_ms`

**Interpreting the summary table:**
```
┌─────────────────────────┬──────────┬──────────┬──────────┬──────────────┐
│ Model                   │ Avg(ms)  │ P95(ms)  │ Tok/s    │ Quantization │
├─────────────────────────┼──────────┼──────────┼──────────┼──────────────┤
│ gemma:google/gemma...   │    420.3 │    680.1 │     11.2 │ int4_nf4     │
└─────────────────────────┴──────────┴──────────┴──────────┴──────────────┘
  gRPC vs REST (gemma:...): +3.2ms overhead
```

**Expected finding**: gRPC adds ~1–5ms overhead for single requests (negligible).  Run the concurrent benchmark to see gRPC's advantage:

```bash
# Run 10 concurrent gRPC requests
for i in $(seq 1 10); do
  grpcurl -plaintext -d '{"prompt":"What is AI?","model_id":"gemma:google/gemma-3-4b-it","temperature":0.0,"max_new_tokens":50}' \
    localhost:7860 llm_service.LLMService/GenerateText &
done
wait
```

vs 10 concurrent REST requests:
```bash
for i in $(seq 1 10); do
  curl -s -X POST http://localhost:8001/predict \
    -H "Content-Type: application/json" \
    -d '{"prompt":"What is AI?","temperature":0.0,"max_new_tokens":50}' &
done
wait
```

The gRPC calls complete faster because Go's goroutines don't block while waiting for Python, and HTTP/2 multiplexing reduces connection overhead.

---

## Experiment 6: Quantization Impact on Quality

**What it tests:** Does NF4 quantization hurt response quality vs full precision?

```bash
# Start Python server WITHOUT quantization
USE_QUANTIZATION=false ./dev.sh python

# Run evaluation
grpcurl -plaintext -d '{
  "prompt": "Summarise: The quick brown fox jumps over the lazy dog.",
  "reference_answer": "A fox jumps over a dog.",
  "model_ids": ["gemma:google/gemma-3-4b-it"],
  "temperature": 0.0,
  "max_new_tokens": 20
}' localhost:7860 llm_service.LLMService/EvaluateGeneration
```

Then restart with quantization and compare:
```bash
USE_QUANTIZATION=true ./dev.sh python
# run same evaluation...
```

**Expected finding**: BLEU/ROUGE difference < 2%, but memory usage drops from ~16GB to ~2.5GB.

---

## Experiment 7: Gemini API vs Local Models

**What it tests:** Cloud Gemini 2.0 Flash vs local Gemma/Qwen quality and latency.

```bash
export GEMINI_API_KEY=your_key_here
./dev.sh go   # restart gateway with Gemini enabled

grpcurl -plaintext -d '{
  "prompt": "What is the Pythagorean theorem?",
  "reference_answer": "In a right triangle, a² + b² = c².",
  "model_ids": [
    "gemma:google/gemma-3-4b-it",
    "qwen:qwen3:4b",
    "gemini:gemini-2.0-flash"
  ],
  "temperature": 0.0,
  "max_new_tokens": 50
}' localhost:7860 llm_service.LLMService/EvaluateGeneration
```

**Expected finding**: Gemini has higher BLEU/ROUGE but adds 100–500ms network latency. Local models are ~$0/request; Gemini costs ~$0.000015/request at this token count.

---

## Reading JSON Reports

After any benchmark, reports are saved to `benchmarks/results_<timestamp>.json`:

```bash
# View latest report
ls -t benchmarks/*.json | head -1 | xargs cat | python3 -m json.tool

# Extract summary table
ls -t benchmarks/*.json | head -1 | xargs python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
print(r['summary'])
" --

# Compare latencies across models
ls -t benchmarks/*.json | head -1 | xargs python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
for a in r['aggregated']:
    print(f\"{a['model_id']:40s} avg={a['avg_latency_ms']:7.1f}ms p95={a['p95_latency_ms']:7.1f}ms tok/s={a['avg_tokens_per_sec']:5.1f}\")
" --
```

---

## Adding a New Model Backend

1. Implement `backends.LLMBackend` interface in `go_grpc_server/backends/yourmodel.go`
2. Register in `main.go` `NewLLMServer()`:
   ```go
   bmap["yourprefix"] = backends.NewYourBackend(...)
   ```
3. Use `model_id: "yourprefix:model-name"` in gRPC requests
4. No proto changes needed — routing is dynamic

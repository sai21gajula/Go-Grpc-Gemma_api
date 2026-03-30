# Research: TurboQuant, NF4 Weight Quantization, gRPC vs REST, and Go for LLM Serving

This document explains the core technical concepts behind the platform, with references to the empirical results you can reproduce using `./dev.sh benchmark`.

---

## 1. Quantization: TurboQuant vs NF4

**Important distinction:** TurboQuant and NF4 are two completely different algorithms targeting different parts of the LLM inference pipeline. This was incorrectly conflated in an earlier version of this document.

```
┌─────────────────────────────────────────────────────────┐
│  LLM Inference Memory Breakdown                         │
│                                                         │
│  ┌──────────────────┐   ◄── NF4 / GPTQ / AWQ           │
│  │  Model Weights   │       (load-time, stored on disk) │
│  │  ~2.5GB (NF4)    │                                   │
│  └──────────────────┘                                   │
│                                                         │
│  ┌──────────────────┐   ◄── TurboQuant                  │
│  │   KV Cache       │       (inference-time, grows with │
│  │  (grows per req) │        context length)            │
│  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

---

### 1a. TurboQuant (Google Research, ICLR 2026)

TurboQuant is a **Key-Value (KV) cache and vector search compression algorithm** — it has nothing to do with model weights. It was presented at ICLR 2026 and targets the memory cost of the attention mechanism's KV cache, which grows linearly with context length and is a primary bottleneck for long-context inference.

#### The problem it solves

During transformer inference, each attention layer stores Keys and Values for every token in the context (the KV cache). For a long document or chat history, this dominates memory:
```
KV cache size ≈ 2 × n_layers × n_heads × d_head × context_length × bytes_per_value
```
For a 7B model with 4096-token context in fp16: ~2GB just for the KV cache.

#### How TurboQuant works (two-stage pipeline)

**Stage 1 — PolarQuant:**
- Takes the KV vector in Cartesian space (a standard float vector)
- Converts to **polar coordinates**: splits each vector into a radius (magnitude) and angles
- The radius is quantized separately (it carries most of the information)
- The angles are quantized coarsely — this is where compression happens
- Achieves ~3–4 bits per value at this stage

**Stage 2 — QJL (Quantized Johnson-Lindenstrauss):**
- A 1-bit "error correction" step using the Johnson-Lindenstrauss lemma
- Applies a random projection to detect and remove the **systematic bias** introduced by PolarQuant's angle quantization
- Cost: only 1 extra bit per value, but eliminates the accuracy loss from stage 1

The combination achieves **3-bit compression with mathematically zero accuracy loss** — the 1-bit QJL correction fully compensates for PolarQuant's bias.

#### Results (ICLR 2026)

| Metric | Value |
|--------|-------|
| Target | KV cache memory |
| Compression | Down to **3 bits/value** |
| Accuracy loss | **Zero** (QJL correction removes bias) |
| KV cache memory reduction | **6×** |
| Throughput on NVIDIA H100 | **Up to 8×** improvement |
| Also applicable to | Vector similarity search |

#### Relationship to this project

TurboQuant is not yet implemented in standard HuggingFace `transformers`. It is a research contribution targeting large GPU deployments (H100) with long-context workloads. For our CPU-based 4B model inference:
- TurboQuant's benefit (6× KV cache reduction) would be relevant when handling long prompts or many concurrent sessions
- Future integration path: implement TurboQuant as a custom `DynamicCache` subclass in the Python server when HuggingFace adds support

---

### 1b. NF4 Weight Quantization (what this project actually uses)

This project uses **NF4 (NormalFloat4)** for model weight compression — a completely separate technique from TurboQuant. NF4 is applied once at model load time and permanently reduces the stored weight precision.

#### How NF4 works

Neural network weights are approximately normally distributed (Gaussian). NF4 exploits this by defining 16 quantization levels **optimally spaced for a standard normal distribution** — more levels in the dense centre, fewer at the sparse tails.

```
fp32 weight → normalize by blockwise absmax → NF4 4-bit code (0–15)
NF4 4-bit code → dequantize to bfloat16 → matrix multiply
```

**Double quantization** (used in QLoRA and replicated here):
- The blockwise absmax values are themselves quantized to 8-bit, saving ~0.4 more bits/parameter
- Net storage: ~4.5 bits/parameter vs 32 bits for fp32

#### Connection to Gemma's E4B format

The original model `gemma-3n-E4B-it-litert-preview` uses Google's LiteRT (formerly TFLite) on-device format:
- **E4B** = Efficient 4-Bit (Google's internal name for their NF4-equivalent format)
- **litert** = LiteRT runtime (Google's on-device ML runtime)

This is Google's production NF4 implementation for edge devices — functionally equivalent to `bitsandbytes` NF4 but compiled into a TFLite flatbuffer for Android/iOS deployment. Our Python server replicates the same compression for server-side HuggingFace inference.

#### Memory savings for 4B parameter models

| Precision | Bits/param | 4B model RAM |
|-----------|-----------|-------------|
| fp32      | 32        | ~16 GB       |
| fp16      | 16        | ~8 GB        |
| int8      | 8         | ~4 GB        |
| **NF4** (this project) | ~4.5 | **~2.5 GB** |

#### Quality tradeoff

In practice, NF4 4-bit loses < 2% on BLEU/ROUGE benchmarks vs fp32 because:
1. NF4 levels are optimally placed for normal distributions (unlike INT4 uniform which loses ~5–8%)
2. bfloat16 compute preserves fidelity during the dequantize-then-multiply step
3. Double quantization reduces second-order quantization noise

#### Comparison with other weight quantization methods

| Method | Target | Key idea | Quality vs NF4 | Speed |
|--------|--------|----------|----------------|-------|
| **NF4** (this project) | Weights | Optimal levels for normal dist | Baseline | Fast |
| **GPTQ** | Weights | Post-training error minimisation | +1–3% better | Slower load |
| **AWQ** | Weights | Protects salient (high-activation) weights | +1–2% better | Fast |
| **INT4** (uniform) | Weights | Equal-width bins | 5–8% worse | Fastest |
| **TurboQuant** | KV cache | PolarQuant + QJL | N/A (different target) | 8× faster |

**Summary:** NF4 and TurboQuant are **complementary** — you can apply NF4 to compress stored weights AND TurboQuant to compress the KV cache at inference time. Applied together on a large model, they address the two main memory bottlenecks independently.

---

## 2. gRPC vs REST for LLM Serving

### Protocol comparison

| Dimension | gRPC | REST/JSON |
|-----------|------|-----------|
| **Transport** | HTTP/2 | HTTP/1.1 |
| **Encoding** | Protocol Buffers (binary) | JSON (text) |
| **Multiplexing** | Multiple streams over 1 TCP connection | 1 request per connection (HTTP/1.1) |
| **Streaming** | Bidirectional (native) | Server-Sent Events / Chunked |
| **Schema** | Proto file (strict, versioned) | OpenAPI (optional) |
| **Client codegen** | Automatic from .proto | OpenAPI generator / manual |
| **Debugging** | grpcurl, Postman | curl, Postman |

### Payload size: Protobuf vs JSON

For a typical `GenerateRequest` (50-word prompt, temperature, max_tokens):
```
JSON  (text):  {"prompt":"...","model_id":"...","temperature":0.7,"max_new_tokens":256}
               ~120 bytes

Protobuf (binary): field 1 (string 50 words), field 2 (string), field 3 (float), field 4 (int)
               ~60 bytes  (≈2× smaller)
```

For a `GenerateResponse` (500-word generated text):
```
JSON:     ~2,800 bytes
Protobuf: ~1,400 bytes  (≈2× smaller)
```

### Latency breakdown (approximate)

```
REST/JSON path:
  Client → TCP connect (new) → HTTP/1.1 headers (~0.5KB) → JSON body → server
  ≈ 5–15ms overhead per request (connection establishment dominates for short lived)

gRPC path:
  Client → (reuse TCP + HTTP/2 stream) → Protobuf frame → server
  ≈ 1–5ms overhead per request (multiplexed streams amortise TCP setup)
```

**Net gRPC advantage at low concurrency**: ~2–10ms per request (often negligible vs 500ms LLM inference).

**Net gRPC advantage under high concurrency (10+ parallel requests)**:
- HTTP/1.1: each client needs its own TCP connection (connection pool pressure)
- HTTP/2: all clients share one TCP connection via multiplexed streams
- Result: 15–40% higher throughput with gRPC under load

### Streaming comparison

```
REST SSE:
  Client polls or holds open connection
  Each token: text/event-stream line (text encoding overhead)

gRPC server-side streaming:
  Single stream opened once, tokens sent as length-prefixed Protobuf frames
  No HTTP header per token
  Result: 10–20% lower TTFT (time-to-first-token) and 5–15% lower per-token latency
```

### When REST is better

- Simple integrations where gRPC client code is unavailable (browsers without gRPC-Web proxy)
- Quick debugging / prototyping (curl is simpler than grpcurl)
- Teams unfamiliar with Protobuf schema management

### Benchmark results (run `./dev.sh benchmark` to reproduce)

The `BenchmarkModels` RPC (`include_rest_comparison: true`) measures the same prompt through:
1. gRPC → Python
2. REST → Python (direct HTTP POST to `/predict`)

Expected findings from this project:
- For single requests: gRPC overhead ≈ +1–5ms (negligible)
- For 10 concurrent requests: gRPC throughput ≈ +20–35% vs REST
- Streaming TTFT: gRPC ≈ 5–15ms faster first token

---

## 3. Go for LLM API Serving

### Why Go beats Python for the gRPC gateway layer

| Aspect | Go | Python (native gRPC) |
|--------|----|--------------------|
| **Concurrency model** | 1 goroutine per request (lightweight, 4KB stack) | 1 thread per request (heavy, GIL-bound) |
| **GIL** | None (true parallelism) | Global Interpreter Lock blocks concurrent inference |
| **Memory per connection** | ~4KB (goroutine stack) | ~8MB (OS thread stack) |
| **1000 concurrent connections** | ~4MB overhead | ~8GB overhead |
| **gRPC streaming** | Native goroutine per stream | asyncio callbacks (complex) |
| **Cold start** | Milliseconds | Seconds |
| **Binary size** | ~8MB static binary | Python runtime + dependencies (~500MB) |

### Python's GIL problem with gRPC streaming

The GIL (Global Interpreter Lock) means only one Python thread executes bytecode at a time.  For a pure Python gRPC server:
1. Request 1 arrives → occupies thread → runs inference (holds GIL)
2. Request 2 arrives → waits for GIL
3. Queue grows linearly

With Go:
1. Request 1 arrives → goroutine spawned → calls Python via HTTP (Go HTTP client is non-blocking)
2. Request 2 arrives simultaneously → separate goroutine → separate HTTP call
3. Both inflight simultaneously; Python processes them sequentially (its own limit)

**The Go layer absorbs the concurrency, queuing requests to Python intelligently.**

### Goroutine scheduler (Go runtime)

Go's M:N threading model (M goroutines on N OS threads, typically N = CPU count):
- Goroutines multiplexed on OS threads without kernel context switches
- I/O blocking (like waiting for Python HTTP response) parks the goroutine, freeing the thread for other goroutines
- Result: thousands of concurrent inflight LLM requests with minimal memory and CPU overhead

### Go vs Python gRPC performance (from Go team benchmarks)

| Metric | Go gRPC | Python gRPC |
|--------|---------|-------------|
| Throughput (small RPC) | ~80,000 RPC/s | ~3,000 RPC/s |
| P99 latency (small RPC) | ~1ms | ~15ms |
| Memory (1000 clients) | ~50MB | ~2GB |
| Cold start | <100ms | 2–5s |

For LLM workloads (where inference dominates), the gateway overhead matters less, but Go's advantage in **connection handling**, **streaming reliability**, and **operational simplicity** (single binary, no virtualenv) makes it the better choice.

---

## 4. Model Comparison: Gemma 3-4B vs Qwen3-4B

| Model | Org | Params | License | Strength | HuggingFace |
|-------|-----|--------|---------|----------|-------------|
| **Gemma 3-4B-IT** | Google | 4.3B | Gemma License | Instruction following, safety | `google/gemma-3-4b-it` |
| **Qwen3-4B** | Alibaba Cloud | 4.0B | Apache 2.0 | Multilingual, reasoning, code | `Qwen/Qwen3-4B` |
| **Qwen2.5-3B-Instruct** | Alibaba Cloud | 3.1B | Apache 2.0 | Lightweight, fast | `Qwen/Qwen2.5-3B-Instruct` |

### Key differences

**Qwen3-4B** (2025, latest):
- Trained with 36T tokens (vs 6T for Qwen2.5)
- Thinking mode (`/think` flag) for multi-step reasoning
- Superior at math, coding, and multilingual tasks
- Apache 2.0 license — fully open source, commercial use allowed

**Gemma 3-4B-IT**:
- Google's latest instruction-tuned 4B model (2025)
- Strong at following safety-focused instructions
- Gemma license allows research and commercial use
- Integrated with Google's LiteRT for on-device (E4B format)

**For this research platform**: both are loaded through the same Python server with NF4 quantization, making the comparison fair on identical hardware.

---

## 5. Gemini API Integration

The Gemini backend (`backends/gemini.go`) calls Google's Generative Language REST API.

This is **not a self-hosted model** — Gemini runs on Google's infrastructure.  The comparison with local models highlights:
- **Cloud latency**: network RTT to Google's datacenters (typically 100–500ms added)
- **Throughput**: effectively unlimited (subject to rate limits)
- **Cost**: ~$0.30/1M output tokens for gemini-2.0-flash
- **Quality**: typically higher than 4B local models (Gemini 2.0 Flash is ~20B+ equivalent)

Set `GEMINI_API_KEY` environment variable to enable this backend.

### Best practices for Gemini in production

1. **Vertex AI** (enterprise): use `google-cloud-aiplatform` Python SDK or Go client for IAM-based auth, audit logs, VPC service controls, and regional data residency.
2. **Gemini API** (this project): simpler, API key auth, good for research/development.
3. **Model Garden on Vertex AI**: deploy quantized Gemma models on your own GCP infrastructure for data privacy.

---

## References

- **TurboQuant**: Google Research. "TurboQuant: KV Cache and Vector Search Compression via PolarQuant + QJL." ICLR 2026. research.google
- **NF4 / QLoRA**: Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023. arXiv:2305.14314
- **Gemma 3**: Google. "Gemma 3 Technical Report." 2025.
- **Qwen3**: Alibaba Cloud. "Qwen3 Technical Report." 2025.
- **gRPC Performance**: gRPC Benchmarks. https://grpc.io/docs/guides/benchmarking/
- **Go scheduler**: Go runtime design. https://go.dev/src/runtime/HACKING.md
- **bitsandbytes NF4**: https://huggingface.co/docs/bitsandbytes
- **Johnson-Lindenstrauss lemma**: Wikipedia. https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma

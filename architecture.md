
# Gemma 2B LLM Hybrid gRPC API on Hugging Face Spaces

This Hugging Face Space hosts a powerful and efficient Large Language Model (LLM) serving solution for the `google/gemma-2b-it` model. It utilizes a **hybrid architecture** combining Go (Golang) and Python to leverage the strengths of each language, providing a high-performance gRPC API for a Flutter application.

## Architectural Considerations: Why a Hybrid Go + Python Approach?

When deploying an LLM like Gemma 2B on resource-constrained environments like Hugging Face Spaces' "CPU Basic" tier (2 vCPUs, 16GB RAM), choosing the right architecture is paramount. While a pure Python solution might seem intuitive for LLMs, it presents significant challenges for high-concurrency, low-latency API serving. Our hybrid approach strategically combines Go and Python to overcome these limitations.

### Challenges with a Pure Python API for LLM Serving:

1.  **Global Interpreter Lock (GIL) and Concurrency Bottlenecks:**
    * **Problem:** The standard CPython interpreter has a Global Interpreter Lock (GIL), which means only one native thread can execute Python bytecode at a time. LLM inference, even for a smaller model like Gemma 2B on CPU, is a computationally intensive, **CPU-bound, and often blocking operation** (meaning the Python code waits for the underlying C/C++/CUDA operations to complete).
    * **Impact:** If a pure Python gRPC server directly calls `model.generate()` from the `transformers` library within its gRPC handler, this blocking operation can effectively **stall the entire `asyncio` event loop**. This means that even if your server is designed with `async`/`await`, a single long-running inference request could prevent other concurrent client requests from being processed until that inference completes. This leads to:
        * **Poor Concurrency:** The server struggles to handle multiple simultaneous requests efficiently.
        * **Increased Latency:** Subsequent requests experience higher latency as they wait for previous inferences to finish.
        * **Reduced Responsiveness:** The API feels sluggish under load.
    * **Mitigation (Complex):** While Python offers multiprocessing to bypass the GIL, integrating it cleanly into a gRPC server that shares a single model instance (to avoid loading the model multiple times) is complex, involving inter-process communication, shared memory management, and careful resource orchestration.

2.  **API Serving Performance Overhead:**
    * **Problem:** While Python web frameworks like FastAPI with Uvicorn are highly performant for I/O-bound tasks, Python generally has higher overhead and slower raw execution speed compared to compiled languages like Go for handling a very large volume of concurrent network connections and processing API request/response logic.
    * **Impact:** This can lead to higher API layer latency and less efficient resource utilization, especially when the API layer itself becomes a bottleneck (even if the LLM inference is relatively fast).

3.  **Cold Start Latency (Less Optimized Startup):**
    * **Problem:** Python environments can have slower startup times due to interpreter initialization, module imports, and dependency resolution compared to statically compiled binaries.
    * **Impact:** While Docker pre-caching helps, the initial spin-up of a complex Python environment can contribute more to cold start latency when the Hugging Face Space first activates.

### Why the Hybrid Go + Python Approach is "Better":

Our hybrid architecture strategically separates concerns, allowing each language to do what it does best, resulting in a more robust and performant system for LLM serving on constrained hardware:

1.  **Python for Robust LLM Inference and Ecosystem (`python_model_server`):**
    * **Unmatched ML Ecosystem:** Python provides seamless access to the `google/gemma-2b-it` model through the industry-standard `transformers` library, along with PyTorch. This ensures easy model loading, tokenization, generation, and access to advanced features like quantization (e.g., `bfloat16` support for CPU-optimized inference).
    * **Manages Model Complexity:** Python effectively handles the intricate details of the LLM itself, which is its undeniable strength in the AI domain.
    * **Local HTTP API:** The Python FastAPI server exposes a simple, local HTTP/JSON API (`/predict`, `/stream_predict`) that is *only accessible from within the same Docker container*. This keeps the Python service focused solely on model inference.

2.  **Go for High-Performance gRPC API Serving (`go_grpc_server`):**
    * **Superior Concurrency and Responsiveness:** Go's lightweight goroutines and efficient concurrency model enable the gRPC server to handle a massive number of concurrent client requests with extremely low overhead. It acts as a highly responsive API gateway, efficiently proxying gRPC requests to the local Python service. This effectively **isolates the GIL-related blocking** to the Python process, preventing it from affecting the Go server's ability to accept new connections.
    * **Minimal API Layer Latency:** As a compiled language, Go executes its API logic (request parsing, HTTP client calls, response serialization) very quickly. The added latency from the local HTTP hop (Go -> Python -> Go) is typically in the range of **1-10 milliseconds**, which is negligible compared to the LLM's actual inference time (hundreds of milliseconds to seconds for Gemma 2B on CPU).
    * **Robustness and Stability:** Go's strong typing, explicit error handling, and simpler runtime lead to more reliable and maintainable API code in production environments.
    * **Native Streaming:** Go's gRPC implementation provides excellent, efficient support for server-side streaming. It can seamlessly consume the JSON-formatted stream from the Python FastAPI server and re-stream it to the Flutter client token-by-token, providing a real-time "typing" experience.

3.  **Resource Efficiency on Hugging Face CPU Basic:**
    * The Go process has a very small memory footprint, leaving the majority of the 16GB RAM available for the Python interpreter and the Gemma 2B model itself (especially when loaded in a memory-efficient `torch_dtype` like `bfloat16`).
    * The Docker multi-stage build pre-downloads the LLM model into the image layer, significantly reducing cold start times when the Space launches.

**In summary, this hybrid architecture provides a performant, concurrent, and maintainable solution for serving LLMs on CPU-constrained environments. It strategically combines Python's deep machine learning capabilities with Go's API efficiency, offering a superior experience compared to a single-language approach.**

---

## API Endpoints

This is a gRPC service. You will need a gRPC client (e.g., in Flutter) to interact with it.

**Service Name:** `llm_service.LLMService`

**Methods:**
* `GenerateText (GenerateRequest) returns (GenerateResponse)`: For non-streaming, full response generation.
* `StreamGenerateText (GenerateRequest) returns (stream StreamGenerateResponse)`: For real-time, token-by-token streaming of generated text.

**`GenerateRequest` fields:**
* `prompt` (string): The input text/query for the LLM.
* `model_id` (string, optional, default: "google/gemma-2b-it"): Specifies the model to use. While the backend loads one model, this field provides flexibility for future extensions.
* `temperature` (float, optional, default: 0.7): Controls the randomness of the generation (0.0 for deterministic, higher for more creative output).
* `max_new_tokens` (int32, optional, default: 500): The maximum number of new tokens the model should generate.

**`GenerateResponse` fields:**
* `generated_text` (string): The complete generated text.

**`StreamGenerateResponse` fields:**
* `partial_text` (string): A piece of text generated by the model (e.g., a word or a sub-word token).
* `done` (bool): `true` if this is the final chunk of the response, `false` otherwise.

## Example using gRPCurl (for testing)

First, install `grpcurl` (e.g., `go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest`).
Then, get your Hugging Face Space URL (e.g., `your-space-name.hf.space`).

**Non-streaming example:**
```bash
grpcurl -plaintext -d '{"prompt": "Tell me a short story about a very curious robot.", "model_id": "google/gemma-2b-it", "max_new_tokens": 120, "temperature": 0.8}' \
  your-space-name.hf.space:443 llm_service.LLMService/GenerateText
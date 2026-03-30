// main.go — Multi-LLM gRPC server for the Go-gRPC-Gemma-API research platform.
//
// This server demonstrates two key theses:
//
//  1. Go + gRPC is significantly better than Python-native gRPC for high-concurrency
//     LLM serving.  Go's goroutine scheduler lets thousands of requests be
//     multiplexed over a small thread pool without Python's GIL blocking them.
//
//  2. gRPC (Protobuf/HTTP-2) outperforms REST (JSON/HTTP-1.1) for LLM gateway
//     traffic, particularly under concurrent load, because of:
//     - Binary framing (Protobuf is 5–10× smaller than equivalent JSON)
//     - HTTP/2 multiplexing (many requests share one TCP connection)
//     - Native streaming (server-push without polling)
//
// Model routing is done by model_id prefix:
//
//	"gemma:*"  → Python FastAPI + HuggingFace (Gemma 3-4B-IT, NF4 4-bit quant)
//	"qwen:*"   → Ollama local server (Qwen2.5-3B, Q4_K_M GGUF quant)
//	"gemini:*" → Google Gemini REST API (requires GEMINI_API_KEY)
//	(default)  → Gemma backend
//
// Environment variables:
//
//	PORT            gRPC listen port (default: 7860)
//	PYTHON_HOST     Python FastAPI URL (default: http://localhost:8001)
//	OLLAMA_HOST     Ollama URL (default: http://localhost:11434)
//	GEMINI_API_KEY  Gemini API key (optional; Gemini backend disabled if unset)
//	GEMINI_MODEL    Gemini model name (default: gemini-2.0-flash)
//	GEMMA_MODEL_ID  HuggingFace model ID (default: google/gemma-3-4b-it)
//	GEMMA_QUANT     Quantization tag for Gemma (default: int4_nf4)
package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	"github.com/bharathgajula/go-grpc-gemma-api/backends"
	"github.com/bharathgajula/go-grpc-gemma-api/metrics"
)

// ─── gRPC server ─────────────────────────────────────────────────────────────

// LLMServer implements the generated LLMServiceServer interface.
type LLMServer struct {
	UnimplementedLLMServiceServer
	backendMap  map[string]backends.LLMBackend // prefix → backend
	defaultBack backends.LLMBackend
	pythonHost  string
	collector   *metrics.Collector
}

// NewLLMServer wires all backends together.
func NewLLMServer(pythonHost, ollamaHost, geminiKey, geminiModel, gemmaModelID, gemmaQuant string) *LLMServer {
	gemma := backends.NewGemmaBackend(pythonHost, gemmaModelID, gemmaQuant)
	ollama := backends.NewOllamaBackend(ollamaHost, "qwen2.5:3b")

	bmap := map[string]backends.LLMBackend{
		"gemma": gemma,
		"qwen":  ollama,
	}

	if geminiKey != "" {
		bmap["gemini"] = backends.NewGeminiBackend(geminiKey, geminiModel)
		log.Printf("Gemini backend enabled: %s", geminiModel)
	}

	return &LLMServer{
		backendMap:  bmap,
		defaultBack: gemma,
		pythonHost:  pythonHost,
		collector:   metrics.NewCollector(),
	}
}

// routeBackend returns the appropriate backend based on model_id prefix.
func (s *LLMServer) routeBackend(modelID string) backends.LLMBackend {
	lower := strings.ToLower(modelID)
	for prefix, b := range s.backendMap {
		if strings.HasPrefix(lower, prefix) {
			return b
		}
	}
	return s.defaultBack
}

// ─── GenerateText (unary) ────────────────────────────────────────────────────

func (s *LLMServer) GenerateText(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	log.Printf("[GenerateText] model=%s prompt=%q temp=%.2f max_tokens=%d",
		req.ModelId, truncate(req.Prompt, 60), req.Temperature, req.MaxNewTokens)

	backend := s.routeBackend(req.ModelId)
	result, err := backend.Generate(ctx, req.Prompt, req.Temperature, req.MaxNewTokens)
	if err != nil {
		log.Printf("[GenerateText] error: %v", err)
		return nil, status.Errorf(codes.Internal, "inference error: %v", err)
	}

	// Record metrics
	s.collector.Record(metrics.Sample{
		ModelID:     backend.ModelID(),
		LatencyMS:   result.LatencyMS,
		TokenCount:  result.TokenCount,
		Timestamp:   time.Now(),
		BackendType: backend.Name(),
		Via:         "grpc",
	})

	log.Printf("[GenerateText] ok: %d chars, %.0fms", len(result.Text), result.LatencyMS)
	return &GenerateResponse{GeneratedText: result.Text}, nil
}

// ─── StreamGenerateText (server-side streaming) ───────────────────────────────

func (s *LLMServer) StreamGenerateText(req *GenerateRequest, stream grpc.ServerStreamingServer[StreamGenerateResponse]) error {
	log.Printf("[StreamGenerateText] model=%s prompt=%q", req.ModelId, truncate(req.Prompt, 60))

	backend := s.routeBackend(req.ModelId)

	tokenCount := 0
	var ttft float64

	err := backend.StreamGenerate(stream.Context(), req.Prompt, req.Temperature, req.MaxNewTokens,
		func(chunk backends.StreamChunk) error {
			if chunk.TTFT > 0 {
				ttft = float64(chunk.TTFT.Milliseconds())
			}
			tokenCount++

			if err := stream.Send(&StreamGenerateResponse{
				PartialText: chunk.PartialText,
				Done:        chunk.Done,
			}); err != nil {
				return err
			}

			select {
			case <-stream.Context().Done():
				return stream.Context().Err()
			default:
				return nil
			}
		})

	if err != nil {
		if strings.Contains(err.Error(), "context") {
			return status.Errorf(codes.Canceled, "stream cancelled: %v", err)
		}
		return status.Errorf(codes.Internal, "stream error: %v", err)
	}

	s.collector.Record(metrics.Sample{
		ModelID:     backend.ModelID(),
		TTFTMS:      ttft,
		TokenCount:  tokenCount,
		Timestamp:   time.Now(),
		BackendType: backend.Name(),
		Via:         "grpc",
	})

	log.Printf("[StreamGenerateText] ok: %d tokens, ttft=%.0fms", tokenCount, ttft)
	return nil
}

// ─── EvaluateGeneration ───────────────────────────────────────────────────────

func (s *LLMServer) EvaluateGeneration(ctx context.Context, req *EvaluateRequest) (*EvaluateResponse, error) {
	log.Printf("[EvaluateGeneration] models=%v prompt=%q", req.ModelIds, truncate(req.Prompt, 60))

	var results []*ModelMetrics

	for _, modelID := range req.ModelIds {
		backend := s.routeBackend(modelID)

		start := time.Now()
		result, err := backend.Generate(ctx, req.Prompt, req.Temperature, req.MaxNewTokens)
		latencyMS := float64(time.Since(start).Milliseconds())

		mm := &ModelMetrics{
			ModelId:      backend.ModelID(),
			BackendType:  backend.Name(),
			Quantization: backend.Quantization(),
			LatencyMs:    latencyMS,
		}

		if err != nil {
			mm.GeneratedText = fmt.Sprintf("[ERROR: %v]", err)
			results = append(results, mm)
			continue
		}

		mm.GeneratedText = result.Text
		mm.TokenCount = int32(result.TokenCount)
		if latencyMS > 0 && result.TokenCount > 0 {
			mm.TokensPerSec = float64(result.TokenCount) / (latencyMS / 1000.0)
		}

		// BLEU + ROUGE via Python /evaluate
		if req.ReferenceAnswer != "" {
			bleu, rougeL, evalErr := metrics.CallEvaluate(ctx, s.pythonHost, result.Text, req.ReferenceAnswer)
			if evalErr != nil {
				log.Printf("[EvaluateGeneration] eval error for %s: %v", modelID, evalErr)
			} else {
				mm.BleuScore = bleu
				mm.RougeL = rougeL
			}
		}

		// Cost estimation (Gemini only; local models are $0)
		if backend.Name() == "gemini" {
			if gb, ok := backend.(*backends.GeminiBackend); ok {
				mm.EstimatedCostUsd = gb.EstimateCost(result.TokenCount)
			}
		}

		s.collector.Record(metrics.Sample{
			ModelID:     backend.ModelID(),
			LatencyMS:   latencyMS,
			TokenCount:  result.TokenCount,
			Timestamp:   time.Now(),
			BackendType: backend.Name(),
			Via:         "grpc",
		})

		results = append(results, mm)
	}

	return &EvaluateResponse{Results: results}, nil
}

// ─── BenchmarkModels ──────────────────────────────────────────────────────────

func (s *LLMServer) BenchmarkModels(ctx context.Context, req *BenchmarkRequest) (*BenchmarkResponse, error) {
	log.Printf("[BenchmarkModels] models=%v prompts=%d runs=%d",
		req.ModelIds, len(req.Prompts), req.RunsPerPrompt)

	if len(req.Prompts) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one prompt required")
	}
	if len(req.ModelIds) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one model_id required")
	}

	runsPerPrompt := int(req.RunsPerPrompt)
	if runsPerPrompt <= 0 {
		runsPerPrompt = 3
	}
	temp := req.Temperature
	if temp == 0 {
		temp = 0.7
	}
	maxTokens := req.MaxNewTokens
	if maxTokens == 0 {
		maxTokens = 128
	}

	// Build model map from request
	modelMap := make(map[string]backends.LLMBackend)
	for _, id := range req.ModelIds {
		modelMap[id] = s.routeBackend(id)
	}

	cfg := metrics.BenchmarkConfig{
		Prompts:            req.Prompts,
		Models:             modelMap,
		RunsPerPrompt:      runsPerPrompt,
		Temperature:        temp,
		MaxNewTokens:       maxTokens,
		IncludeRESTCompare: req.IncludeRestComparison,
		PythonHost:         s.pythonHost,
		OutputDir:          "benchmarks",
	}

	runner := metrics.NewBenchmarkRunner(cfg)
	report, path, err := runner.Run(ctx)
	if err != nil {
		log.Printf("[BenchmarkModels] error: %v", err)
		return nil, status.Errorf(codes.Internal, "benchmark failed: %v", err)
	}

	// Convert aggregated results to proto messages
	var aggProtos []*AggregatedMetrics
	for _, a := range report.Aggregated {
		aggProtos = append(aggProtos, &AggregatedMetrics{
			ModelId:              a.ModelID,
			AvgLatencyMs:         a.AvgLatencyMS,
			P50LatencyMs:         a.P50LatencyMS,
			P95LatencyMs:         a.P95LatencyMS,
			P99LatencyMs:         a.P99LatencyMS,
			AvgTokensPerSec:      a.AvgTokensPerSec,
			AvgBleu:              0, // filled by EvaluateGeneration
			AvgRougeL:            0,
			BackendType:          a.BackendType,
			Quantization:         a.Quantization,
			TotalRuns:            int32(a.TotalRuns),
			RestAvgLatencyMs:     a.RESTAvgLatencyMS,
			GrpcVsRestDeltaMs:    a.GRPCvsRESTDeltaMS,
		})
	}

	log.Printf("[BenchmarkModels] done. Report: %s", path)
	return &BenchmarkResponse{
		Aggregated:      aggProtos,
		JsonReportPath:  path,
		Summary:         report.Summary,
	}, nil
}

// ─── Startup helpers ──────────────────────────────────────────────────────────

func waitForPythonServer(pythonHost string, maxRetries int) error {
	client := &http.Client{Timeout: 5 * time.Second}
	url := pythonHost + "/health"

	for i := 0; i < maxRetries; i++ {
		resp, err := client.Get(url)
		if err == nil && resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			log.Printf("Python server is ready at %s", pythonHost)
			return nil
		}
		if resp != nil {
			resp.Body.Close()
		}
		log.Printf("Waiting for Python server... attempt %d/%d", i+1, maxRetries)
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("python server not ready after %d attempts", maxRetries)
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// ─── main ─────────────────────────────────────────────────────────────────────

func main() {
	pythonHost  := getenv("PYTHON_HOST",     "http://localhost:8001")
	ollamaHost  := getenv("OLLAMA_HOST",     "http://localhost:11434")
	geminiKey   := getenv("GEMINI_API_KEY",  "")
	geminiModel := getenv("GEMINI_MODEL",    "gemini-2.0-flash")
	gemmaModelID := getenv("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
	gemmaQuant  := getenv("GEMMA_QUANT",     "int4_nf4")
	port        := getenv("PORT",            "7860")

	log.Printf("═══════════════════════════════════════════════════════════")
	log.Printf("  Go-gRPC Multi-LLM Research Platform")
	log.Printf("  Port: %s | Python: %s | Ollama: %s", port, pythonHost, ollamaHost)
	log.Printf("  Gemma model: %s (%s)", gemmaModelID, gemmaQuant)
	if geminiKey != "" {
		log.Printf("  Gemini model: %s", geminiModel)
	}
	log.Printf("═══════════════════════════════════════════════════════════")

	// Wait for Python server (required for Gemma + evaluation)
	log.Printf("Waiting for Python server to be ready...")
	if err := waitForPythonServer(pythonHost, 30); err != nil {
		log.Fatalf("Python server unavailable: %v", err)
	}

	// Create listener
	listener, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("Failed to listen on port %s: %v", port, err)
	}

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(4*1024*1024),
		grpc.MaxSendMsgSize(4*1024*1024),
	)

	llmServer := NewLLMServer(pythonHost, ollamaHost, geminiKey, geminiModel, gemmaModelID, gemmaQuant)
	RegisterLLMServiceServer(grpcServer, llmServer)
	reflection.Register(grpcServer)

	log.Printf("gRPC server listening on :%s", port)
	log.Printf("Test with: grpcurl -plaintext localhost:%s list", port)
	log.Printf("Models available:")
	log.Printf("  gemma:<model_id>  → HuggingFace/Python (NF4 4-bit)")
	log.Printf("  qwen:<model_tag>  → Ollama (Q4_K_M GGUF)")
	if geminiKey != "" {
		log.Printf("  gemini:<model>    → Google Gemini API")
	}

	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("gRPC server failed: %v", err)
	}
}

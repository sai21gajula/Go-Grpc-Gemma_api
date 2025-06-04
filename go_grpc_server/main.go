package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// LLMServer implements the LLMServiceServer interface
type LLMServer struct {
	UnimplementedLLMServiceServer
	pythonHost string
	httpClient *http.Client
}

var (
	unaryRequestsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "llm_unary_requests_total",
		Help: "Total number of unary GenerateText requests",
	})
	streamRequestsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "llm_stream_requests_total",
		Help: "Total number of StreamGenerateText requests",
	})
	streamTokensTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "llm_stream_tokens_total",
		Help: "Total number of tokens streamed to clients",
	})
	requestDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "llm_request_duration_seconds",
		Help:    "Duration of requests to the Python server",
		Buckets: prometheus.DefBuckets,
	})
)

// PythonRequest represents the request format for the Python server
type PythonRequest struct {
	Prompt       string  `json:"prompt"`
	ModelID      string  `json:"model_id"`
	Temperature  float32 `json:"temperature"`
	MaxNewTokens int32   `json:"max_new_tokens"`
}

// PythonResponse represents the response format from the Python server
type PythonResponse struct {
	GeneratedText string `json:"generated_text"`
	Error         string `json:"error,omitempty"`
}

// PythonStreamResponse represents the streaming response format from the Python server
type PythonStreamResponse struct {
	PartialText string `json:"partial_text"`
	Done        bool   `json:"done"`
	Error       string `json:"error,omitempty"`
}

// NewLLMServer creates a new LLM server instance
func NewLLMServer(pythonHost string) *LLMServer {
	return &LLMServer{
		pythonHost: pythonHost,
		httpClient: &http.Client{
			Timeout: 300 * time.Second, // 5 minutes timeout for LLM requests
		},
	}
}

func init() {
	prometheus.MustRegister(unaryRequestsTotal)
	prometheus.MustRegister(streamRequestsTotal)
	prometheus.MustRegister(streamTokensTotal)
	prometheus.MustRegister(requestDuration)
}

// GenerateText implements the unary RPC for text generation
func (s *LLMServer) GenerateText(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	log.Printf("Received GenerateText request: prompt=%s, model_id=%s, temperature=%f, max_new_tokens=%d",
		req.Prompt, req.ModelId, req.Temperature, req.MaxNewTokens)

	start := time.Now()
	unaryRequestsTotal.Inc()

	// Prepare request for Python server
	pythonReq := PythonRequest{
		Prompt:       req.Prompt,
		ModelID:      req.ModelId,
		Temperature:  req.Temperature,
		MaxNewTokens: req.MaxNewTokens,
	}

	jsonData, err := json.Marshal(pythonReq)
	if err != nil {
		log.Printf("Error marshaling request: %v", err)
		return nil, status.Errorf(codes.Internal, "Failed to marshal request: %v", err)
	}

	// Make HTTP request to Python server
	url := fmt.Sprintf("%s/predict", s.pythonHost)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Error creating HTTP request: %v", err)
		return nil, status.Errorf(codes.Internal, "Failed to create HTTP request: %v", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		log.Printf("Error making HTTP request: %v", err)
		return nil, status.Errorf(codes.Unavailable, "Python server unavailable: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("Python server returned status %d: %s", resp.StatusCode, string(body))
		return nil, status.Errorf(codes.Internal, "Python server error: status %d", resp.StatusCode)
	}

	// Parse response
	var pythonResp PythonResponse
	if err := json.NewDecoder(resp.Body).Decode(&pythonResp); err != nil {
		log.Printf("Error decoding response: %v", err)
		return nil, status.Errorf(codes.Internal, "Failed to decode response: %v", err)
	}

	if pythonResp.Error != "" {
		log.Printf("Python server returned error: %s", pythonResp.Error)
		return nil, status.Errorf(codes.Internal, "Python server error: %s", pythonResp.Error)
	}

	log.Printf("Successfully generated text: %d characters", len(pythonResp.GeneratedText))
	requestDuration.Observe(time.Since(start).Seconds())
	return &GenerateResponse{
		GeneratedText: pythonResp.GeneratedText,
	}, nil
}

// StreamGenerateText implements the server-side streaming RPC for text generation
func (s *LLMServer) StreamGenerateText(req *GenerateRequest, stream grpc.ServerStreamingServer[StreamGenerateResponse]) error {
	log.Printf("Received StreamGenerateText request: prompt=%s, model_id=%s, temperature=%f, max_new_tokens=%d",
		req.Prompt, req.ModelId, req.Temperature, req.MaxNewTokens)

	start := time.Now()
	streamRequestsTotal.Inc()

	// Prepare request for Python server
	pythonReq := PythonRequest{
		Prompt:       req.Prompt,
		ModelID:      req.ModelId,
		Temperature:  req.Temperature,
		MaxNewTokens: req.MaxNewTokens,
	}

	jsonData, err := json.Marshal(pythonReq)
	if err != nil {
		log.Printf("Error marshaling request: %v", err)
		return status.Errorf(codes.Internal, "Failed to marshal request: %v", err)
	}

	// Make HTTP request to Python server for streaming
	url := fmt.Sprintf("%s/stream_predict", s.pythonHost)
	httpReq, err := http.NewRequestWithContext(stream.Context(), "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Error creating HTTP request: %v", err)
		return status.Errorf(codes.Internal, "Failed to create HTTP request: %v", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/plain")

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		log.Printf("Error making HTTP request: %v", err)
		return status.Errorf(codes.Unavailable, "Python server unavailable: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("Python server returned status %d: %s", resp.StatusCode, string(body))
		return status.Errorf(codes.Internal, "Python server error: status %d", resp.StatusCode)
	}

	// Read streaming response line by line
	decoder := json.NewDecoder(resp.Body)
	tokenCount := 0

	for {
		var pythonResp PythonStreamResponse
		if err := decoder.Decode(&pythonResp); err != nil {
			if err == io.EOF {
				break
			}
			log.Printf("Error decoding streaming response: %v", err)
			return status.Errorf(codes.Internal, "Failed to decode streaming response: %v", err)
		}

		if pythonResp.Error != "" {
			log.Printf("Python server returned streaming error: %s", pythonResp.Error)
			return status.Errorf(codes.Internal, "Python server error: %s", pythonResp.Error)
		}

		// Send the partial text to the client
		if err := stream.Send(&StreamGenerateResponse{
			PartialText: pythonResp.PartialText,
			Done:        pythonResp.Done,
		}); err != nil {
			log.Printf("Error sending stream response: %v", err)
			return status.Errorf(codes.Internal, "Failed to send stream response: %v", err)
		}

		tokenCount++
		streamTokensTotal.Inc()
		if pythonResp.Done {
			log.Printf("Streaming completed: sent %d tokens", tokenCount)
			break
		}

		// Check if context is cancelled
		select {
		case <-stream.Context().Done():
			log.Printf("Stream context cancelled")
			return status.Errorf(codes.Canceled, "Stream cancelled by client")
		default:
		}
	}

	requestDuration.Observe(time.Since(start).Seconds())

	return nil
}

// waitForPythonServer waits for the Python server to be ready
func waitForPythonServer(pythonHost string, maxRetries int) error {
	client := &http.Client{Timeout: 5 * time.Second}
	url := fmt.Sprintf("%s/health", pythonHost)

	for i := 0; i < maxRetries; i++ {
		resp, err := client.Get(url)
		if err == nil && resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			log.Printf("Python server is ready")
			return nil
		}
		if resp != nil {
			resp.Body.Close()
		}

		log.Printf("Waiting for Python server... attempt %d/%d", i+1, maxRetries)
		time.Sleep(2 * time.Second)
	}

	return fmt.Errorf("Python server not ready after %d attempts", maxRetries)
}

func startMetricsServer(port string) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	go func() {
		log.Printf("Metrics server listening on :%s/metrics", port)
		if err := http.ListenAndServe(":"+port, mux); err != nil {
			log.Printf("Metrics server error: %v", err)
		}
	}()
}

func main() {
	// Configuration from environment variables
	pythonHost := os.Getenv("PYTHON_HOST")
	if pythonHost == "" {
		pythonHost = "http://localhost:8001"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "7860"
	}

	metricsPort := os.Getenv("METRICS_PORT")
	if metricsPort == "" {
		metricsPort = "9090"
	}

	log.Printf("Starting Go gRPC server on port %s", port)
	log.Printf("Python server host: %s", pythonHost)
	log.Printf("Metrics exposed at :%s/metrics", metricsPort)
	startMetricsServer(metricsPort)

	// Wait for Python server to be ready
	log.Printf("Waiting for Python server to be ready...")
	if err := waitForPythonServer(pythonHost, 30); err != nil {
		log.Fatalf("Failed to connect to Python server: %v", err)
	}

	// Create listener
	listener, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("Failed to listen on port %s: %v", port, err)
	}

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(4*1024*1024), // 4MB
		grpc.MaxSendMsgSize(4*1024*1024), // 4MB
	)

	// Register LLM service
	llmServer := NewLLMServer(pythonHost)
	RegisterLLMServiceServer(grpcServer, llmServer)

	// Enable reflection for debugging
	reflection.Register(grpcServer)

	log.Printf("gRPC server listening on :%s", port)
	log.Printf("Use grpcurl for testing: grpcurl -plaintext localhost:%s list", port)

	// Start serving
	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("Failed to serve gRPC server: %v", err)
	}
}

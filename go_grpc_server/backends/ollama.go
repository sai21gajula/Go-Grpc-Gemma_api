// Package backends — Ollama backend.
//
// Routes inference to a locally-running Ollama instance (default port 11434).
// Ollama serves Qwen2.5-3B and other GGUF-quantized models via an
// OpenAI-compatible REST API.
//
// Model quantization: Ollama uses Q4_K_M GGUF by default when pulling
// standard model tags, which is functionally equivalent to INT4 / 4-bit
// quantization.  This matches the NF4 precision of Google's E4B LiteRT models.
//
// Usage:
//
//	backend := backends.NewOllamaBackend("http://localhost:11434", "qwen2.5:3b")
package backends

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OllamaBackend calls Ollama's /api/generate endpoint.
type OllamaBackend struct {
	host         string
	ollamaModel  string // e.g. "qwen2.5:3b"
	quantization string
	httpClient   *http.Client
}

// NewOllamaBackend creates an OllamaBackend.
// ollamaModel is the tag used with `ollama pull`, e.g. "qwen2.5:3b".
func NewOllamaBackend(host, ollamaModel string) *OllamaBackend {
	return &OllamaBackend{
		host:         strings.TrimRight(host, "/"),
		ollamaModel:  ollamaModel,
		quantization: "q4_k_m", // Ollama default GGUF quantization
		httpClient:   &http.Client{Timeout: 300 * time.Second},
	}
}

func (b *OllamaBackend) Name() string        { return "ollama" }
func (b *OllamaBackend) ModelID() string      { return "qwen:" + b.ollamaModel }
func (b *OllamaBackend) Quantization() string { return b.quantization }

// ollamaRequest matches Ollama's /api/generate request schema.
type ollamaRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	Stream  bool           `json:"stream"`
	Options ollamaOptions  `json:"options"`
}

type ollamaOptions struct {
	Temperature float32 `json:"temperature"`
	NumPredict  int32   `json:"num_predict"`
}

// ollamaResponse is one NDJSON line from /api/generate.
type ollamaResponse struct {
	Response string `json:"response"` // partial token
	Done     bool   `json:"done"`
	// Ollama includes eval stats on the final done=true line
	EvalCount    int     `json:"eval_count"`
	EvalDuration int64   `json:"eval_duration"` // nanoseconds
}

func (b *OllamaBackend) Generate(ctx context.Context, prompt string, temperature float32, maxTokens int32) (*GenerateResult, error) {
	start := time.Now()

	reqBody, _ := json.Marshal(ollamaRequest{
		Model:  b.ollamaModel,
		Prompt: prompt,
		Stream: false, // non-streaming → full response in one JSON object
		Options: ollamaOptions{
			Temperature: temperature,
			NumPredict:  maxTokens,
		},
	})

	httpReq, err := http.NewRequestWithContext(ctx, "POST", b.host+"/api/generate", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("ollama build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama status %d: %s", resp.StatusCode, raw)
	}

	var ollamaResp ollamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("ollama decode: %w", err)
	}

	latency := float64(time.Since(start).Milliseconds())
	tokensPerSec := 0.0
	if ollamaResp.EvalDuration > 0 {
		tokensPerSec = float64(ollamaResp.EvalCount) / (float64(ollamaResp.EvalDuration) / 1e9)
	}
	_ = tokensPerSec // used in streaming path

	return &GenerateResult{
		Text:       ollamaResp.Response,
		TokenCount: ollamaResp.EvalCount,
		LatencyMS:  latency,
	}, nil
}

func (b *OllamaBackend) StreamGenerate(ctx context.Context, prompt string, temperature float32, maxTokens int32, send func(StreamChunk) error) error {
	reqBody, _ := json.Marshal(ollamaRequest{
		Model:  b.ollamaModel,
		Prompt: prompt,
		Stream: true,
		Options: ollamaOptions{
			Temperature: temperature,
			NumPredict:  maxTokens,
		},
	})

	httpReq, err := http.NewRequestWithContext(ctx, "POST", b.host+"/api/generate", bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("ollama stream build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("ollama stream http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("ollama stream status %d: %s", resp.StatusCode, raw)
	}

	start := time.Now()
	firstToken := true
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var chunk ollamaResponse
		if err := json.Unmarshal(line, &chunk); err != nil {
			return fmt.Errorf("ollama stream decode: %w", err)
		}

		sc := StreamChunk{PartialText: chunk.Response, Done: chunk.Done}
		if firstToken && chunk.Response != "" {
			sc.TTFT = time.Since(start)
			firstToken = false
		}
		if err := send(sc); err != nil {
			return err
		}
		if chunk.Done {
			break
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("ollama stream scan: %w", err)
	}
	return nil
}

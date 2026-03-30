// Package backends — Gemma backend.
// Wraps the local Python FastAPI inference server (port 8001 by default).
// Supports any HuggingFace model including Gemma 3-4B-IT with NF4 4-bit quantization.
package backends

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// GemmaBackend proxies requests to the Python FastAPI model server.
type GemmaBackend struct {
	pythonHost   string
	modelID      string
	quantization string
	httpClient   *http.Client
}

// NewGemmaBackend creates a GemmaBackend that talks to pythonHost.
// modelID should be the HuggingFace model identifier.
// quant should be "fp32", "int4_nf4", etc.
func NewGemmaBackend(pythonHost, modelID, quant string) *GemmaBackend {
	return &GemmaBackend{
		pythonHost:   strings.TrimRight(pythonHost, "/"),
		modelID:      modelID,
		quantization: quant,
		httpClient:   &http.Client{Timeout: 300 * time.Second},
	}
}

func (b *GemmaBackend) Name() string         { return "gemma_hf" }
func (b *GemmaBackend) ModelID() string       { return b.modelID }
func (b *GemmaBackend) Quantization() string  { return b.quantization }

type pythonRequest struct {
	Prompt       string  `json:"prompt"`
	ModelID      string  `json:"model_id"`
	Temperature  float32 `json:"temperature"`
	MaxNewTokens int32   `json:"max_new_tokens"`
}

type pythonResponse struct {
	GeneratedText string `json:"generated_text"`
	Error         string `json:"error,omitempty"`
}

type pythonStreamResponse struct {
	PartialText string `json:"partial_text"`
	Done        bool   `json:"done"`
	Error       string `json:"error,omitempty"`
}

func (b *GemmaBackend) Generate(ctx context.Context, prompt string, temperature float32, maxTokens int32) (*GenerateResult, error) {
	start := time.Now()

	body, _ := json.Marshal(pythonRequest{
		Prompt:       prompt,
		ModelID:      b.modelID,
		Temperature:  temperature,
		MaxNewTokens: maxTokens,
	})

	httpReq, err := http.NewRequestWithContext(ctx, "POST", b.pythonHost+"/predict", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("gemma build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemma http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemma server status %d: %s", resp.StatusCode, raw)
	}

	var pyResp pythonResponse
	if err := json.NewDecoder(resp.Body).Decode(&pyResp); err != nil {
		return nil, fmt.Errorf("gemma decode: %w", err)
	}
	if pyResp.Error != "" {
		return nil, fmt.Errorf("gemma inference: %s", pyResp.Error)
	}

	words := strings.Fields(pyResp.GeneratedText)
	return &GenerateResult{
		Text:       pyResp.GeneratedText,
		TokenCount: len(words), // rough approximation; Python server may return exact count
		LatencyMS:  float64(time.Since(start).Milliseconds()),
	}, nil
}

func (b *GemmaBackend) StreamGenerate(ctx context.Context, prompt string, temperature float32, maxTokens int32, send func(StreamChunk) error) error {
	body, _ := json.Marshal(pythonRequest{
		Prompt:       prompt,
		ModelID:      b.modelID,
		Temperature:  temperature,
		MaxNewTokens: maxTokens,
	})

	httpReq, err := http.NewRequestWithContext(ctx, "POST", b.pythonHost+"/stream_predict", bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("gemma stream build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/x-ndjson")

	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("gemma stream http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("gemma stream status %d: %s", resp.StatusCode, raw)
	}

	start := time.Now()
	firstToken := true
	decoder := json.NewDecoder(resp.Body)

	for {
		var chunk pythonStreamResponse
		if err := decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("gemma stream decode: %w", err)
		}
		if chunk.Error != "" {
			return fmt.Errorf("gemma stream inference: %s", chunk.Error)
		}

		sc := StreamChunk{PartialText: chunk.PartialText, Done: chunk.Done}
		if firstToken && chunk.PartialText != "" {
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
	return nil
}

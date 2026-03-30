// Package backends — Gemini backend (optional, cloud).
//
// Calls Google's Generative Language REST API.
// Set GEMINI_API_KEY to enable; if unset the backend returns an error.
// Supported models: gemini-2.0-flash, gemini-1.5-pro, gemini-2.5-pro
//
// Cost estimation (as of 2025-Q4 pricing):
//   gemini-2.0-flash:  $0.075 / 1M input tokens, $0.30 / 1M output tokens
//   gemini-1.5-pro:    $1.25  / 1M input tokens, $5.00 / 1M output tokens
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

const geminiBaseURL = "https://generativelanguage.googleapis.com/v1beta/models"

// Cost per 1M output tokens in USD (approximate 2025-Q4 pricing)
var geminiCostPer1MOutputTokens = map[string]float64{
	"gemini-2.0-flash": 0.30,
	"gemini-1.5-pro":   5.00,
	"gemini-2.5-pro":   10.00,
}

// GeminiBackend calls the Gemini REST API.
type GeminiBackend struct {
	apiKey     string
	modelName  string // e.g. "gemini-2.0-flash"
	httpClient *http.Client
}

// NewGeminiBackend creates a GeminiBackend.
// apiKey must be non-empty; modelName is e.g. "gemini-2.0-flash".
func NewGeminiBackend(apiKey, modelName string) *GeminiBackend {
	return &GeminiBackend{
		apiKey:     apiKey,
		modelName:  modelName,
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}
}

func (b *GeminiBackend) Name() string        { return "gemini" }
func (b *GeminiBackend) ModelID() string      { return "gemini:" + b.modelName }
func (b *GeminiBackend) Quantization() string { return "none" } // cloud model, no local quant

// Gemini REST API request/response structures
type geminiContent struct {
	Parts []geminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenerateRequest struct {
	Contents         []geminiContent         `json:"contents"`
	GenerationConfig geminiGenerationConfig  `json:"generationConfig"`
}

type geminiGenerationConfig struct {
	Temperature    float32 `json:"temperature"`
	MaxOutputTokens int32  `json:"maxOutputTokens"`
}

type geminiCandidate struct {
	Content geminiContent `json:"content"`
}

type geminiUsageMetadata struct {
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type geminiGenerateResponse struct {
	Candidates    []geminiCandidate   `json:"candidates"`
	UsageMetadata geminiUsageMetadata `json:"usageMetadata"`
}

func (b *GeminiBackend) Generate(ctx context.Context, prompt string, temperature float32, maxTokens int32) (*GenerateResult, error) {
	if b.apiKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY not set")
	}

	start := time.Now()

	reqBody, _ := json.Marshal(geminiGenerateRequest{
		Contents: []geminiContent{
			{Parts: []geminiPart{{Text: prompt}}, Role: "user"},
		},
		GenerationConfig: geminiGenerationConfig{
			Temperature:     temperature,
			MaxOutputTokens: maxTokens,
		},
	})

	url := fmt.Sprintf("%s/%s:generateContent?key=%s", geminiBaseURL, b.modelName, b.apiKey)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("gemini build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini status %d: %s", resp.StatusCode, raw)
	}

	var gemResp geminiGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&gemResp); err != nil {
		return nil, fmt.Errorf("gemini decode: %w", err)
	}

	if len(gemResp.Candidates) == 0 || len(gemResp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("gemini: empty response")
	}

	text := gemResp.Candidates[0].Content.Parts[0].Text
	tokens := gemResp.UsageMetadata.CandidatesTokenCount

	return &GenerateResult{
		Text:       text,
		TokenCount: tokens,
		LatencyMS:  float64(time.Since(start).Milliseconds()),
	}, nil
}

// StreamGenerate for Gemini — uses SSE streaming endpoint.
func (b *GeminiBackend) StreamGenerate(ctx context.Context, prompt string, temperature float32, maxTokens int32, send func(StreamChunk) error) error {
	if b.apiKey == "" {
		return fmt.Errorf("GEMINI_API_KEY not set")
	}

	reqBody, _ := json.Marshal(geminiGenerateRequest{
		Contents: []geminiContent{
			{Parts: []geminiPart{{Text: prompt}}, Role: "user"},
		},
		GenerationConfig: geminiGenerationConfig{
			Temperature:     temperature,
			MaxOutputTokens: maxTokens,
		},
	})

	url := fmt.Sprintf("%s/%s:streamGenerateContent?key=%s&alt=sse", geminiBaseURL, b.modelName, b.apiKey)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("gemini stream build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("gemini stream http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("gemini stream status %d: %s", resp.StatusCode, raw)
	}

	start := time.Now()
	firstToken := true

	// Gemini SSE lines are: "data: {json}" or "data: [DONE]"
	buf := make([]byte, 32*1024)
	remainder := ""
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			chunk := remainder + string(buf[:n])
			lines := strings.Split(chunk, "\n")
			remainder = lines[len(lines)-1]

			for _, line := range lines[:len(lines)-1] {
				line = strings.TrimSpace(line)
				if !strings.HasPrefix(line, "data:") {
					continue
				}
				data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				if data == "[DONE]" {
					_ = send(StreamChunk{Done: true})
					return nil
				}

				var gemResp geminiGenerateResponse
				if err := json.Unmarshal([]byte(data), &gemResp); err != nil {
					continue
				}
				if len(gemResp.Candidates) == 0 || len(gemResp.Candidates[0].Content.Parts) == 0 {
					continue
				}

				text := gemResp.Candidates[0].Content.Parts[0].Text
				sc := StreamChunk{PartialText: text}
				if firstToken && text != "" {
					sc.TTFT = time.Since(start)
					firstToken = false
				}
				if err := send(sc); err != nil {
					return err
				}
			}
		}

		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return fmt.Errorf("gemini stream read: %w", readErr)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}

	_ = send(StreamChunk{Done: true})
	return nil
}

// EstimateCost returns the approximate USD cost for generating outputTokens tokens.
func (b *GeminiBackend) EstimateCost(outputTokens int) float64 {
	costPer1M, ok := geminiCostPer1MOutputTokens[b.modelName]
	if !ok {
		costPer1M = 1.0 // conservative default
	}
	return float64(outputTokens) / 1e6 * costPer1M
}

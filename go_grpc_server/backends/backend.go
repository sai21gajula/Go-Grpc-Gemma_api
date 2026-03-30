// Package backends defines the LLMBackend interface and common types used by
// all model backends (Gemma/HuggingFace, Ollama/Qwen, Gemini).
package backends

import (
	"context"
	"time"
)

// GenerateResult holds the output of a single unary generation call.
type GenerateResult struct {
	Text      string
	TokenCount int
	LatencyMS  float64 // wall-clock ms
}

// StreamChunk is one token/chunk from a streaming generation call.
type StreamChunk struct {
	PartialText string
	Done        bool
	// TTFT is populated on the very first chunk (Done==false, PartialText!="").
	TTFT time.Duration
}

// LLMBackend is the interface every model backend must satisfy.
type LLMBackend interface {
	// Name returns the canonical backend name, e.g. "gemma_hf", "ollama", "gemini".
	Name() string

	// ModelID returns the full model identifier served by this backend.
	ModelID() string

	// Quantization describes how the model weights are stored/served.
	// Examples: "fp32", "int4_nf4", "q4_k_m", "none"
	Quantization() string

	// Generate performs a unary (non-streaming) inference call.
	Generate(ctx context.Context, prompt string, temperature float32, maxTokens int32) (*GenerateResult, error)

	// StreamGenerate performs a streaming inference call.
	// send is called for every chunk received from the backend.
	StreamGenerate(ctx context.Context, prompt string, temperature float32, maxTokens int32, send func(StreamChunk) error) error
}

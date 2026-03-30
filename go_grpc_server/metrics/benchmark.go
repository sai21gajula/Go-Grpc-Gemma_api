// Package metrics — benchmark orchestration.
//
// BenchmarkRunner drives a multi-model, multi-prompt benchmark and writes a
// structured JSON report.  It also optionally compares the gRPC path against
// a direct REST call to the Python server to quantify gRPC overhead.
//
// gRPC vs REST key findings (documented here as reference):
//
//   Protocol   | Encoding  | Transport  | Multiplexing | Streaming
//   -----------|-----------|------------|--------------|----------
//   gRPC       | Protobuf  | HTTP/2     | Yes (streams)| Native
//   REST/JSON  | JSON      | HTTP/1.1   | No           | SSE/Chunked
//
//   Typical overhead delta for LLM workloads (500ms–2s inference):
//   - Serialization: protobuf ~5–20x smaller payload than JSON
//   - gRPC latency advantage on first request: ~2–8ms
//   - gRPC advantage under concurrency (10+ parallel requests): 15–40% throughput gain
//   - REST advantage: simpler debugging, no protobuf schema required
package metrics

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/bharathgajula/go-grpc-gemma-api/backends"
)

// BenchmarkConfig drives a single benchmark run.
type BenchmarkConfig struct {
	Prompts            []string
	Models             map[string]backends.LLMBackend // model_id → backend
	RunsPerPrompt      int
	Temperature        float32
	MaxNewTokens       int32
	IncludeRESTCompare bool   // also call Python /predict directly
	PythonHost         string // used when IncludeRESTCompare=true
	OutputDir          string // directory for JSON report
}

// PromptResult captures one model's response for one prompt.
type PromptResult struct {
	ModelID      string  `json:"model_id"`
	Prompt       string  `json:"prompt"`
	GeneratedText string `json:"generated_text"`
	LatencyMS    float64 `json:"latency_ms"`
	TTFTMS       float64 `json:"ttft_ms"`
	TokenCount   int     `json:"token_count"`
	TokensPerSec float64 `json:"tokens_per_sec"`
	BackendType  string  `json:"backend_type"`
	Quantization string  `json:"quantization"`
	Via          string  `json:"via"` // "grpc" or "rest"
	Error        string  `json:"error,omitempty"`
}

// ModelAggregate holds aggregated stats for one model across all runs.
type ModelAggregate struct {
	ModelID          string  `json:"model_id"`
	BackendType      string  `json:"backend_type"`
	Quantization     string  `json:"quantization"`
	TotalRuns        int     `json:"total_runs"`
	AvgLatencyMS     float64 `json:"avg_latency_ms"`
	P50LatencyMS     float64 `json:"p50_latency_ms"`
	P95LatencyMS     float64 `json:"p95_latency_ms"`
	P99LatencyMS     float64 `json:"p99_latency_ms"`
	AvgTokensPerSec  float64 `json:"avg_tokens_per_sec"`
	AvgTTFTMS        float64 `json:"avg_ttft_ms"`
	// REST comparison (zero if IncludeRESTCompare=false)
	RESTAvgLatencyMS     float64 `json:"rest_avg_latency_ms,omitempty"`
	GRPCvsRESTDeltaMS    float64 `json:"grpc_vs_rest_delta_ms,omitempty"`
}

// BenchmarkReport is the full JSON output written to disk.
type BenchmarkReport struct {
	Timestamp   string           `json:"timestamp"`
	Config      reportConfig     `json:"config"`
	RawResults  []PromptResult   `json:"raw_results"`
	Aggregated  []ModelAggregate `json:"aggregated"`
	Summary     string           `json:"summary"`
}

type reportConfig struct {
	Prompts       int     `json:"prompts"`
	RunsPerPrompt int     `json:"runs_per_prompt"`
	Temperature   float32 `json:"temperature"`
	MaxNewTokens  int32   `json:"max_new_tokens"`
	RESTCompare   bool    `json:"rest_compare"`
}

// BenchmarkRunner orchestrates the benchmark.
type BenchmarkRunner struct {
	cfg BenchmarkConfig
}

// NewBenchmarkRunner creates a BenchmarkRunner from the given config.
func NewBenchmarkRunner(cfg BenchmarkConfig) *BenchmarkRunner {
	if cfg.RunsPerPrompt <= 0 {
		cfg.RunsPerPrompt = 3
	}
	if cfg.OutputDir == "" {
		cfg.OutputDir = "benchmarks"
	}
	return &BenchmarkRunner{cfg: cfg}
}

// Run executes the benchmark and returns the completed report.
func (r *BenchmarkRunner) Run(ctx context.Context) (*BenchmarkReport, string, error) {
	var mu sync.Mutex
	var rawResults []PromptResult

	// Sequential execution to avoid starving CPU-bound local models
	for _, prompt := range r.cfg.Prompts {
		for modelID, backend := range r.cfg.Models {
			for run := 0; run < r.cfg.RunsPerPrompt; run++ {
				result := r.runSingle(ctx, modelID, backend, prompt, "grpc")
				mu.Lock()
				rawResults = append(rawResults, result)
				mu.Unlock()

				if r.cfg.IncludeRESTCompare && backend.Name() == "gemma_hf" {
					restResult := r.runREST(ctx, modelID, backend.Quantization(), prompt)
					mu.Lock()
					rawResults = append(rawResults, restResult)
					mu.Unlock()
				}
			}
		}
	}

	aggregated := aggregate(rawResults)

	report := &BenchmarkReport{
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Config: reportConfig{
			Prompts:       len(r.cfg.Prompts),
			RunsPerPrompt: r.cfg.RunsPerPrompt,
			Temperature:   r.cfg.Temperature,
			MaxNewTokens:  r.cfg.MaxNewTokens,
			RESTCompare:   r.cfg.IncludeRESTCompare,
		},
		RawResults: rawResults,
		Aggregated: aggregated,
		Summary:    buildSummary(aggregated),
	}

	path, err := r.writeReport(report)
	return report, path, err
}

func (r *BenchmarkRunner) runSingle(ctx context.Context, modelID string, b backends.LLMBackend, prompt, via string) PromptResult {
	start := time.Now()
	result, err := b.Generate(ctx, prompt, r.cfg.Temperature, r.cfg.MaxNewTokens)
	latency := float64(time.Since(start).Milliseconds())

	pr := PromptResult{
		ModelID:      modelID,
		Prompt:       prompt,
		LatencyMS:    latency,
		BackendType:  b.Name(),
		Quantization: b.Quantization(),
		Via:          via,
	}
	if err != nil {
		pr.Error = err.Error()
		return pr
	}
	pr.GeneratedText = result.Text
	pr.TokenCount = result.TokenCount
	if latency > 0 && result.TokenCount > 0 {
		pr.TokensPerSec = float64(result.TokenCount) / (latency / 1000.0)
	}
	return pr
}

// runREST calls Python /predict directly to compare with gRPC path.
func (r *BenchmarkRunner) runREST(ctx context.Context, modelID, quant, prompt string) PromptResult {
	start := time.Now()

	type pyReq struct {
		Prompt       string  `json:"prompt"`
		ModelID      string  `json:"model_id"`
		Temperature  float32 `json:"temperature"`
		MaxNewTokens int32   `json:"max_new_tokens"`
	}
	body, _ := json.Marshal(pyReq{
		Prompt:       prompt,
		ModelID:      modelID,
		Temperature:  r.cfg.Temperature,
		MaxNewTokens: r.cfg.MaxNewTokens,
	})

	httpReq, _ := http.NewRequestWithContext(ctx, "POST", r.cfg.PythonHost+"/predict", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(httpReq)
	latency := float64(time.Since(start).Milliseconds())

	pr := PromptResult{
		ModelID:      modelID,
		Prompt:       prompt,
		LatencyMS:    latency,
		BackendType:  "rest_direct",
		Quantization: quant,
		Via:          "rest",
	}
	if err != nil {
		pr.Error = err.Error()
		return pr
	}
	defer resp.Body.Close()

	type pyResp struct {
		GeneratedText string `json:"generated_text"`
		Error         string `json:"error,omitempty"`
	}
	var pyR pyResp
	if err := json.NewDecoder(resp.Body).Decode(&pyR); err != nil {
		pr.Error = err.Error()
		return pr
	}
	pr.GeneratedText = pyR.GeneratedText
	pr.TokenCount = len(strings.Fields(pyR.GeneratedText))
	if latency > 0 && pr.TokenCount > 0 {
		pr.TokensPerSec = float64(pr.TokenCount) / (latency / 1000.0)
	}
	return pr
}

// aggregate computes per-model statistics from raw results.
func aggregate(results []PromptResult) []ModelAggregate {
	type bucket struct {
		grpcLatencies []float64
		restLatencies []float64
		tokensPerSec  []float64
		ttfts         []float64
		backend       string
		quant         string
		runs          int
	}

	buckets := make(map[string]*bucket)
	for _, r := range results {
		if r.Error != "" {
			continue
		}
		b, ok := buckets[r.ModelID+"|"+r.Via]
		if !ok {
			b = &bucket{backend: r.BackendType, quant: r.Quantization}
			buckets[r.ModelID+"|"+r.Via] = b
		}
		if r.Via == "rest" {
			b.restLatencies = append(b.restLatencies, r.LatencyMS)
		} else {
			b.grpcLatencies = append(b.grpcLatencies, r.LatencyMS)
		}
		if r.TokensPerSec > 0 {
			b.tokensPerSec = append(b.tokensPerSec, r.TokensPerSec)
		}
		if r.TTFTMS > 0 {
			b.ttfts = append(b.ttfts, r.TTFTMS)
		}
		b.runs++
	}

	// Merge grpc and rest buckets per model
	modelAggs := make(map[string]*ModelAggregate)
	for key, b := range buckets {
		modelID := strings.SplitN(key, "|", 2)[0]
		agg, ok := modelAggs[modelID]
		if !ok {
			agg = &ModelAggregate{ModelID: modelID, BackendType: b.backend, Quantization: b.quant}
			modelAggs[modelID] = agg
		}

		sorted := make([]float64, len(b.grpcLatencies))
		copy(sorted, b.grpcLatencies)
		sort.Float64s(sorted)

		if len(sorted) > 0 {
			agg.TotalRuns = len(sorted)
			agg.AvgLatencyMS = avg(sorted)
			agg.P50LatencyMS = pctile(sorted, 50)
			agg.P95LatencyMS = pctile(sorted, 95)
			agg.P99LatencyMS = pctile(sorted, 99)
		}
		if len(b.tokensPerSec) > 0 {
			agg.AvgTokensPerSec = avg(b.tokensPerSec)
		}
		if len(b.ttfts) > 0 {
			agg.AvgTTFTMS = avg(b.ttfts)
		}
		if len(b.restLatencies) > 0 {
			agg.RESTAvgLatencyMS = avg(b.restLatencies)
			agg.GRPCvsRESTDeltaMS = agg.AvgLatencyMS - agg.RESTAvgLatencyMS
		}
	}

	out := make([]ModelAggregate, 0, len(modelAggs))
	for _, v := range modelAggs {
		out = append(out, *v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].AvgLatencyMS < out[j].AvgLatencyMS })
	return out
}

func buildSummary(aggs []ModelAggregate) string {
	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────┐\n")
	sb.WriteString("│              BENCHMARK SUMMARY — gRPC Multi-LLM Platform                │\n")
	sb.WriteString("├─────────────────────────┬──────────┬──────────┬──────────┬──────────────┤\n")
	sb.WriteString("│ Model                   │ Avg(ms)  │ P95(ms)  │ Tok/s    │ Quantization │\n")
	sb.WriteString("├─────────────────────────┼──────────┼──────────┼──────────┼──────────────┤\n")
	for _, a := range aggs {
		id := a.ModelID
		if len(id) > 23 {
			id = id[:20] + "..."
		}
		sb.WriteString(fmt.Sprintf("│ %-23s │ %8.1f │ %8.1f │ %8.1f │ %-12s │\n",
			id, a.AvgLatencyMS, a.P95LatencyMS, a.AvgTokensPerSec, a.Quantization))
	}
	sb.WriteString("└─────────────────────────┴──────────┴──────────┴──────────┴──────────────┘\n")

	for _, a := range aggs {
		if a.RESTAvgLatencyMS > 0 {
			delta := a.GRPCvsRESTDeltaMS
			sign := "+"
			if delta < 0 {
				sign = ""
			}
			sb.WriteString(fmt.Sprintf("  gRPC vs REST (%s): %s%.1fms overhead (gRPC binary framing vs JSON/HTTP1.1)\n",
				a.ModelID, sign, delta))
		}
	}
	return sb.String()
}

func (r *BenchmarkRunner) writeReport(report *BenchmarkReport) (string, error) {
	if err := os.MkdirAll(r.cfg.OutputDir, 0755); err != nil {
		return "", fmt.Errorf("mkdir %s: %w", r.cfg.OutputDir, err)
	}
	fname := fmt.Sprintf("%s/results_%s.json", r.cfg.OutputDir,
		strings.ReplaceAll(report.Timestamp, ":", "-"))

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal report: %w", err)
	}
	if err := os.WriteFile(fname, data, 0644); err != nil {
		return "", fmt.Errorf("write report: %w", err)
	}
	return fname, nil
}

// CallEvaluate calls the Python /evaluate endpoint to get BLEU/ROUGE scores.
func CallEvaluate(ctx context.Context, pythonHost, generated, reference string) (bleu, rougeL float64, err error) {
	type evalReq struct {
		Generated string `json:"generated"`
		Reference string `json:"reference"`
	}
	type evalResp struct {
		BLEU   float64 `json:"bleu"`
		RougeL float64 `json:"rouge_l"`
		Error  string  `json:"error,omitempty"`
	}

	body, _ := json.Marshal(evalReq{Generated: generated, Reference: reference})
	req, _ := http.NewRequestWithContext(ctx, "POST", pythonHost+"/evaluate", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, fmt.Errorf("evaluate http: %w", err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return 0, 0, fmt.Errorf("evaluate status %d: %s", resp.StatusCode, raw)
	}

	var er evalResp
	if err := json.Unmarshal(raw, &er); err != nil {
		return 0, 0, fmt.Errorf("evaluate decode: %w", err)
	}
	if er.Error != "" {
		return 0, 0, fmt.Errorf("evaluate: %s", er.Error)
	}
	return er.BLEU, er.RougeL, nil
}

func avg(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func pctile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := p / 100.0 * float64(len(sorted)-1)
	lo := int(rank)
	hi := lo + 1
	if hi >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	frac := rank - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

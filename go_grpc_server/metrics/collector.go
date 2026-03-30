// Package metrics provides thread-safe latency/throughput collection and
// percentile reporting for LLM inference requests.
//
// Design goals:
//   - Zero-allocation fast path for recording (ring buffer, no heap per sample)
//   - Lock-free counters for throughput (atomic)
//   - Per-model isolation (map keyed by model_id)
package metrics

import (
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

const ringSize = 1024 // keep last N latency samples per model

// Sample is a single inference observation.
type Sample struct {
	ModelID    string
	LatencyMS  float64
	TTFTMS     float64 // time-to-first-token; 0 for unary calls
	TokenCount int
	Timestamp  time.Time
	BackendType string
	Via        string // "grpc" or "rest" (for comparison runs)
}

// Stats summarises the collected samples for one model.
type Stats struct {
	ModelID       string
	BackendType   string
	TotalRequests int64
	TotalTokens   int64
	AvgLatencyMS  float64
	P50LatencyMS  float64
	P95LatencyMS  float64
	P99LatencyMS  float64
	AvgTTFTMS     float64
	AvgTokensPerSec float64
}

// modelRing is a fixed-size circular buffer of latency samples.
type modelRing struct {
	mu       sync.Mutex
	buf      [ringSize]Sample
	head     int
	count    int
	total    atomic.Int64 // request counter
	tokens   atomic.Int64 // cumulative token counter
	backend  string
}

func (r *modelRing) record(s Sample) {
	r.mu.Lock()
	r.buf[r.head] = s
	r.head = (r.head + 1) % ringSize
	if r.count < ringSize {
		r.count++
	}
	r.mu.Unlock()
	r.total.Add(1)
	r.tokens.Add(int64(s.TokenCount))
}

func (r *modelRing) stats(modelID string) Stats {
	r.mu.Lock()
	n := r.count
	samples := make([]Sample, n)
	for i := 0; i < n; i++ {
		idx := (r.head - n + i + ringSize) % ringSize
		samples[i] = r.buf[idx]
	}
	r.mu.Unlock()

	if n == 0 {
		return Stats{ModelID: modelID, BackendType: r.backend, TotalRequests: r.total.Load(), TotalTokens: r.tokens.Load()}
	}

	latencies := make([]float64, n)
	ttfts := make([]float64, 0, n)
	sumLatency := 0.0
	sumTTFT := 0.0
	sumTokensPerSec := 0.0

	for i, s := range samples {
		latencies[i] = s.LatencyMS
		sumLatency += s.LatencyMS
		if s.TTFTMS > 0 {
			ttfts = append(ttfts, s.TTFTMS)
			sumTTFT += s.TTFTMS
		}
		if s.LatencyMS > 0 && s.TokenCount > 0 {
			sumTokensPerSec += float64(s.TokenCount) / (s.LatencyMS / 1000.0)
		}
	}

	sort.Float64s(latencies)

	avgTTFT := 0.0
	if len(ttfts) > 0 {
		avgTTFT = sumTTFT / float64(len(ttfts))
	}

	return Stats{
		ModelID:         modelID,
		BackendType:     r.backend,
		TotalRequests:   r.total.Load(),
		TotalTokens:     r.tokens.Load(),
		AvgLatencyMS:    sumLatency / float64(n),
		P50LatencyMS:    percentile(latencies, 50),
		P95LatencyMS:    percentile(latencies, 95),
		P99LatencyMS:    percentile(latencies, 99),
		AvgTTFTMS:       avgTTFT,
		AvgTokensPerSec: sumTokensPerSec / float64(n),
	}
}

// Collector is the top-level per-model metrics store.
type Collector struct {
	mu     sync.RWMutex
	models map[string]*modelRing
}

// NewCollector creates a new Collector.
func NewCollector() *Collector {
	return &Collector{models: make(map[string]*modelRing)}
}

// Record adds a sample for the given model.
func (c *Collector) Record(s Sample) {
	c.mu.RLock()
	ring, ok := c.models[s.ModelID]
	c.mu.RUnlock()

	if !ok {
		c.mu.Lock()
		if ring, ok = c.models[s.ModelID]; !ok {
			ring = &modelRing{backend: s.BackendType}
			c.models[s.ModelID] = ring
		}
		c.mu.Unlock()
	}
	ring.record(s)
}

// GetStats returns current statistics for a specific model.
func (c *Collector) GetStats(modelID string) Stats {
	c.mu.RLock()
	ring, ok := c.models[modelID]
	c.mu.RUnlock()
	if !ok {
		return Stats{ModelID: modelID}
	}
	return ring.stats(modelID)
}

// AllStats returns statistics for every model seen so far.
func (c *Collector) AllStats() []Stats {
	c.mu.RLock()
	keys := make([]string, 0, len(c.models))
	for k := range c.models {
		keys = append(keys, k)
	}
	c.mu.RUnlock()

	out := make([]Stats, 0, len(keys))
	for _, k := range keys {
		out = append(out, c.GetStats(k))
	}
	return out
}

// percentile returns the p-th percentile value from a sorted float64 slice.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := p / 100.0 * float64(len(sorted)-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return sorted[lo]
	}
	frac := rank - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

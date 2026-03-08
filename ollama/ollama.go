// Package ollama provides convenience functions to use the Anthropic SDK
// with a local Ollama instance.
//
// Ollama exposes an OpenAI-compatible API at http://localhost:11434.
// This package wraps the openai middleware with Ollama-specific defaults.
//
// Usage:
//
//	client := anthropic.NewClient(
//	    ollama.New("llama3"),
//	)
//
//	// Or with a custom host:
//	client := anthropic.NewClient(
//	    ollama.NewWithBaseURL("http://myhost:11434", "llama3"),
//	)
package ollama

import (
	"github.com/anthropics/anthropic-sdk-go/openai"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// DefaultBaseURL is the default Ollama API endpoint.
// Note: the SDK appends "v1/messages" to the base URL, so this should NOT
// include a "/v1" suffix.
const DefaultBaseURL = "http://localhost:11434"

// New returns a request option that configures the client to use a local
// Ollama instance at the default address (localhost:11434) with the
// specified model.
func New(model string) option.RequestOption {
	return openai.WithBaseURL(DefaultBaseURL, model)
}

// NewWithBaseURL returns a request option that configures the client to use
// an Ollama instance at the specified base URL with the given model.
func NewWithBaseURL(baseURL string, model string) option.RequestOption {
	return openai.WithBaseURL(baseURL, model)
}

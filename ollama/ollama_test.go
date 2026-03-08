package ollama

import (
	"testing"
)

func TestDefaultBaseURL(t *testing.T) {
	if DefaultBaseURL != "http://localhost:11434/v1" {
		t.Errorf("expected default base URL http://localhost:11434/v1, got %s", DefaultBaseURL)
	}
}

func TestNewReturnsNonNil(t *testing.T) {
	opt := New("llama3")
	if opt == nil {
		t.Fatal("expected non-nil request option")
	}
}

func TestNewWithBaseURLReturnsNonNil(t *testing.T) {
	opt := NewWithBaseURL("http://myhost:11434/v1", "llama3")
	if opt == nil {
		t.Fatal("expected non-nil request option")
	}
}

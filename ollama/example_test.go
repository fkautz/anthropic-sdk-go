//go:build integration

package ollama_test

import (
	"context"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/ollama"
)

func TestOllamaIntegration(t *testing.T) {
	client := anthropic.NewClient(ollama.New("gemma3:1b"))

	message, err := client.Messages.New(context.Background(), anthropic.MessageNewParams{
		Model:     "gemma3:1b",
		MaxTokens: 64,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Say hello in one sentence.")),
		},
	})
	if err != nil {
		t.Fatalf("Messages.New failed: %v", err)
	}

	if len(message.Content) == 0 {
		t.Fatal("expected at least one content block")
	}

	t.Logf("Model: %s", message.Model)
	t.Logf("Stop reason: %s", message.StopReason)
	t.Logf("Response: %s", message.Content[0].Text)
	t.Logf("Usage: input=%d output=%d", message.Usage.InputTokens, message.Usage.OutputTokens)
}

package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
)

func TestTranslateRequestBasic(t *testing.T) {
	anthropicReq := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"temperature": 0.7,
		"top_p": 0.9,
		"stop_sequences": ["END"],
		"system": [{"type": "text", "text": "You are helpful."}],
		"messages": [
			{"role": "user", "content": "Hello"}
		]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	if err := json.Unmarshal(result, &req); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}

	if req.Model != "claude-sonnet-4-20250514" {
		t.Errorf("expected model claude-sonnet-4-20250514, got %s", req.Model)
	}
	if req.MaxTokens == nil || *req.MaxTokens != 1024 {
		t.Errorf("expected max_tokens 1024, got %v", req.MaxTokens)
	}
	if req.Temperature == nil || *req.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %v", req.Temperature)
	}
	if req.TopP == nil || *req.TopP != 0.9 {
		t.Errorf("expected top_p 0.9, got %v", req.TopP)
	}
	if len(req.Stop) != 1 || req.Stop[0] != "END" {
		t.Errorf("expected stop [END], got %v", req.Stop)
	}

	// system message + user message = 2
	if len(req.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(req.Messages))
	}

	// First should be system
	if req.Messages[0].Role != "system" {
		t.Errorf("expected first message role system, got %s", req.Messages[0].Role)
	}
	var sysContent string
	json.Unmarshal(req.Messages[0].Content, &sysContent)
	if sysContent != "You are helpful." {
		t.Errorf("expected system content 'You are helpful.', got %s", sysContent)
	}

	// Second should be user
	if req.Messages[1].Role != "user" {
		t.Errorf("expected second message role user, got %s", req.Messages[1].Role)
	}
}

func TestTranslateRequestDefaultModel(t *testing.T) {
	anthropicReq := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "llama3")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if req.Model != "llama3" {
		t.Errorf("expected model llama3, got %s", req.Model)
	}
}

func TestTranslateRequestSystemString(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"system": "Be concise.",
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if len(req.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(req.Messages))
	}

	var sysContent string
	json.Unmarshal(req.Messages[0].Content, &sysContent)
	if sysContent != "Be concise." {
		t.Errorf("expected 'Be concise.', got %s", sysContent)
	}
}

func TestTranslateRequestToolUse(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "What is the weather?"},
			{"role": "assistant", "content": [
				{"type": "text", "text": "Let me check."},
				{"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"city": "NYC"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "call_123", "content": "72°F sunny"}
			]}
		],
		"tools": [
			{"type": "custom", "name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
		]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	if err := json.Unmarshal(result, &req); err != nil {
		t.Fatalf("failed to parse result: %v", err)
	}

	// user + assistant (with tool_calls) + tool = 3
	if len(req.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(req.Messages))
	}

	// Assistant message should have tool_calls
	assistantMsg := req.Messages[1]
	if assistantMsg.Role != "assistant" {
		t.Errorf("expected assistant role, got %s", assistantMsg.Role)
	}
	if len(assistantMsg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(assistantMsg.ToolCalls))
	}
	if assistantMsg.ToolCalls[0].ID != "call_123" {
		t.Errorf("expected tool call id call_123, got %s", assistantMsg.ToolCalls[0].ID)
	}
	if assistantMsg.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("expected function name get_weather, got %s", assistantMsg.ToolCalls[0].Function.Name)
	}

	// Tool result should be role=tool
	toolMsg := req.Messages[2]
	if toolMsg.Role != "tool" {
		t.Errorf("expected tool role, got %s", toolMsg.Role)
	}
	if toolMsg.ToolCallID != "call_123" {
		t.Errorf("expected tool_call_id call_123, got %s", toolMsg.ToolCallID)
	}

	// Tools
	if len(req.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(req.Tools))
	}
	if req.Tools[0].Function.Name != "get_weather" {
		t.Errorf("expected tool name get_weather, got %s", req.Tools[0].Function.Name)
	}
}

func TestTranslateRequestStream(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"stream": true,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if !req.Stream {
		t.Error("expected stream=true")
	}
	if req.StreamOptions == nil || !req.StreamOptions.IncludeUsage {
		t.Error("expected stream_options.include_usage=true")
	}
}

func TestTranslateResponseBasic(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-123",
		"object": "chat.completion",
		"model": "llama3",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Hello! How can I help?"
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 8,
			"total_tokens": 18
		}
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	if res["id"] != "chatcmpl-123" {
		t.Errorf("expected id chatcmpl-123, got %v", res["id"])
	}
	if res["type"] != "message" {
		t.Errorf("expected type message, got %v", res["type"])
	}
	if res["role"] != "assistant" {
		t.Errorf("expected role assistant, got %v", res["role"])
	}
	if res["model"] != "llama3" {
		t.Errorf("expected model llama3, got %v", res["model"])
	}
	if res["stop_reason"] != "end_turn" {
		t.Errorf("expected stop_reason end_turn, got %v", res["stop_reason"])
	}

	content := res["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(content))
	}
	block := content[0].(map[string]any)
	if block["type"] != "text" {
		t.Errorf("expected type text, got %v", block["type"])
	}
	if block["text"] != "Hello! How can I help?" {
		t.Errorf("expected text content, got %v", block["text"])
	}

	usage := res["usage"].(map[string]any)
	if usage["input_tokens"].(float64) != 10 {
		t.Errorf("expected input_tokens 10, got %v", usage["input_tokens"])
	}
	if usage["output_tokens"].(float64) != 8 {
		t.Errorf("expected output_tokens 8, got %v", usage["output_tokens"])
	}
}

func TestTranslateResponseToolCalls(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-456",
		"object": "chat.completion",
		"model": "gpt-4",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"tool_calls": [{
					"id": "call_abc",
					"type": "function",
					"function": {
						"name": "get_weather",
						"arguments": "{\"city\":\"NYC\"}"
					}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	if res["stop_reason"] != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %v", res["stop_reason"])
	}

	content := res["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(content))
	}
	block := content[0].(map[string]any)
	if block["type"] != "tool_use" {
		t.Errorf("expected type tool_use, got %v", block["type"])
	}
	if block["id"] != "call_abc" {
		t.Errorf("expected id call_abc, got %v", block["id"])
	}
	if block["name"] != "get_weather" {
		t.Errorf("expected name get_weather, got %v", block["name"])
	}
}

func TestTranslateResponseMaxTokens(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-789",
		"object": "chat.completion",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "partial resp"},
			"finish_reason": "length"
		}]
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	if res["stop_reason"] != "max_tokens" {
		t.Errorf("expected stop_reason max_tokens, got %v", res["stop_reason"])
	}
}

func TestTranslateResponseContentFilter(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-000",
		"object": "chat.completion",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": ""},
			"finish_reason": "content_filter"
		}]
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	if res["stop_reason"] != "refusal" {
		t.Errorf("expected stop_reason refusal, got %v", res["stop_reason"])
	}
}

func TestMapFinishReason(t *testing.T) {
	cases := []struct {
		input    string
		expected string
	}{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
		{"function_call", "tool_use"},
		{"content_filter", "refusal"},
		{"unknown", "end_turn"},
		{"", "end_turn"},
	}

	for _, tc := range cases {
		got := mapFinishReason(tc.input)
		if got != tc.expected {
			t.Errorf("mapFinishReason(%q) = %q, want %q", tc.input, got, tc.expected)
		}
	}
}

func TestMiddlewarePathRewrite(t *testing.T) {
	middleware := openaiMiddleware("")

	body := `{
		"model": "llama3",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	req, _ := http.NewRequest("POST", "http://localhost:11434/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("X-Api-Key", "test-key")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("expected path /v1/chat/completions, got %s", r.URL.Path)
		}
		if r.Header.Get("anthropic-version") != "" {
			t.Error("expected anthropic-version header to be removed")
		}
		if r.Header.Get("X-Api-Key") != "" {
			t.Error("expected X-Api-Key header to be removed")
		}

		// Verify the body is valid OpenAI format
		resBody, _ := io.ReadAll(r.Body)
		var oaiReq oaiRequest
		if err := json.Unmarshal(resBody, &oaiReq); err != nil {
			t.Fatalf("body is not valid OpenAI request: %v", err)
		}
		if oaiReq.Model != "llama3" {
			t.Errorf("expected model llama3, got %s", oaiReq.Model)
		}

		// Return a mock OpenAI response
		oaiRes := `{"id":"chatcmpl-test","object":"chat.completion","model":"llama3","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(bytes.NewReader([]byte(oaiRes))),
		}, nil
	})

	if err != nil {
		t.Fatalf("middleware failed: %v", err)
	}
}

func TestMiddlewareNonMessagesPassthrough(t *testing.T) {
	middleware := openaiMiddleware("")

	body := `{"model": "test"}`
	req, _ := http.NewRequest("POST", "http://localhost/v1/models", bytes.NewReader([]byte(body)))

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		// Path should NOT be rewritten
		if r.URL.Path != "/v1/models" {
			t.Errorf("expected path /v1/models, got %s", r.URL.Path)
		}
		return &http.Response{
			StatusCode: 200,
			Body:       http.NoBody,
		}, nil
	})

	if err != nil {
		t.Fatalf("middleware failed: %v", err)
	}
}

func TestMiddlewareResponseTranslation(t *testing.T) {
	middleware := openaiMiddleware("")

	body := `{
		"model": "llama3",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	req, _ := http.NewRequest("POST", "http://localhost/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	res, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		oaiRes := `{"id":"chatcmpl-test","object":"chat.completion","model":"llama3","choices":[{"index":0,"message":{"role":"assistant","content":"Hi there!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(bytes.NewReader([]byte(oaiRes))),
		}, nil
	})

	if err != nil {
		t.Fatalf("middleware failed: %v", err)
	}

	resBody, _ := io.ReadAll(res.Body)
	var anthropicRes map[string]any
	if err := json.Unmarshal(resBody, &anthropicRes); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if anthropicRes["type"] != "message" {
		t.Errorf("expected type message, got %v", anthropicRes["type"])
	}
	if anthropicRes["role"] != "assistant" {
		t.Errorf("expected role assistant, got %v", anthropicRes["role"])
	}

	content := anthropicRes["content"].([]any)
	block := content[0].(map[string]any)
	if block["text"] != "Hi there!" {
		t.Errorf("expected 'Hi there!', got %v", block["text"])
	}
}

func TestTranslateRequestMultiBlockContent(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "First part."},
				{"type": "text", "text": "Second part."}
			]
		}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(req.Messages))
	}

	var content string
	json.Unmarshal(req.Messages[0].Content, &content)
	if content != "First part.\nSecond part." {
		t.Errorf("expected concatenated text, got %q", content)
	}
}

func TestTranslateRequestToolResultWithTextBlocks(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "tool_result", "tool_use_id": "call_1", "content": [
					{"type": "text", "text": "Result line 1"},
					{"type": "text", "text": "Result line 2"}
				]}
			]
		}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(req.Messages))
	}

	if req.Messages[0].Role != "tool" {
		t.Errorf("expected role tool, got %s", req.Messages[0].Role)
	}

	var content string
	json.Unmarshal(req.Messages[0].Content, &content)
	if content != "Result line 1\nResult line 2" {
		t.Errorf("expected concatenated result, got %q", content)
	}
}

func TestTranslateResponseEmptyContent(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-empty",
		"object": "chat.completion",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": ""},
			"finish_reason": "stop"
		}]
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	// Should have a text block even for empty content
	content := res["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content block for empty response, got %d", len(content))
	}
	block := content[0].(map[string]any)
	if block["type"] != "text" {
		t.Errorf("expected type text, got %v", block["type"])
	}
}

func TestStreamTranslatorBasic(t *testing.T) {
	// Simulate OpenAI streaming response
	chunks := []string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"llama3","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"llama3","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"llama3","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"llama3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		``,
		`data: [DONE]`,
		``,
	}

	body := io.NopCloser(bytes.NewReader([]byte(fmt.Sprintf("%s\n", joinLines(chunks)))))
	translator := &streamTranslator{rc: body}

	output, err := io.ReadAll(translator)
	if err != nil {
		t.Fatalf("failed to read stream: %v", err)
	}

	outputStr := string(output)

	// Should contain Anthropic event types
	if !containsSubstring(outputStr, "message_start") {
		t.Error("expected message_start event")
	}
	if !containsSubstring(outputStr, "content_block_delta") {
		t.Error("expected content_block_delta event")
	}
	if !containsSubstring(outputStr, "text_delta") {
		t.Error("expected text_delta in content")
	}
	if !containsSubstring(outputStr, "Hello") {
		t.Error("expected 'Hello' in stream output")
	}
	if !containsSubstring(outputStr, " world") {
		t.Error("expected ' world' in stream output")
	}
	if !containsSubstring(outputStr, "message_stop") {
		t.Error("expected message_stop event")
	}
}

func joinLines(lines []string) string {
	result := ""
	for _, l := range lines {
		result += l + "\n"
	}
	return result
}

func containsSubstring(s, substr string) bool {
	return bytes.Contains([]byte(s), []byte(substr))
}

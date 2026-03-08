package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// ---------------------------------------------------------------------------
// Request translation tests
// ---------------------------------------------------------------------------

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

	if req.Messages[0].Role != "system" {
		t.Errorf("expected first message role system, got %s", req.Messages[0].Role)
	}
	var sysContent string
	json.Unmarshal(req.Messages[0].Content, &sysContent)
	if sysContent != "You are helpful." {
		t.Errorf("expected system content 'You are helpful.', got %s", sysContent)
	}

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

func TestTranslateRequestEmptyDefaultModel(t *testing.T) {
	anthropicReq := `{
		"model": "my-model",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if req.Model != "my-model" {
		t.Errorf("expected model my-model, got %s", req.Model)
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

func TestTranslateRequestSystemMultiBlock(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"system": [
			{"type": "text", "text": "You are a helpful assistant."},
			{"type": "text", "text": "Be concise."}
		],
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	var sysContent string
	json.Unmarshal(req.Messages[0].Content, &sysContent)
	if sysContent != "You are a helpful assistant.\nBe concise." {
		t.Errorf("expected concatenated system text, got %q", sysContent)
	}
}

func TestTranslateRequestNoSystem(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message (no system), got %d", len(req.Messages))
	}
	if req.Messages[0].Role != "user" {
		t.Errorf("expected user role, got %s", req.Messages[0].Role)
	}
}

func TestTranslateRequestNoOptionalParams(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if req.Temperature != nil {
		t.Errorf("expected nil temperature, got %v", *req.Temperature)
	}
	if req.TopP != nil {
		t.Errorf("expected nil top_p, got %v", *req.TopP)
	}
	if len(req.Stop) != 0 {
		t.Errorf("expected no stop sequences, got %v", req.Stop)
	}
	if len(req.Tools) != 0 {
		t.Errorf("expected no tools, got %d", len(req.Tools))
	}
}

func TestTranslateRequestMultipleStopSequences(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"stop_sequences": ["END", "STOP", "---"],
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if len(req.Stop) != 3 {
		t.Fatalf("expected 3 stop sequences, got %d", len(req.Stop))
	}
	expected := []string{"END", "STOP", "---"}
	for i, s := range expected {
		if req.Stop[i] != s {
			t.Errorf("stop[%d]: expected %q, got %q", i, s, req.Stop[i])
		}
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

	// Verify assistant text content
	var assistantContent string
	json.Unmarshal(assistantMsg.Content, &assistantContent)
	if assistantContent != "Let me check." {
		t.Errorf("expected assistant text 'Let me check.', got %q", assistantContent)
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

func TestTranslateRequestMultipleToolCalls(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "Weather in NYC and LA?"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "NYC"}},
				{"type": "tool_use", "id": "call_2", "name": "get_weather", "input": {"city": "LA"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "call_1", "content": "72°F"},
				{"type": "tool_result", "tool_use_id": "call_2", "content": "85°F"}
			]}
		]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	// user + assistant (2 tool_calls) + tool + tool = 4
	if len(req.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(req.Messages))
	}

	// Assistant should have 2 tool calls
	if len(req.Messages[1].ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(req.Messages[1].ToolCalls))
	}
	if req.Messages[1].ToolCalls[0].ID != "call_1" {
		t.Errorf("expected first tool call id call_1, got %s", req.Messages[1].ToolCalls[0].ID)
	}
	if req.Messages[1].ToolCalls[1].ID != "call_2" {
		t.Errorf("expected second tool call id call_2, got %s", req.Messages[1].ToolCalls[1].ID)
	}

	// Two separate tool messages
	if req.Messages[2].Role != "tool" || req.Messages[2].ToolCallID != "call_1" {
		t.Errorf("expected tool message for call_1, got role=%s id=%s", req.Messages[2].Role, req.Messages[2].ToolCallID)
	}
	if req.Messages[3].Role != "tool" || req.Messages[3].ToolCallID != "call_2" {
		t.Errorf("expected tool message for call_2, got role=%s id=%s", req.Messages[3].Role, req.Messages[3].ToolCallID)
	}
}

func TestTranslateRequestToolUseOnlyNoText(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "Do something"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "call_1", "name": "do_thing", "input": {}}
			]}
		]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	assistantMsg := req.Messages[1]

	// Content should be null when no text
	var content *string
	if err := json.Unmarshal(assistantMsg.Content, &content); err == nil && content != nil {
		t.Errorf("expected null content for tool-only assistant message, got %q", *content)
	}

	if len(assistantMsg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(assistantMsg.ToolCalls))
	}
}

func TestTranslateRequestToolResultStringContent(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "tool_result", "tool_use_id": "call_1", "content": "plain string result"}
			]
		}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if req.Messages[0].Role != "tool" {
		t.Errorf("expected role tool, got %s", req.Messages[0].Role)
	}

	var content string
	json.Unmarshal(req.Messages[0].Content, &content)
	if content != "plain string result" {
		t.Errorf("expected 'plain string result', got %q", content)
	}
}

func TestTranslateRequestToolResultWithTextAndToolResult(t *testing.T) {
	// User message with both text and tool_result blocks
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "Here are the results:"},
				{"type": "tool_result", "tool_use_id": "call_1", "content": "42"}
			]
		}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	// Should split into user text message + tool message
	if len(req.Messages) != 2 {
		t.Fatalf("expected 2 messages (user + tool), got %d", len(req.Messages))
	}

	if req.Messages[0].Role != "user" {
		t.Errorf("expected first message role user, got %s", req.Messages[0].Role)
	}
	if req.Messages[1].Role != "tool" {
		t.Errorf("expected second message role tool, got %s", req.Messages[1].Role)
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

func TestTranslateRequestStreamFalse(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"stream": false,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if req.Stream {
		t.Error("expected stream=false")
	}
	if req.StreamOptions != nil {
		t.Error("expected nil stream_options when not streaming")
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

func TestTranslateRequestMultiTurnConversation(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"system": "You are helpful.",
		"messages": [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi there!"},
			{"role": "user", "content": "How are you?"},
			{"role": "assistant", "content": "I'm doing well."},
			{"role": "user", "content": "Great!"}
		]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	// system + 5 messages = 6
	if len(req.Messages) != 6 {
		t.Fatalf("expected 6 messages, got %d", len(req.Messages))
	}

	expectedRoles := []string{"system", "user", "assistant", "user", "assistant", "user"}
	for i, role := range expectedRoles {
		if req.Messages[i].Role != role {
			t.Errorf("message[%d]: expected role %s, got %s", i, role, req.Messages[i].Role)
		}
	}
}

func TestTranslateRequestToolEmptyInput(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "Do it"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "call_1", "name": "no_args_tool", "input": {}}
			]}
		]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if len(req.Messages[1].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(req.Messages[1].ToolCalls))
	}
	args := req.Messages[1].ToolCalls[0].Function.Arguments
	if args != "{}" {
		t.Errorf("expected empty object arguments, got %q", args)
	}
}

func TestTranslateRequestMultipleTools(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}],
		"tools": [
			{"name": "tool_a", "description": "First tool", "input_schema": {"type": "object"}},
			{"name": "tool_b", "description": "Second tool", "input_schema": {"type": "object"}},
			{"name": "tool_c", "description": "Third tool", "input_schema": {"type": "object"}}
		]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	if len(req.Tools) != 3 {
		t.Fatalf("expected 3 tools, got %d", len(req.Tools))
	}
	names := []string{"tool_a", "tool_b", "tool_c"}
	for i, name := range names {
		if req.Tools[i].Function.Name != name {
			t.Errorf("tool[%d]: expected name %s, got %s", i, name, req.Tools[i].Function.Name)
		}
		if req.Tools[i].Type != "function" {
			t.Errorf("tool[%d]: expected type function, got %s", i, req.Tools[i].Type)
		}
	}
}

func TestTranslateRequestServerToolSkipped(t *testing.T) {
	anthropicReq := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}],
		"tools": [
			{"type": "web_search"},
			{"name": "my_tool", "description": "Custom tool", "input_schema": {"type": "object"}}
		]
	}`

	result, err := translateRequest([]byte(anthropicReq), "")
	if err != nil {
		t.Fatalf("translateRequest failed: %v", err)
	}

	var req oaiRequest
	json.Unmarshal(result, &req)

	// web_search should be skipped (no name), only my_tool should remain
	if len(req.Tools) != 1 {
		t.Fatalf("expected 1 tool (server tool skipped), got %d", len(req.Tools))
	}
	if req.Tools[0].Function.Name != "my_tool" {
		t.Errorf("expected my_tool, got %s", req.Tools[0].Function.Name)
	}
}

func TestTranslateRequestToolNoName(t *testing.T) {
	tool := translateTool(gjson.Parse(`{"type": "custom"}`))
	if tool != nil {
		t.Error("expected nil for tool with no name")
	}
}

// ---------------------------------------------------------------------------
// Response translation tests
// ---------------------------------------------------------------------------

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

func TestTranslateResponseMultipleToolCalls(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-multi",
		"object": "chat.completion",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": "{\"x\":1}"}},
					{"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": "{\"y\":2}"}}
				]
			},
			"finish_reason": "tool_calls"
		}]
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	content := res["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(content))
	}

	block0 := content[0].(map[string]any)
	block1 := content[1].(map[string]any)

	if block0["type"] != "tool_use" || block0["id"] != "call_1" {
		t.Errorf("expected tool_use call_1, got type=%v id=%v", block0["type"], block0["id"])
	}
	if block1["type"] != "tool_use" || block1["id"] != "call_2" {
		t.Errorf("expected tool_use call_2, got type=%v id=%v", block1["type"], block1["id"])
	}
}

func TestTranslateResponseTextPlusToolCalls(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-mixed",
		"object": "chat.completion",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Let me check that for you.",
				"tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{\"q\":\"test\"}"}}
				]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	content := res["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected 2 content blocks (text + tool_use), got %d", len(content))
	}

	textBlock := content[0].(map[string]any)
	if textBlock["type"] != "text" || textBlock["text"] != "Let me check that for you." {
		t.Errorf("expected text block, got %v", textBlock)
	}

	toolBlock := content[1].(map[string]any)
	if toolBlock["type"] != "tool_use" || toolBlock["id"] != "call_1" {
		t.Errorf("expected tool_use block, got %v", toolBlock)
	}
}

func TestTranslateResponseEmptyToolArguments(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-empty-args",
		"object": "chat.completion",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "no_args", "arguments": ""}}
				]
			},
			"finish_reason": "tool_calls"
		}]
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	content := res["content"].([]any)
	block := content[0].(map[string]any)
	// Empty arguments should default to {} (deserialized as empty map)
	inputMap, ok := block["input"].(map[string]any)
	if !ok {
		t.Fatalf("expected input to be a map, got %T", block["input"])
	}
	if len(inputMap) != 0 {
		t.Errorf("expected empty object input for empty arguments, got %v", inputMap)
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

	content := res["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content block for empty response, got %d", len(content))
	}
	block := content[0].(map[string]any)
	if block["type"] != "text" {
		t.Errorf("expected type text, got %v", block["type"])
	}
}

func TestTranslateResponseNullContent(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-null",
		"object": "chat.completion",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": null},
			"finish_reason": "stop"
		}]
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	// Should still have a content block even with null content
	content := res["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 content block for null content, got %d", len(content))
	}
}

func TestTranslateResponseEmptyChoices(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-empty-choices",
		"object": "chat.completion",
		"model": "test",
		"choices": []
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	// Should have a fallback empty text block
	content := res["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 fallback content block, got %d", len(content))
	}
	if res["stop_reason"] != "end_turn" {
		t.Errorf("expected stop_reason end_turn for empty choices, got %v", res["stop_reason"])
	}
}

func TestTranslateResponseNullUsage(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-no-usage",
		"object": "chat.completion",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Hello"},
			"finish_reason": "stop"
		}],
		"usage": null
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	usage := res["usage"].(map[string]any)
	if usage["input_tokens"].(float64) != 0 {
		t.Errorf("expected input_tokens 0 for null usage, got %v", usage["input_tokens"])
	}
	if usage["output_tokens"].(float64) != 0 {
		t.Errorf("expected output_tokens 0 for null usage, got %v", usage["output_tokens"])
	}
}

func TestTranslateResponseMissingUsage(t *testing.T) {
	oaiRes := `{
		"id": "chatcmpl-missing-usage",
		"object": "chat.completion",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Hello"},
			"finish_reason": "stop"
		}]
	}`

	result, err := translateResponse([]byte(oaiRes))
	if err != nil {
		t.Fatalf("translateResponse failed: %v", err)
	}

	var res map[string]any
	json.Unmarshal(result, &res)

	usage := res["usage"].(map[string]any)
	if usage["input_tokens"].(float64) != 0 {
		t.Errorf("expected input_tokens 0 for missing usage, got %v", usage["input_tokens"])
	}
}

func TestTranslateResponseInvalidJSON(t *testing.T) {
	_, err := translateResponse([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON response")
	}
}

// ---------------------------------------------------------------------------
// mapFinishReason tests
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Middleware tests
// ---------------------------------------------------------------------------

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

		resBody, _ := io.ReadAll(r.Body)
		var oaiReq oaiRequest
		if err := json.Unmarshal(resBody, &oaiReq); err != nil {
			t.Fatalf("body is not valid OpenAI request: %v", err)
		}
		if oaiReq.Model != "llama3" {
			t.Errorf("expected model llama3, got %s", oaiReq.Model)
		}

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

func TestMiddlewareGETPassthrough(t *testing.T) {
	middleware := openaiMiddleware("")

	req, _ := http.NewRequest("GET", "http://localhost/v1/models", nil)

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("expected path /v1/models unchanged, got %s", r.URL.Path)
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

func TestMiddlewareHTTPErrorPassthrough(t *testing.T) {
	middleware := openaiMiddleware("")

	body := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	req, _ := http.NewRequest("POST", "http://localhost/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	errorBody := `{"error": {"message": "rate limited", "type": "rate_limit_error"}}`
	res, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 429,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(bytes.NewReader([]byte(errorBody))),
		}, nil
	})

	if err != nil {
		t.Fatalf("middleware failed: %v", err)
	}

	// Error responses should pass through untranslated
	if res.StatusCode != 429 {
		t.Errorf("expected status 429, got %d", res.StatusCode)
	}

	resBody, _ := io.ReadAll(res.Body)
	if !strings.Contains(string(resBody), "rate limited") {
		t.Error("expected error body to pass through unchanged")
	}
}

func TestMiddleware500ErrorPassthrough(t *testing.T) {
	middleware := openaiMiddleware("")

	body := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	req, _ := http.NewRequest("POST", "http://localhost/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	res, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 500,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(bytes.NewReader([]byte(`{"error": "internal"}`))),
		}, nil
	})

	if err != nil {
		t.Fatalf("middleware failed: %v", err)
	}

	if res.StatusCode != 500 {
		t.Errorf("expected status 500, got %d", res.StatusCode)
	}
}

func TestMiddlewareNetworkError(t *testing.T) {
	middleware := openaiMiddleware("")

	body := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	req, _ := http.NewRequest("POST", "http://localhost/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		return nil, fmt.Errorf("connection refused")
	})

	if err == nil {
		t.Fatal("expected error for network failure")
	}
	if !strings.Contains(err.Error(), "connection refused") {
		t.Errorf("expected connection refused error, got %v", err)
	}
}

func TestMiddlewareStreamingResponseWrap(t *testing.T) {
	middleware := openaiMiddleware("")

	body := `{
		"model": "test",
		"max_tokens": 100,
		"stream": true,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	req, _ := http.NewRequest("POST", "http://localhost/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	sseBody := "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n"

	res, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(bytes.NewReader([]byte(sseBody))),
		}, nil
	})

	if err != nil {
		t.Fatalf("middleware failed: %v", err)
	}

	// Body should be wrapped in streamTranslator
	output, _ := io.ReadAll(res.Body)
	outputStr := string(output)

	if !strings.Contains(outputStr, "message_start") {
		t.Error("expected message_start in translated stream")
	}
	if !strings.Contains(outputStr, "message_stop") {
		t.Error("expected message_stop in translated stream")
	}
}

func TestMiddlewareContentLengthUpdated(t *testing.T) {
	middleware := openaiMiddleware("")

	body := `{
		"model": "test",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "Hi"}]
	}`

	req, _ := http.NewRequest("POST", "http://localhost/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	res, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		oaiRes := `{"id":"x","object":"chat.completion","model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"Ok"},"finish_reason":"stop"}]}`
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
	if res.ContentLength != int64(len(resBody)) {
		t.Errorf("content-length mismatch: header=%d actual=%d", res.ContentLength, len(resBody))
	}
}

// ---------------------------------------------------------------------------
// Streaming translation tests
// ---------------------------------------------------------------------------

func TestStreamTranslatorBasic(t *testing.T) {
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
	if !containsSubstring(outputStr, "message_delta") {
		t.Error("expected message_delta event")
	}
}

func TestStreamTranslatorNullChoices(t *testing.T) {
	// This is the Cline crash case — some providers send null choices before [DONE]
	chunks := []string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":null}`,
		``,
		`data: [DONE]`,
		``,
	}

	body := io.NopCloser(bytes.NewReader([]byte(joinLines(chunks))))
	translator := &streamTranslator{rc: body}

	// Should not panic
	output, err := io.ReadAll(translator)
	if err != nil {
		t.Fatalf("failed to read stream (should handle null choices): %v", err)
	}

	if !containsSubstring(string(output), "message_start") {
		t.Error("expected message_start even with null choices chunk")
	}
}

func TestStreamTranslatorEmptyDelta(t *testing.T) {
	chunks := []string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"content":"text"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}

	body := io.NopCloser(bytes.NewReader([]byte(joinLines(chunks))))
	translator := &streamTranslator{rc: body}

	output, err := io.ReadAll(translator)
	if err != nil {
		t.Fatalf("failed to read stream: %v", err)
	}

	if !containsSubstring(string(output), "text") {
		t.Error("expected text content in output")
	}
}

func TestStreamTranslatorUsageOnlyChunk(t *testing.T) {
	// Some providers send a usage-only chunk when stream_options.include_usage=true
	chunks := []string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
		``,
		`data: [DONE]`,
		``,
	}

	body := io.NopCloser(bytes.NewReader([]byte(joinLines(chunks))))
	translator := &streamTranslator{rc: body}

	output, err := io.ReadAll(translator)
	if err != nil {
		t.Fatalf("failed to read stream: %v", err)
	}

	outputStr := string(output)
	if !containsSubstring(outputStr, "message_stop") {
		t.Error("expected message_stop")
	}
}

func TestStreamTranslatorToolCallDelta(t *testing.T) {
	chunks := []string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"NYC\"}"}}]},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}

	body := io.NopCloser(bytes.NewReader([]byte(joinLines(chunks))))
	translator := &streamTranslator{rc: body}

	output, err := io.ReadAll(translator)
	if err != nil {
		t.Fatalf("failed to read stream: %v", err)
	}

	outputStr := string(output)

	if !containsSubstring(outputStr, "content_block_start") {
		t.Error("expected content_block_start for tool call")
	}
	if !containsSubstring(outputStr, "tool_use") {
		t.Error("expected tool_use in content block")
	}
	if !containsSubstring(outputStr, "input_json_delta") {
		t.Error("expected input_json_delta for tool arguments")
	}
	if !containsSubstring(outputStr, "get_weather") {
		t.Error("expected function name get_weather")
	}
}

func TestStreamTranslatorClose(t *testing.T) {
	closed := false
	body := &mockReadCloser{
		Reader: bytes.NewReader([]byte("data: [DONE]\n\n")),
		closeFn: func() error {
			closed = true
			return nil
		},
	}
	translator := &streamTranslator{rc: body}
	translator.Close()

	if !closed {
		t.Error("expected underlying reader to be closed")
	}
}

func TestStreamTranslatorMalformedJSON(t *testing.T) {
	chunks := []string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}`,
		``,
		`data: {invalid json}`,
		``,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}`,
		``,
		`data: [DONE]`,
		``,
	}

	body := io.NopCloser(bytes.NewReader([]byte(joinLines(chunks))))
	translator := &streamTranslator{rc: body}

	// Should not panic on malformed JSON — gjson handles gracefully
	output, err := io.ReadAll(translator)
	if err != nil {
		t.Fatalf("failed to read stream: %v", err)
	}

	if !containsSubstring(string(output), "Hi") {
		t.Error("expected valid chunks to still be processed")
	}
}

func TestStreamTranslatorEventPrefix(t *testing.T) {
	// Verify non-data lines are skipped
	chunks := []string{
		`: this is a comment`,
		`event: something`,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}`,
		``,
		`data: [DONE]`,
		``,
	}

	body := io.NopCloser(bytes.NewReader([]byte(joinLines(chunks))))
	translator := &streamTranslator{rc: body}

	output, err := io.ReadAll(translator)
	if err != nil {
		t.Fatalf("failed to read stream: %v", err)
	}

	if !containsSubstring(string(output), "OK") {
		t.Error("expected valid data line to be processed despite other SSE fields")
	}
}

// ---------------------------------------------------------------------------
// extractSystemText tests
// ---------------------------------------------------------------------------

func TestExtractSystemTextString(t *testing.T) {
	result := extractSystemText(gjson.Parse(`"Hello"`))
	if result != "Hello" {
		t.Errorf("expected 'Hello', got %q", result)
	}
}

func TestExtractSystemTextArray(t *testing.T) {
	result := extractSystemText(gjson.Parse(`[{"type":"text","text":"A"},{"type":"text","text":"B"}]`))
	if result != "A\nB" {
		t.Errorf("expected 'A\\nB', got %q", result)
	}
}

func TestExtractSystemTextEmptyArray(t *testing.T) {
	result := extractSystemText(gjson.Parse(`[]`))
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

type mockReadCloser struct {
	*bytes.Reader
	closeFn func() error
}

func (m *mockReadCloser) Close() error {
	if m.closeFn != nil {
		return m.closeFn()
	}
	return nil
}

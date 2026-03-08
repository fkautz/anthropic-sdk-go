// Package openai provides middleware to use the Anthropic SDK with any
// OpenAI-compatible API (OpenAI, Ollama, vLLM, LiteLLM, etc.).
//
// The middleware intercepts requests to /v1/messages, translates them from
// Anthropic's format to the OpenAI chat completions format, sends the
// request, and translates the response back into Anthropic's format.
//
// Usage:
//
//	client := anthropic.NewClient(
//	    openai.WithBaseURL("http://localhost:11434/v1", "llama3"),  // Ollama
//	)
//
//	// Or with an API key for OpenAI itself:
//	client := anthropic.NewClient(
//	    openai.WithAPIKey("http://localhost:11434/v1", "llama3", "sk-..."),
//	)
package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go/internal/requestconfig"
	sdkoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/tidwall/gjson"
)

// WithBaseURL returns a request option that configures the client to use an
// OpenAI-compatible endpoint. The defaultModel is used when no model mapping
// is needed (pass "" to use the model from the request as-is).
func WithBaseURL(baseURL string, defaultModel string) sdkoption.RequestOption {
	middleware := openaiMiddleware(defaultModel)
	return requestconfig.RequestOptionFunc(func(rc *requestconfig.RequestConfig) error {
		return rc.Apply(
			sdkoption.WithBaseURL(baseURL),
			sdkoption.WithMiddleware(middleware),
		)
	})
}

// WithAPIKey returns a request option like WithBaseURL but also sets a Bearer
// token for authentication.
func WithAPIKey(baseURL string, defaultModel string, apiKey string) sdkoption.RequestOption {
	middleware := openaiMiddleware(defaultModel)
	return requestconfig.RequestOptionFunc(func(rc *requestconfig.RequestConfig) error {
		return rc.Apply(
			sdkoption.WithBaseURL(baseURL),
			sdkoption.WithMiddleware(middleware),
			sdkoption.WithHeader("Authorization", "Bearer "+apiKey),
		)
	})
}

func init() {
	// Register a decoder for OpenAI-style SSE streaming. OpenAI uses the same
	// text/event-stream content type as Anthropic, so the default SSE decoder
	// already handles it. We register a custom decoder that translates each
	// OpenAI chunk into an Anthropic-format SSE event before passing it along.
	//
	// NOTE: We do NOT register a custom content-type decoder here because the
	// response rewriting in the middleware already handles the translation.
	// The streaming translation happens inline in the middleware by wrapping
	// the response body.
	_ = 0 // placeholder — streaming is handled via response body wrapping
}

// openaiMiddleware returns middleware that translates Anthropic requests/responses
// to/from the OpenAI chat completions format.
func openaiMiddleware(defaultModel string) sdkoption.Middleware {
	return func(r *http.Request, next sdkoption.MiddlewareNext) (*http.Response, error) {
		if r.Body == nil {
			return next(r)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		r.Body.Close()

		// Only intercept messages endpoint
		if r.URL.Path != "/v1/messages" || r.Method != http.MethodPost {
			reader := bytes.NewReader(body)
			r.Body = io.NopCloser(reader)
			r.GetBody = func() (io.ReadCloser, error) {
				_, err := reader.Seek(0, 0)
				return io.NopCloser(reader), err
			}
			r.ContentLength = int64(len(body))
			return next(r)
		}

		stream := gjson.GetBytes(body, "stream").Bool()

		// Translate request
		oaiBody, err := translateRequest(body, defaultModel)
		if err != nil {
			return nil, fmt.Errorf("openai: failed to translate request: %w", err)
		}

		reader := bytes.NewReader(oaiBody)
		r.Body = io.NopCloser(reader)
		r.GetBody = func() (io.ReadCloser, error) {
			_, err := reader.Seek(0, 0)
			return io.NopCloser(reader), err
		}
		r.ContentLength = int64(len(oaiBody))

		// Rewrite the path to the OpenAI chat completions endpoint
		r.URL.Path = "/v1/chat/completions"

		// Remove Anthropic-specific headers
		r.Header.Del("anthropic-version")
		r.Header.Del("X-Api-Key")

		res, err := next(r)
		if err != nil {
			return res, err
		}

		if res.StatusCode >= 400 {
			return res, nil
		}

		if stream {
			// Wrap the response body to translate streaming chunks
			res.Body = &streamTranslator{rc: res.Body}
		} else {
			// Translate the non-streaming response
			resBody, err := io.ReadAll(res.Body)
			res.Body.Close()
			if err != nil {
				return nil, fmt.Errorf("openai: failed to read response: %w", err)
			}

			anthropicBody, err := translateResponse(resBody)
			if err != nil {
				return nil, fmt.Errorf("openai: failed to translate response: %w", err)
			}

			res.Body = io.NopCloser(bytes.NewReader(anthropicBody))
			res.ContentLength = int64(len(anthropicBody))
			res.Header.Set("Content-Length", fmt.Sprintf("%d", len(anthropicBody)))
		}

		return res, nil
	}
}

// ---------- Request translation ----------

// oaiRequest is the OpenAI chat completion request format.
type oaiRequest struct {
	Model            string       `json:"model"`
	Messages         []oaiMessage `json:"messages"`
	MaxTokens        *int64       `json:"max_tokens,omitempty"`
	Temperature      *float64     `json:"temperature,omitempty"`
	TopP             *float64     `json:"top_p,omitempty"`
	Stop             []string     `json:"stop,omitempty"`
	Stream           bool         `json:"stream,omitempty"`
	StreamOptions    *streamOpts  `json:"stream_options,omitempty"`
	Tools            []oaiTool    `json:"tools,omitempty"`
	FrequencyPenalty *float64     `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64     `json:"presence_penalty,omitempty"`
}

type streamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

type oaiMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`           // string or array
	Name       string          `json:"name,omitempty"`
	ToolCalls  []oaiToolCall   `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

type oaiTool struct {
	Type     string      `json:"type"`
	Function oaiFunction `json:"function"`
}

type oaiFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type oaiToolCall struct {
	ID       string      `json:"id"`
	Type     string      `json:"type"`
	Function oaiCallFunc `json:"function"`
}

type oaiCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func translateRequest(body []byte, defaultModel string) ([]byte, error) {
	// Parse the Anthropic request using gjson for flexibility
	parsed := gjson.ParseBytes(body)

	model := parsed.Get("model").String()
	if defaultModel != "" {
		model = defaultModel
	}

	req := oaiRequest{
		Model:  model,
		Stream: parsed.Get("stream").Bool(),
	}

	if req.Stream {
		req.StreamOptions = &streamOpts{IncludeUsage: true}
	}

	// max_tokens
	if mt := parsed.Get("max_tokens"); mt.Exists() {
		v := mt.Int()
		req.MaxTokens = &v
	}

	// temperature
	if t := parsed.Get("temperature"); t.Exists() {
		v := t.Float()
		req.Temperature = &v
	}

	// top_p
	if tp := parsed.Get("top_p"); tp.Exists() {
		v := tp.Float()
		req.TopP = &v
	}

	// stop_sequences -> stop
	if ss := parsed.Get("stop_sequences"); ss.Exists() && ss.IsArray() {
		for _, s := range ss.Array() {
			req.Stop = append(req.Stop, s.String())
		}
	}

	// system prompt -> system message
	if sys := parsed.Get("system"); sys.Exists() {
		systemText := extractSystemText(sys)
		if systemText != "" {
			content, _ := json.Marshal(systemText)
			req.Messages = append(req.Messages, oaiMessage{
				Role:    "system",
				Content: content,
			})
		}
	}

	// messages
	if msgs := parsed.Get("messages"); msgs.Exists() && msgs.IsArray() {
		for _, msg := range msgs.Array() {
			oaiMsgs, err := translateMessage(msg)
			if err != nil {
				return nil, err
			}
			req.Messages = append(req.Messages, oaiMsgs...)
		}
	}

	// tools
	if tools := parsed.Get("tools"); tools.Exists() && tools.IsArray() {
		for _, tool := range tools.Array() {
			oaiT := translateTool(tool)
			if oaiT != nil {
				req.Tools = append(req.Tools, *oaiT)
			}
		}
	}

	return json.Marshal(req)
}

func extractSystemText(sys gjson.Result) string {
	// system can be a string or an array of text blocks
	if sys.IsArray() {
		var parts []string
		for _, block := range sys.Array() {
			if block.Get("type").String() == "text" || !block.Get("type").Exists() {
				parts = append(parts, block.Get("text").String())
			}
		}
		return strings.Join(parts, "\n")
	}
	return sys.String()
}

func translateMessage(msg gjson.Result) ([]oaiMessage, error) {
	role := msg.Get("role").String()
	content := msg.Get("content")

	// Handle tool_result role -> tool role in OpenAI
	if role == "user" && content.IsArray() {
		var userParts []string
		var toolResults []oaiMessage

		for _, block := range content.Array() {
			blockType := block.Get("type").String()
			if blockType == "tool_result" {
				toolContent := block.Get("content")
				var text string
				if toolContent.IsArray() {
					// array of content blocks
					var parts []string
					for _, tc := range toolContent.Array() {
						if tc.Get("type").String() == "text" {
							parts = append(parts, tc.Get("text").String())
						}
					}
					text = strings.Join(parts, "\n")
				} else {
					text = toolContent.String()
				}
				contentJSON, _ := json.Marshal(text)
				toolResults = append(toolResults, oaiMessage{
					Role:       "tool",
					Content:    contentJSON,
					ToolCallID: block.Get("tool_use_id").String(),
				})
			} else if blockType == "text" {
				userParts = append(userParts, block.Get("text").String())
			}
		}

		var result []oaiMessage
		if len(userParts) > 0 {
			c, _ := json.Marshal(strings.Join(userParts, "\n"))
			result = append(result, oaiMessage{Role: "user", Content: c})
		}
		result = append(result, toolResults...)
		return result, nil
	}

	// Handle assistant messages with tool_use blocks
	if role == "assistant" && content.IsArray() {
		var textParts []string
		var toolCalls []oaiToolCall

		for _, block := range content.Array() {
			blockType := block.Get("type").String()
			switch blockType {
			case "text":
				textParts = append(textParts, block.Get("text").String())
			case "tool_use":
				args := block.Get("input").Raw
				if args == "" {
					args = "{}"
				}
				toolCalls = append(toolCalls, oaiToolCall{
					ID:   block.Get("id").String(),
					Type: "function",
					Function: oaiCallFunc{
						Name:      block.Get("name").String(),
						Arguments: args,
					},
				})
			}
		}

		text := strings.Join(textParts, "\n")
		var contentJSON json.RawMessage
		if text != "" {
			contentJSON, _ = json.Marshal(text)
		} else {
			contentJSON = json.RawMessage("null")
		}
		msg := oaiMessage{
			Role:      "assistant",
			Content:   contentJSON,
			ToolCalls: toolCalls,
		}
		return []oaiMessage{msg}, nil
	}

	// Simple string or text-block content
	if content.IsArray() {
		var parts []string
		for _, block := range content.Array() {
			if block.Get("type").String() == "text" || !block.Get("type").Exists() {
				parts = append(parts, block.Get("text").String())
			}
		}
		text := strings.Join(parts, "\n")
		c, _ := json.Marshal(text)
		return []oaiMessage{{Role: role, Content: c}}, nil
	}

	// Content is already a string
	c, _ := json.Marshal(content.String())
	return []oaiMessage{{Role: role, Content: c}}, nil
}

func translateTool(tool gjson.Result) *oaiTool {
	toolType := tool.Get("type").String()
	if toolType != "custom" && toolType != "" {
		// Skip server tools (web_search, etc.) — not supported by OpenAI
		if toolType != "custom" && tool.Get("name").String() == "" {
			return nil
		}
	}

	name := tool.Get("name").String()
	if name == "" {
		return nil
	}

	return &oaiTool{
		Type: "function",
		Function: oaiFunction{
			Name:        name,
			Description: tool.Get("description").String(),
			Parameters:  json.RawMessage(tool.Get("input_schema").Raw),
		},
	}
}

// ---------- Response translation ----------

// oaiResponse is the OpenAI chat completion response format.
type oaiResponse struct {
	ID      string     `json:"id"`
	Object  string     `json:"object"`
	Model   string     `json:"model"`
	Choices []oaiChoice `json:"choices"`
	Usage   *oaiUsage  `json:"usage,omitempty"`
}

type oaiChoice struct {
	Index        int           `json:"index"`
	Message      oaiChoiceMsg  `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type oaiChoiceMsg struct {
	Role      string        `json:"role"`
	Content   *string       `json:"content"`
	ToolCalls []oaiToolCall `json:"tool_calls,omitempty"`
}

type oaiUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

func translateResponse(body []byte) ([]byte, error) {
	var oaiRes oaiResponse
	if err := json.Unmarshal(body, &oaiRes); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI response: %w", err)
	}

	// Build Anthropic response
	anthropic := map[string]any{
		"id":   oaiRes.ID,
		"type": "message",
		"role": "assistant",
		"model": oaiRes.Model,
	}

	var content []map[string]any
	stopReason := "end_turn"

	if len(oaiRes.Choices) > 0 {
		choice := oaiRes.Choices[0]

		// Map finish_reason
		stopReason = mapFinishReason(choice.FinishReason)

		// Text content
		if choice.Message.Content != nil && *choice.Message.Content != "" {
			content = append(content, map[string]any{
				"type": "text",
				"text": *choice.Message.Content,
			})
		}

		// Tool calls
		for _, tc := range choice.Message.ToolCalls {
			var input json.RawMessage
			if tc.Function.Arguments != "" {
				input = json.RawMessage(tc.Function.Arguments)
			} else {
				input = json.RawMessage("{}")
			}
			content = append(content, map[string]any{
				"type":  "tool_use",
				"id":    tc.ID,
				"name":  tc.Function.Name,
				"input": input,
			})
		}
	}

	if len(content) == 0 {
		content = append(content, map[string]any{
			"type": "text",
			"text": "",
		})
	}

	anthropic["content"] = content
	anthropic["stop_reason"] = stopReason
	anthropic["stop_sequence"] = nil

	// Usage
	usage := map[string]any{
		"input_tokens":  int64(0),
		"output_tokens": int64(0),
	}
	if oaiRes.Usage != nil {
		usage["input_tokens"] = oaiRes.Usage.PromptTokens
		usage["output_tokens"] = oaiRes.Usage.CompletionTokens
	}
	anthropic["usage"] = usage

	return json.Marshal(anthropic)
}

func mapFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	case "content_filter":
		return "refusal"
	default:
		return "end_turn"
	}
}

// ---------- Streaming translation ----------

// streamTranslator wraps an OpenAI SSE response body and translates each
// chunk into Anthropic SSE format on the fly.
type streamTranslator struct {
	rc     io.ReadCloser
	buf    bytes.Buffer
	model  string
	id     string
	started bool
	index  int
}

func (s *streamTranslator) Read(p []byte) (int, error) {
	// If we have buffered translated data, return it first
	if s.buf.Len() > 0 {
		return s.buf.Read(p)
	}

	// Read lines from the underlying response
	tmp := make([]byte, 4096)
	n, err := s.rc.Read(tmp)
	if n > 0 {
		lines := strings.Split(string(tmp[:n]), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				s.writeMessageStop()
				continue
			}

			s.translateChunk([]byte(data))
		}
	}

	if s.buf.Len() > 0 {
		nn, _ := s.buf.Read(p)
		return nn, err
	}

	return 0, err
}

func (s *streamTranslator) Close() error {
	return s.rc.Close()
}

func (s *streamTranslator) translateChunk(data []byte) {
	parsed := gjson.ParseBytes(data)

	if !s.started {
		s.id = parsed.Get("id").String()
		s.model = parsed.Get("model").String()
		s.writeMessageStart()
		s.started = true
	}

	choices := parsed.Get("choices")
	if !choices.Exists() || !choices.IsArray() {
		// Check for usage-only chunk (stream_options)
		if usage := parsed.Get("usage"); usage.Exists() {
			s.writeMessageDelta(usage)
		}
		return
	}

	for _, choice := range choices.Array() {
		delta := choice.Get("delta")
		finishReason := choice.Get("finish_reason").String()

		// Tool calls in delta
		if tcs := delta.Get("tool_calls"); tcs.Exists() && tcs.IsArray() {
			for _, tc := range tcs.Array() {
				idx := int(tc.Get("index").Int())

				// If this is a new tool call (has id), start a content block
				if tc.Get("id").Exists() {
					s.writeContentBlockStart(idx, map[string]any{
						"type":  "tool_use",
						"id":    tc.Get("id").String(),
						"name":  tc.Get("function.name").String(),
						"input": json.RawMessage("{}"),
					})
				}

				// Arguments delta
				if args := tc.Get("function.arguments").String(); args != "" {
					s.writeSSE("content_block_delta", map[string]any{
						"type":  "content_block_delta",
						"index": idx,
						"delta": map[string]any{
							"type":          "input_json_delta",
							"partial_json": args,
						},
					})
				}
			}
		}

		// Text content delta
		if content := delta.Get("content").String(); content != "" {
			if !s.started || s.index == 0 {
				// Start a text content block if we haven't yet
				if s.index == 0 && !s.hasStartedTextBlock(data) {
					s.writeContentBlockStart(0, map[string]any{
						"type": "text",
						"text": "",
					})
					s.index = 1
				}
			}
			s.writeSSE("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]any{
					"type": "text_delta",
					"text": content,
				},
			})
		}

		if finishReason != "" && finishReason != "null" {
			// Close any open content blocks
			s.writeSSE("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": 0,
			})

			// Usage from the final chunk
			usage := parsed.Get("usage")
			s.writeMessageDelta(usage)
		}
	}
}

func (s *streamTranslator) hasStartedTextBlock(_ []byte) bool {
	return s.index > 0
}

func (s *streamTranslator) writeMessageStart() {
	s.writeSSE("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            s.id,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         s.model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	})
}

func (s *streamTranslator) writeContentBlockStart(index int, block map[string]any) {
	s.writeSSE("content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         index,
		"content_block": block,
	})
}

func (s *streamTranslator) writeMessageDelta(usage gjson.Result) {
	delta := map[string]any{
		"stop_reason":   "end_turn",
		"stop_sequence": nil,
	}

	usageMap := map[string]any{
		"output_tokens": int64(0),
	}
	if usage.Exists() {
		usageMap["output_tokens"] = usage.Get("completion_tokens").Int()
	}

	s.writeSSE("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": delta,
		"usage": usageMap,
	})
}

func (s *streamTranslator) writeMessageStop() {
	s.writeSSE("message_stop", map[string]any{
		"type": "message_stop",
	})
}

func (s *streamTranslator) writeSSE(event string, data any) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return
	}
	fmt.Fprintf(&s.buf, "event: %s\ndata: %s\n\n", event, jsonData)
}

// Ensure streamTranslator is compatible with the ssestream package
var _ io.ReadCloser = (*streamTranslator)(nil)

// Ensure ssestream.Decoder is used (prevents unused import).
var _ ssestream.Decoder = nil

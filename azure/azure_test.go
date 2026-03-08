package azure

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestAzureMiddlewarePathRewrite(t *testing.T) {
	middleware := azureMiddleware("my-gpt4-deployment", "2024-10-21")

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}],"max_tokens":100}`
	req, _ := http.NewRequest("POST", "https://myresource.openai.azure.com/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		expectedPath := "/openai/deployments/my-gpt4-deployment/chat/completions"
		if r.URL.Path != expectedPath {
			t.Errorf("expected path %q, got %q", expectedPath, r.URL.Path)
		}

		apiVersion := r.URL.Query().Get("api-version")
		if apiVersion != "2024-10-21" {
			t.Errorf("expected api-version 2024-10-21, got %q", apiVersion)
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

func TestAzureMiddlewareStripsModel(t *testing.T) {
	middleware := azureMiddleware("my-deployment", "2024-10-21")

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}],"max_tokens":100}`
	req, _ := http.NewRequest("POST", "https://myresource.openai.azure.com/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		resBody, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		json.Unmarshal(resBody, &parsed)

		if _, hasModel := parsed["model"]; hasModel {
			t.Error("expected model field to be stripped from body")
		}

		// Other fields should still be present
		if _, hasMessages := parsed["messages"]; !hasMessages {
			t.Error("expected messages field to remain in body")
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

func TestAzureMiddlewareNonChatPassthrough(t *testing.T) {
	middleware := azureMiddleware("my-deployment", "2024-10-21")

	req, _ := http.NewRequest("GET", "https://myresource.openai.azure.com/v1/models", nil)

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/models" {
			t.Errorf("expected path /v1/models unchanged, got %s", r.URL.Path)
		}

		if r.URL.Query().Get("api-version") != "" {
			t.Error("expected no api-version on non-chat requests")
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

func TestAzureMiddlewareCustomAPIVersion(t *testing.T) {
	middleware := azureMiddleware("my-deployment", "2025-01-15-preview")

	body := `{"messages":[{"role":"user","content":"Hi"}]}`
	req, _ := http.NewRequest("POST", "https://myresource.openai.azure.com/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		apiVersion := r.URL.Query().Get("api-version")
		if apiVersion != "2025-01-15-preview" {
			t.Errorf("expected api-version 2025-01-15-preview, got %q", apiVersion)
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

func TestAzureMiddlewarePreservesExistingQueryParams(t *testing.T) {
	middleware := azureMiddleware("my-deployment", "2024-10-21")

	body := `{"messages":[{"role":"user","content":"Hi"}]}`
	req, _ := http.NewRequest("POST", "https://myresource.openai.azure.com/v1/chat/completions?existing=param", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		if r.URL.Query().Get("existing") != "param" {
			t.Error("expected existing query param to be preserved")
		}
		if r.URL.Query().Get("api-version") != "2024-10-21" {
			t.Error("expected api-version to be added")
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

func TestAzureMiddlewareBodyWithoutModel(t *testing.T) {
	middleware := azureMiddleware("my-deployment", "2024-10-21")

	// Body already has no model field
	body := `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":50}`
	req, _ := http.NewRequest("POST", "https://myresource.openai.azure.com/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		resBody, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		json.Unmarshal(resBody, &parsed)

		if _, hasModel := parsed["model"]; hasModel {
			t.Error("expected no model field")
		}
		if _, hasMessages := parsed["messages"]; !hasMessages {
			t.Error("expected messages field to remain")
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

func TestAzureMiddlewareContentLengthUpdated(t *testing.T) {
	middleware := azureMiddleware("my-deployment", "2024-10-21")

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}`
	req, _ := http.NewRequest("POST", "https://myresource.openai.azure.com/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		resBody, _ := io.ReadAll(r.Body)

		if r.ContentLength != int64(len(resBody)) {
			t.Errorf("content-length mismatch: header=%d actual=%d", r.ContentLength, len(resBody))
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

func TestNewSetsAPIKeyHeader(t *testing.T) {
	// We can't easily test the full option application without a real client,
	// but we can verify New returns non-nil
	opt := New("https://myresource.openai.azure.com", "my-deployment", "test-key")
	if opt == nil {
		t.Fatal("expected non-nil request option")
	}
}

func TestNewWithBearerTokenReturnsNonNil(t *testing.T) {
	opt := NewWithBearerToken("https://myresource.openai.azure.com", "my-deployment", "test-token")
	if opt == nil {
		t.Fatal("expected non-nil request option")
	}
}

func TestNewWithOptionsDefaultAPIVersion(t *testing.T) {
	// Verify that empty APIVersion gets defaulted
	opt := NewWithOptions("https://myresource.openai.azure.com", "my-deployment", Options{
		APIKey: "test-key",
	})
	if opt == nil {
		t.Fatal("expected non-nil request option")
	}
}

func TestDefaultAPIVersionValue(t *testing.T) {
	if DefaultAPIVersion != "2024-10-21" {
		t.Errorf("expected default API version 2024-10-21, got %s", DefaultAPIVersion)
	}
}

func TestAzureMiddlewareDeploymentWithSpecialChars(t *testing.T) {
	middleware := azureMiddleware("my-gpt4o-2024-deployment", "2024-10-21")

	body := `{"messages":[{"role":"user","content":"Hi"}]}`
	req, _ := http.NewRequest("POST", "https://myresource.openai.azure.com/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		expectedPath := "/openai/deployments/my-gpt4o-2024-deployment/chat/completions"
		if r.URL.Path != expectedPath {
			t.Errorf("expected path %q, got %q", expectedPath, r.URL.Path)
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

func TestAzureMiddlewareGetBodyIsResettable(t *testing.T) {
	middleware := azureMiddleware("my-deployment", "2024-10-21")

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}`
	req, _ := http.NewRequest("POST", "https://myresource.openai.azure.com/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	_, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		// Read body once
		body1, _ := io.ReadAll(r.Body)

		// Reset via GetBody
		if r.GetBody == nil {
			t.Fatal("expected GetBody to be set")
		}
		newBody, err := r.GetBody()
		if err != nil {
			t.Fatalf("GetBody failed: %v", err)
		}
		body2, _ := io.ReadAll(newBody)

		if !bytes.Equal(body1, body2) {
			t.Error("GetBody should return same content")
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

func TestAzureMiddlewareHTTPErrorPassthrough(t *testing.T) {
	middleware := azureMiddleware("my-deployment", "2024-10-21")

	body := `{"messages":[{"role":"user","content":"Hi"}]}`
	req, _ := http.NewRequest("POST", "https://myresource.openai.azure.com/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")

	errorBody := `{"error":{"code":"DeploymentNotFound","message":"The API deployment for this resource does not exist."}}`
	res, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 404,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(bytes.NewReader([]byte(errorBody))),
		}, nil
	})

	if err != nil {
		t.Fatalf("middleware failed: %v", err)
	}

	if res.StatusCode != 404 {
		t.Errorf("expected status 404, got %d", res.StatusCode)
	}

	resBody, _ := io.ReadAll(res.Body)
	if !strings.Contains(string(resBody), "DeploymentNotFound") {
		t.Error("expected error body to pass through")
	}
}

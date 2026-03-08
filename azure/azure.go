// Package azure provides middleware to use the Anthropic SDK with Azure OpenAI.
//
// Azure OpenAI uses a different URL scheme and authentication from standard
// OpenAI. The model is determined by the deployment name in the URL, not the
// request body.
//
// Usage with API key:
//
//	client := anthropic.NewClient(
//	    azure.New("https://myresource.openai.azure.com", "my-deployment", "my-api-key"),
//	)
//
// Usage with Entra ID (Bearer token):
//
//	client := anthropic.NewClient(
//	    azure.NewWithBearerToken("https://myresource.openai.azure.com", "my-deployment", token),
//	)
package azure

import (
	"bytes"
	"fmt"
	"io"
	"net/http"

	"github.com/anthropics/anthropic-sdk-go/internal/requestconfig"
	"github.com/anthropics/anthropic-sdk-go/openai"
	sdkoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// DefaultAPIVersion is the latest GA API version for Azure OpenAI.
const DefaultAPIVersion = "2024-10-21"

// New returns a request option that configures the client to use Azure OpenAI
// with API key authentication.
//
// Parameters:
//   - endpoint: Your Azure OpenAI resource endpoint (e.g., "https://myresource.openai.azure.com")
//   - deploymentID: The name of your model deployment
//   - apiKey: Your Azure OpenAI API key
func New(endpoint string, deploymentID string, apiKey string) sdkoption.RequestOption {
	return NewWithOptions(endpoint, deploymentID, Options{
		APIKey:     apiKey,
		APIVersion: DefaultAPIVersion,
	})
}

// NewWithBearerToken returns a request option that configures the client to use
// Azure OpenAI with Microsoft Entra ID (Bearer token) authentication.
func NewWithBearerToken(endpoint string, deploymentID string, token string) sdkoption.RequestOption {
	return NewWithOptions(endpoint, deploymentID, Options{
		BearerToken: token,
		APIVersion:  DefaultAPIVersion,
	})
}

// Options configures the Azure OpenAI middleware.
type Options struct {
	// APIKey for api-key header authentication. Mutually exclusive with BearerToken.
	APIKey string

	// BearerToken for Microsoft Entra ID authentication. Mutually exclusive with APIKey.
	BearerToken string

	// APIVersion is the Azure OpenAI API version query parameter.
	// Defaults to DefaultAPIVersion if empty.
	APIVersion string
}

// NewWithOptions returns a request option with full control over Azure OpenAI
// configuration.
func NewWithOptions(endpoint string, deploymentID string, opts Options) sdkoption.RequestOption {
	if opts.APIVersion == "" {
		opts.APIVersion = DefaultAPIVersion
	}

	// The openai middleware handles Anthropic→OpenAI request/response translation.
	// We layer an Azure middleware on top to rewrite the URL and auth.
	azureMiddleware := azureMiddleware(deploymentID, opts.APIVersion)

	return requestconfig.RequestOptionFunc(func(rc *requestconfig.RequestConfig) error {
		err := rc.Apply(
			// Use the openai middleware for format translation, passing empty
			// defaultModel since Azure determines the model by deployment.
			openai.WithBaseURL(endpoint, ""),
			sdkoption.WithMiddleware(azureMiddleware),
		)
		if err != nil {
			return err
		}

		// Set authentication
		if opts.APIKey != "" {
			rc.Request.Header.Set("api-key", opts.APIKey)
		} else if opts.BearerToken != "" {
			rc.Request.Header.Set("Authorization", "Bearer "+opts.BearerToken)
		}

		return nil
	})
}

// azureMiddleware rewrites the OpenAI-format URL to Azure's deployment-based
// URL scheme and strips the model from the request body (Azure ignores it).
func azureMiddleware(deploymentID string, apiVersion string) sdkoption.Middleware {
	return func(r *http.Request, next sdkoption.MiddlewareNext) (*http.Response, error) {
		// Only rewrite chat completions (the openai middleware already rewrote
		// /v1/messages → /v1/chat/completions)
		if r.URL.Path == "/v1/chat/completions" && r.Method == http.MethodPost {
			// Rewrite to Azure's deployment URL
			r.URL.Path = fmt.Sprintf("/openai/deployments/%s/chat/completions", deploymentID)

			// Add api-version query parameter
			q := r.URL.Query()
			q.Set("api-version", apiVersion)
			r.URL.RawQuery = q.Encode()

			// Remove the model field from the body — Azure uses the deployment ID
			if r.Body != nil {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				r.Body.Close()

				// Strip model if present (Azure ignores it but cleaner without)
				if gjson.GetBytes(body, "model").Exists() {
					body, _ = sjson.DeleteBytes(body, "model")
				}

				reader := bytes.NewReader(body)
				r.Body = io.NopCloser(reader)
				r.GetBody = func() (io.ReadCloser, error) {
					_, err := reader.Seek(0, 0)
					return io.NopCloser(reader), err
				}
				r.ContentLength = int64(len(body))
			}
		}

		return next(r)
	}
}

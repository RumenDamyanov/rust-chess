use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::config::ProviderConfig;

use super::types::ChatError;

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

/// Trait for LLM providers.  Each provider takes a system prompt + user
/// message and returns the assistant's reply text.
pub trait LlmProvider: Send + Sync {
    /// Send a prompt to the LLM and return the response text.
    fn ask<'a>(
        &'a self,
        system_prompt: &'a str,
        user_message: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, ChatError>> + Send + 'a>>;

    /// Provider name for logging / response metadata.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// OpenAI-compatible provider (OpenAI, xAI, DeepSeek)
// ---------------------------------------------------------------------------

/// Works with any OpenAI-compatible chat-completions API.
#[derive(Debug)]
pub struct OpenAiCompatible {
    pub provider_name: String,
    pub api_key: String,
    pub model: String,
    pub endpoint: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

impl OpenAiCompatible {
    pub fn new(provider_name: &str, config: &ProviderConfig) -> Result<Self, ChatError> {
        if config.api_key.is_empty() {
            return Err(ChatError::MissingApiKey(provider_name.to_string()));
        }
        Ok(Self {
            provider_name: provider_name.to_string(),
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            endpoint: config.endpoint.clone(),
            client: reqwest::Client::new(),
        })
    }
}

impl LlmProvider for OpenAiCompatible {
    fn ask<'a>(
        &'a self,
        system_prompt: &'a str,
        user_message: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, ChatError>> + Send + 'a>> {
        Box::pin(async move {
            let body = OpenAiRequest {
                model: self.model.clone(),
                messages: vec![
                    OpenAiMessage {
                        role: "system".to_string(),
                        content: system_prompt.to_string(),
                    },
                    OpenAiMessage {
                        role: "user".to_string(),
                        content: user_message.to_string(),
                    },
                ],
                max_tokens: 300,
                temperature: 0.7,
            };

            let resp = self
                .client
                .post(&self.endpoint)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| ChatError::RequestFailed(e.to_string()))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(ChatError::ProviderError(format!(
                    "{} returned {}: {}",
                    self.provider_name, status, text
                )));
            }

            let parsed: OpenAiResponse = resp
                .json()
                .await
                .map_err(|e| ChatError::ParseError(e.to_string()))?;

            parsed
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| ChatError::ParseError("empty choices array".to_string()))
        })
    }

    fn name(&self) -> &str {
        &self.provider_name
    }
}

// ---------------------------------------------------------------------------
// Anthropic provider
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct AnthropicProvider {
    pub api_key: String,
    pub model: String,
    pub endpoint: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    system: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

impl AnthropicProvider {
    pub fn new(config: &ProviderConfig) -> Result<Self, ChatError> {
        if config.api_key.is_empty() {
            return Err(ChatError::MissingApiKey("anthropic".to_string()));
        }
        Ok(Self {
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            endpoint: config.endpoint.clone(),
            client: reqwest::Client::new(),
        })
    }
}

impl LlmProvider for AnthropicProvider {
    fn ask<'a>(
        &'a self,
        system_prompt: &'a str,
        user_message: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, ChatError>> + Send + 'a>> {
        Box::pin(async move {
            let body = AnthropicRequest {
                model: self.model.clone(),
                system: system_prompt.to_string(),
                messages: vec![AnthropicMessage {
                    role: "user".to_string(),
                    content: user_message.to_string(),
                }],
                max_tokens: 300,
            };

            let resp = self
                .client
                .post(&self.endpoint)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| ChatError::RequestFailed(e.to_string()))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(ChatError::ProviderError(format!(
                    "anthropic returned {}: {}",
                    status, text
                )));
            }

            let parsed: AnthropicResponse = resp
                .json()
                .await
                .map_err(|e| ChatError::ParseError(e.to_string()))?;

            parsed
                .content
                .first()
                .map(|c| c.text.clone())
                .ok_or_else(|| ChatError::ParseError("empty content array".to_string()))
        })
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

// ---------------------------------------------------------------------------
// Gemini provider
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GeminiProvider {
    pub api_key: String,
    pub model: String,
    pub endpoint: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    system_instruction: GeminiSystemInstruction,
    contents: Vec<GeminiContent>,
    generation_config: GeminiGenerationConfig,
}

#[derive(Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    max_output_tokens: u32,
    temperature: f32,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiCandidateContent,
}

#[derive(Deserialize)]
struct GeminiCandidateContent {
    parts: Vec<GeminiPart>,
}

impl GeminiProvider {
    pub fn new(config: &ProviderConfig) -> Result<Self, ChatError> {
        if config.api_key.is_empty() {
            return Err(ChatError::MissingApiKey("gemini".to_string()));
        }
        Ok(Self {
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            endpoint: config.endpoint.clone(),
            client: reqwest::Client::new(),
        })
    }
}

impl LlmProvider for GeminiProvider {
    fn ask<'a>(
        &'a self,
        system_prompt: &'a str,
        user_message: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, ChatError>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!("{}/{}:generateContent", self.endpoint, self.model);

            let body = GeminiRequest {
                system_instruction: GeminiSystemInstruction {
                    parts: vec![GeminiPart {
                        text: system_prompt.to_string(),
                    }],
                },
                contents: vec![GeminiContent {
                    parts: vec![GeminiPart {
                        text: user_message.to_string(),
                    }],
                }],
                generation_config: GeminiGenerationConfig {
                    max_output_tokens: 300,
                    temperature: 0.7,
                },
            };

            let resp = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("x-goog-api-key", &self.api_key)
                .json(&body)
                .send()
                .await
                .map_err(|e| ChatError::RequestFailed(e.to_string()))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(ChatError::ProviderError(format!(
                    "gemini returned {}: {}",
                    status, text
                )));
            }

            let parsed: GeminiResponse = resp
                .json()
                .await
                .map_err(|e| ChatError::ParseError(e.to_string()))?;

            parsed
                .candidates
                .first()
                .and_then(|c| c.content.parts.first())
                .map(|p| p.text.clone())
                .ok_or_else(|| ChatError::ParseError("empty candidates".to_string()))
        })
    }

    fn name(&self) -> &str {
        "gemini"
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create an LLM provider from a provider name and config.
pub fn create_provider(
    name: &str,
    config: &ProviderConfig,
) -> Result<Box<dyn LlmProvider>, ChatError> {
    match name {
        "openai" | "xai" | "deepseek" => Ok(Box::new(OpenAiCompatible::new(name, config)?)),
        "anthropic" => Ok(Box::new(AnthropicProvider::new(config)?)),
        "gemini" => Ok(Box::new(GeminiProvider::new(config)?)),
        other => Err(ChatError::UnsupportedProvider(other.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_compatible_rejects_empty_key() {
        let cfg = ProviderConfig {
            api_key: String::new(),
            model: "gpt-4o-mini".to_string(),
            endpoint: "https://api.openai.com".to_string(),
        };
        let result = OpenAiCompatible::new("openai", &cfg);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ChatError::MissingApiKey(_)));
    }

    #[test]
    fn anthropic_rejects_empty_key() {
        let cfg = ProviderConfig {
            api_key: String::new(),
            model: "claude-3-haiku".to_string(),
            endpoint: "https://api.anthropic.com".to_string(),
        };
        assert!(AnthropicProvider::new(&cfg).is_err());
    }

    #[test]
    fn gemini_rejects_empty_key() {
        let cfg = ProviderConfig {
            api_key: String::new(),
            model: "gemini-1.5-flash".to_string(),
            endpoint: "https://generativelanguage.googleapis.com".to_string(),
        };
        assert!(GeminiProvider::new(&cfg).is_err());
    }

    #[test]
    fn openai_compatible_accepts_valid_key() {
        let cfg = ProviderConfig {
            api_key: "sk-test-key".to_string(),
            model: "gpt-4o-mini".to_string(),
            endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
        };
        let provider = OpenAiCompatible::new("openai", &cfg).unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.model, "gpt-4o-mini");
    }

    #[test]
    fn factory_creates_openai() {
        let cfg = ProviderConfig {
            api_key: "sk-test".to_string(),
            model: "gpt-4o-mini".to_string(),
            endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
        };
        let p = create_provider("openai", &cfg).unwrap();
        assert_eq!(p.name(), "openai");
    }

    #[test]
    fn factory_creates_xai() {
        let cfg = ProviderConfig {
            api_key: "xai-test".to_string(),
            model: "grok-1.5".to_string(),
            endpoint: "https://api.x.ai/v1/chat/completions".to_string(),
        };
        let p = create_provider("xai", &cfg).unwrap();
        assert_eq!(p.name(), "xai");
    }

    #[test]
    fn factory_creates_deepseek() {
        let cfg = ProviderConfig {
            api_key: "ds-test".to_string(),
            model: "deepseek-chat".to_string(),
            endpoint: "https://api.deepseek.com/v1/chat/completions".to_string(),
        };
        let p = create_provider("deepseek", &cfg).unwrap();
        assert_eq!(p.name(), "deepseek");
    }

    #[test]
    fn factory_creates_anthropic() {
        let cfg = ProviderConfig {
            api_key: "sk-ant-test".to_string(),
            model: "claude-3-haiku-20240307".to_string(),
            endpoint: "https://api.anthropic.com/v1/messages".to_string(),
        };
        let p = create_provider("anthropic", &cfg).unwrap();
        assert_eq!(p.name(), "anthropic");
    }

    #[test]
    fn factory_creates_gemini() {
        let cfg = ProviderConfig {
            api_key: "gem-test".to_string(),
            model: "gemini-1.5-flash".to_string(),
            endpoint: "https://generativelanguage.googleapis.com/v1beta/models".to_string(),
        };
        let p = create_provider("gemini", &cfg).unwrap();
        assert_eq!(p.name(), "gemini");
    }

    #[test]
    fn factory_rejects_unknown() {
        let cfg = ProviderConfig {
            api_key: "key".to_string(),
            model: "model".to_string(),
            endpoint: "https://example.com".to_string(),
        };
        assert!(create_provider("unknown", &cfg).is_err());
    }
}

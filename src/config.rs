/// Server configuration parsed from environment variables.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Server listen port.
    pub port: u16,
    /// Server bind host.
    pub host: String,
    /// Default AI difficulty when not specified in request.
    pub default_difficulty: String,
    /// AI move timeout in milliseconds.
    pub ai_timeout_ms: u64,
    /// LLM chat configuration.
    pub llm: LlmConfig,
}

/// LLM provider configuration.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Whether LLM chat is enabled.
    pub enabled: bool,
    /// Default LLM provider name (openai, anthropic, gemini, xai, deepseek).
    pub provider: String,
    /// OpenAI configuration.
    pub openai: ProviderConfig,
    /// Anthropic configuration.
    pub anthropic: ProviderConfig,
    /// Gemini configuration.
    pub gemini: ProviderConfig,
    /// xAI configuration.
    pub xai: ProviderConfig,
    /// DeepSeek configuration.
    pub deepseek: ProviderConfig,
}

/// Configuration for a single LLM provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// API key (empty = disabled).
    pub api_key: String,
    /// Model name override.
    pub model: String,
    /// API endpoint override.
    pub endpoint: String,
}

impl LlmConfig {
    /// Load LLM configuration from environment variables.
    pub fn from_env() -> Self {
        let provider = std::env::var("CHESS_LLM_PROVIDER")
            .unwrap_or_else(|_| "openai".to_string())
            .to_lowercase();

        LlmConfig {
            enabled: std::env::var("CHESS_LLM_ENABLED")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            provider,
            openai: ProviderConfig {
                api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
                model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
                endpoint: std::env::var("OPENAI_ENDPOINT")
                    .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string()),
            },
            anthropic: ProviderConfig {
                api_key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
                model: std::env::var("ANTHROPIC_MODEL")
                    .unwrap_or_else(|_| "claude-3-haiku-20240307".to_string()),
                endpoint: std::env::var("ANTHROPIC_ENDPOINT")
                    .unwrap_or_else(|_| "https://api.anthropic.com/v1/messages".to_string()),
            },
            gemini: ProviderConfig {
                api_key: std::env::var("GEMINI_API_KEY").unwrap_or_default(),
                model: std::env::var("GEMINI_MODEL")
                    .unwrap_or_else(|_| "gemini-1.5-flash".to_string()),
                endpoint: std::env::var("GEMINI_ENDPOINT").unwrap_or_else(|_| {
                    "https://generativelanguage.googleapis.com/v1beta/models".to_string()
                }),
            },
            xai: ProviderConfig {
                api_key: std::env::var("XAI_API_KEY").unwrap_or_default(),
                model: std::env::var("XAI_MODEL").unwrap_or_else(|_| "grok-1.5".to_string()),
                endpoint: std::env::var("XAI_ENDPOINT")
                    .unwrap_or_else(|_| "https://api.x.ai/v1/chat/completions".to_string()),
            },
            deepseek: ProviderConfig {
                api_key: std::env::var("DEEPSEEK_API_KEY").unwrap_or_default(),
                model: std::env::var("DEEPSEEK_MODEL")
                    .unwrap_or_else(|_| "deepseek-chat".to_string()),
                endpoint: std::env::var("DEEPSEEK_ENDPOINT")
                    .unwrap_or_else(|_| "https://api.deepseek.com/v1/chat/completions".to_string()),
            },
        }
    }

    /// Auto-detect the best available provider by checking which API keys are set.
    pub fn auto_detect_provider(&self) -> Option<&str> {
        if !self.openai.api_key.is_empty() {
            return Some("openai");
        }
        if !self.anthropic.api_key.is_empty() {
            return Some("anthropic");
        }
        if !self.gemini.api_key.is_empty() {
            return Some("gemini");
        }
        if !self.xai.api_key.is_empty() {
            return Some("xai");
        }
        if !self.deepseek.api_key.is_empty() {
            return Some("deepseek");
        }
        None
    }

    /// Get the provider config for a given provider name.
    pub fn provider_config(&self, name: &str) -> Option<&ProviderConfig> {
        match name {
            "openai" => Some(&self.openai),
            "anthropic" => Some(&self.anthropic),
            "gemini" => Some(&self.gemini),
            "xai" => Some(&self.xai),
            "deepseek" => Some(&self.deepseek),
            _ => None,
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        LlmConfig {
            enabled: false,
            provider: "openai".to_string(),
            openai: ProviderConfig {
                api_key: String::new(),
                model: "gpt-4o-mini".to_string(),
                endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            },
            anthropic: ProviderConfig {
                api_key: String::new(),
                model: "claude-3-haiku-20240307".to_string(),
                endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            },
            gemini: ProviderConfig {
                api_key: String::new(),
                model: "gemini-1.5-flash".to_string(),
                endpoint: "https://generativelanguage.googleapis.com/v1beta/models".to_string(),
            },
            xai: ProviderConfig {
                api_key: String::new(),
                model: "grok-1.5".to_string(),
                endpoint: "https://api.x.ai/v1/chat/completions".to_string(),
            },
            deepseek: ProviderConfig {
                api_key: String::new(),
                model: "deepseek-chat".to_string(),
                endpoint: "https://api.deepseek.com/v1/chat/completions".to_string(),
            },
        }
    }
}

impl AppConfig {
    /// Load configuration from environment variables with defaults.
    pub fn from_env() -> Self {
        AppConfig {
            port: std::env::var("PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8082),
            host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            default_difficulty: std::env::var("CHESS_AI_DEFAULT_DIFFICULTY")
                .unwrap_or_else(|_| "medium".to_string()),
            ai_timeout_ms: std::env::var("CHESS_AI_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5000),
            llm: LlmConfig::from_env(),
        }
    }

    /// Socket address string for binding.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            port: 8082,
            host: "0.0.0.0".to_string(),
            default_difficulty: "medium".to_string(),
            ai_timeout_ms: 5000,
            llm: LlmConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = AppConfig::default();
        assert_eq!(config.port, 8082);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.default_difficulty, "medium");
        assert_eq!(config.ai_timeout_ms, 5000);
        assert_eq!(config.bind_addr(), "0.0.0.0:8082");
    }

    #[test]
    fn from_env_defaults() {
        // Without setting env vars, should fall back to defaults
        let config = AppConfig::from_env();
        assert_eq!(config.port, 8082);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn llm_config_defaults() {
        let llm = LlmConfig::default();
        assert!(!llm.enabled);
        assert_eq!(llm.provider, "openai");
        assert!(llm.openai.api_key.is_empty());
        assert_eq!(llm.openai.model, "gpt-4o-mini");
        assert_eq!(llm.anthropic.model, "claude-3-haiku-20240307");
        assert_eq!(llm.gemini.model, "gemini-1.5-flash");
        assert_eq!(llm.xai.model, "grok-1.5");
        assert_eq!(llm.deepseek.model, "deepseek-chat");
    }

    #[test]
    fn llm_auto_detect_no_keys() {
        let llm = LlmConfig::default();
        assert!(llm.auto_detect_provider().is_none());
    }

    #[test]
    fn llm_auto_detect_openai() {
        let mut llm = LlmConfig::default();
        llm.openai.api_key = "sk-test".to_string();
        assert_eq!(llm.auto_detect_provider(), Some("openai"));
    }

    #[test]
    fn llm_auto_detect_anthropic() {
        let mut llm = LlmConfig::default();
        llm.anthropic.api_key = "sk-ant-test".to_string();
        assert_eq!(llm.auto_detect_provider(), Some("anthropic"));
    }

    #[test]
    fn llm_provider_config_lookup() {
        let llm = LlmConfig::default();
        assert!(llm.provider_config("openai").is_some());
        assert!(llm.provider_config("anthropic").is_some());
        assert!(llm.provider_config("gemini").is_some());
        assert!(llm.provider_config("xai").is_some());
        assert!(llm.provider_config("deepseek").is_some());
        assert!(llm.provider_config("unknown").is_none());
    }
}

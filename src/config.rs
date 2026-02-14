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
}

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Conversation tracking
// ---------------------------------------------------------------------------

/// A conversation associated with a game.
#[derive(Debug, Clone)]
pub struct Conversation {
    pub game_id: String,
    pub messages: Vec<Message>,
    pub context: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: MessageType,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub game_state: Option<HashMap<String, serde_json::Value>>,
    pub timestamp: DateTime<Utc>,
}

/// Message author type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    User,
    Ai,
    System,
}

impl MessageType {
    pub fn as_str(self) -> &'static str {
        match self {
            MessageType::User => "user",
            MessageType::Ai => "ai",
            MessageType::System => "system",
        }
    }
}

// ---------------------------------------------------------------------------
// Move context (passed from game state into chat)
// ---------------------------------------------------------------------------

/// Context about the current game state, used to enrich LLM prompts.
#[derive(Debug, Clone, Default)]
pub struct MoveContext {
    pub last_move: String,
    pub move_count: usize,
    pub current_player: String,
    pub game_status: String,
    pub position_fen: String,
    pub legal_moves: Vec<String>,
    pub in_check: bool,
    pub captured_piece: Option<String>,
}

// ---------------------------------------------------------------------------
// Chat request / response (internal service types)
// ---------------------------------------------------------------------------

/// Internal chat request used by the service (not the API model).
#[derive(Debug)]
pub struct ChatInput {
    pub game_id: String,
    pub message: String,
    pub move_data: Option<MoveContext>,
    pub provider: Option<String>,
    pub api_key: Option<String>,
}

/// Internal chat response returned by the service.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatOutput {
    pub message: String,
    pub message_id: String,
    pub personality: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub game_context: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestions: Option<Vec<String>>,
    pub timestamp: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// LLM errors
// ---------------------------------------------------------------------------

/// Errors that can occur in the chat / LLM subsystem.
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("LLM chat is not enabled")]
    Disabled,

    #[error("unsupported provider: {0}")]
    UnsupportedProvider(String),

    #[error("no API key configured for provider: {0}")]
    MissingApiKey(String),

    #[error("LLM request failed: {0}")]
    RequestFailed(String),

    #[error("failed to parse LLM response: {0}")]
    ParseError(String),

    #[error("provider error: {0}")]
    ProviderError(String),
}

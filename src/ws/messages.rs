//! WebSocket message types for real-time game events.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Server → Client events
// ---------------------------------------------------------------------------

/// Envelope sent from server to every subscribed WebSocket client.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WsEvent {
    /// Discriminator so clients can switch on event type.
    #[serde(rename = "type")]
    pub event_type: WsEventType,
    /// Event-specific payload.
    #[serde(flatten)]
    pub payload: WsPayload,
}

/// Event type discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WsEventType {
    GameState,
    MoveMade,
    AiThinking,
    AiMoveComplete,
    GameOver,
    Error,
    Pong,
    Subscribed,
}

/// Event payload variants.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum WsPayload {
    GameState(GameStatePayload),
    MoveMade(MoveMadePayload),
    AiThinking(AiThinkingPayload),
    AiMoveComplete(AiMoveCompletePayload),
    GameOver(GameOverPayload),
    Error(ErrorPayload),
    Pong(PongPayload),
    Subscribed(SubscribedPayload),
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStatePayload {
    pub game_id: String,
    pub fen: String,
    pub status: String,
    pub current_player: String,
    pub move_count: usize,
    pub check: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveMadePayload {
    pub game_id: String,
    pub san: String,
    pub from: String,
    pub to: String,
    pub player: String,
    pub fen: String,
    pub status: String,
    pub move_count: usize,
    pub check: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiThinkingPayload {
    pub game_id: String,
    pub difficulty: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiMoveCompletePayload {
    pub game_id: String,
    pub san: String,
    pub from: String,
    pub to: String,
    pub fen: String,
    pub status: String,
    pub thinking_time_ms: u64,
    pub move_count: usize,
    pub check: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GameOverPayload {
    pub game_id: String,
    pub result: String,
    pub fen: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorPayload {
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PongPayload {
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscribedPayload {
    pub game_id: String,
    #[serde(flatten)]
    pub state: GameStatePayload,
}

// ---------------------------------------------------------------------------
// Client → Server commands
// ---------------------------------------------------------------------------

/// Commands sent from client to server over WebSocket.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsCommand {
    Subscribe { game_id: String },
    Unsubscribe { game_id: String },
    Ping,
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

impl WsEvent {
    pub fn game_state(
        game_id: &str,
        fen: &str,
        status: &str,
        player: &str,
        moves: usize,
        check: bool,
    ) -> Self {
        WsEvent {
            event_type: WsEventType::GameState,
            payload: WsPayload::GameState(GameStatePayload {
                game_id: game_id.to_string(),
                fen: fen.to_string(),
                status: status.to_string(),
                current_player: player.to_string(),
                move_count: moves,
                check,
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn move_made(
        game_id: &str,
        san: &str,
        from: &str,
        to: &str,
        player: &str,
        fen: &str,
        status: &str,
        moves: usize,
        check: bool,
    ) -> Self {
        WsEvent {
            event_type: WsEventType::MoveMade,
            payload: WsPayload::MoveMade(MoveMadePayload {
                game_id: game_id.to_string(),
                san: san.to_string(),
                from: from.to_string(),
                to: to.to_string(),
                player: player.to_string(),
                fen: fen.to_string(),
                status: status.to_string(),
                move_count: moves,
                check,
            }),
        }
    }

    pub fn ai_thinking(game_id: &str, difficulty: &str) -> Self {
        WsEvent {
            event_type: WsEventType::AiThinking,
            payload: WsPayload::AiThinking(AiThinkingPayload {
                game_id: game_id.to_string(),
                difficulty: difficulty.to_string(),
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ai_move_complete(
        game_id: &str,
        san: &str,
        from: &str,
        to: &str,
        fen: &str,
        status: &str,
        thinking_time_ms: u64,
        moves: usize,
        check: bool,
    ) -> Self {
        WsEvent {
            event_type: WsEventType::AiMoveComplete,
            payload: WsPayload::AiMoveComplete(AiMoveCompletePayload {
                game_id: game_id.to_string(),
                san: san.to_string(),
                from: from.to_string(),
                to: to.to_string(),
                fen: fen.to_string(),
                status: status.to_string(),
                thinking_time_ms,
                move_count: moves,
                check,
            }),
        }
    }

    pub fn game_over(game_id: &str, result: &str, fen: &str) -> Self {
        WsEvent {
            event_type: WsEventType::GameOver,
            payload: WsPayload::GameOver(GameOverPayload {
                game_id: game_id.to_string(),
                result: result.to_string(),
                fen: fen.to_string(),
            }),
        }
    }

    pub fn error(message: &str) -> Self {
        WsEvent {
            event_type: WsEventType::Error,
            payload: WsPayload::Error(ErrorPayload {
                message: message.to_string(),
            }),
        }
    }

    pub fn pong() -> Self {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        WsEvent {
            event_type: WsEventType::Pong,
            payload: WsPayload::Pong(PongPayload { timestamp: ts }),
        }
    }

    pub fn subscribed(
        game_id: &str,
        fen: &str,
        status: &str,
        player: &str,
        moves: usize,
        check: bool,
    ) -> Self {
        WsEvent {
            event_type: WsEventType::Subscribed,
            payload: WsPayload::Subscribed(SubscribedPayload {
                game_id: game_id.to_string(),
                state: GameStatePayload {
                    game_id: game_id.to_string(),
                    fen: fen.to_string(),
                    status: status.to_string(),
                    current_player: player.to_string(),
                    move_count: moves,
                    check,
                },
            }),
        }
    }

    /// Serialize to JSON text for sending over WebSocket.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self)
            .unwrap_or_else(|_| r#"{"type":"error","message":"serialization failed"}"#.to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_state_event_serializes() {
        let evt = WsEvent::game_state(
            "g1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "active",
            "white",
            0,
            false,
        );
        let json = evt.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "game_state");
        assert_eq!(parsed["gameId"], "g1");
        assert_eq!(parsed["currentPlayer"], "white");
    }

    #[test]
    fn move_made_event_serializes() {
        let evt = WsEvent::move_made(
            "g1", "e4", "e2", "e4", "white", "fen...", "active", 1, false,
        );
        let json = evt.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "move_made");
        assert_eq!(parsed["san"], "e4");
        assert_eq!(parsed["player"], "white");
    }

    #[test]
    fn ai_thinking_event_serializes() {
        let evt = WsEvent::ai_thinking("g1", "hard");
        let json = evt.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "ai_thinking");
        assert_eq!(parsed["difficulty"], "hard");
    }

    #[test]
    fn ai_move_complete_event_serializes() {
        let evt =
            WsEvent::ai_move_complete("g1", "Nf3", "g1", "f3", "fen...", "active", 150, 2, false);
        let json = evt.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "ai_move_complete");
        assert_eq!(parsed["thinkingTimeMs"], 150);
    }

    #[test]
    fn game_over_event_serializes() {
        let evt = WsEvent::game_over("g1", "checkmate", "fen...");
        let json = evt.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "game_over");
        assert_eq!(parsed["result"], "checkmate");
    }

    #[test]
    fn error_event_serializes() {
        let evt = WsEvent::error("something went wrong");
        let json = evt.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["message"], "something went wrong");
    }

    #[test]
    fn pong_event_serializes() {
        let evt = WsEvent::pong();
        let json = evt.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "pong");
        assert!(parsed["timestamp"].is_number());
    }

    #[test]
    fn subscribed_event_serializes() {
        let evt = WsEvent::subscribed("g1", "startfen", "active", "white", 0, false);
        let json = evt.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "subscribed");
        assert_eq!(parsed["gameId"], "g1");
    }

    #[test]
    fn ws_command_subscribe_deserializes() {
        let json = r#"{"type":"subscribe","game_id":"g1"}"#;
        let cmd: WsCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, WsCommand::Subscribe { game_id } if game_id == "g1"));
    }

    #[test]
    fn ws_command_unsubscribe_deserializes() {
        let json = r#"{"type":"unsubscribe","game_id":"g1"}"#;
        let cmd: WsCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, WsCommand::Unsubscribe { game_id } if game_id == "g1"));
    }

    #[test]
    fn ws_command_ping_deserializes() {
        let json = r#"{"type":"ping"}"#;
        let cmd: WsCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, WsCommand::Ping));
    }

    #[test]
    fn event_type_values() {
        assert_eq!(WsEventType::GameState, WsEventType::GameState);
        assert_ne!(WsEventType::GameState, WsEventType::MoveMade);
    }
}

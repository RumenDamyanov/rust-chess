use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use chrono::Utc;

use crate::config::LlmConfig;

use super::providers::{self, LlmProvider};
use super::types::*;

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = r#"You are a friendly chess opponent and AI coach. Your personality traits:
- Encouraging and supportive, but honest about mistakes
- Knowledgeable about chess openings, tactics, and strategy
- You use chess terminology naturally (pins, forks, discovered attacks, etc.)
- You keep responses concise (1-3 sentences) unless asked for detail
- You occasionally use chess-related emoji (♟️ ♞ ♛ ⚡ 🎯)
- You celebrate good moves and gently suggest improvements for weak ones

Guidelines:
- Always be encouraging, even when pointing out mistakes
- Reference specific chess concepts when relevant
- Keep responses conversational, not lecture-like
- If asked about non-chess topics, politely redirect to the game
- When discussing positions, reference specific squares and pieces
- Adapt your language to the apparent skill level of the player"#;

// ---------------------------------------------------------------------------
// Welcome messages
// ---------------------------------------------------------------------------

const WELCOME_MESSAGES: &[&str] = &[
    "Hello! I'm your AI chess companion. Ready for a great game? ♟️",
    "Welcome to our chess match! I'm excited to play and chat with you. 🎯",
    "Hi there! Let's have some fun with chess. Feel free to ask me anything about the game! ♞",
    "Greetings, chess friend! I'm here to play, chat, and maybe share some chess wisdom. 🤔",
    "Hello! Ready to make some great moves? I love discussing chess strategy and tactics! ⚡",
];

// ---------------------------------------------------------------------------
// ChatService
// ---------------------------------------------------------------------------

/// The chat service manages conversations and routes messages to LLM providers.
pub struct ChatService {
    config: LlmConfig,
    conversations: RwLock<HashMap<String, Conversation>>,
}

impl ChatService {
    /// Create a new `ChatService`.
    pub fn new(config: LlmConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            conversations: RwLock::new(HashMap::new()),
        })
    }

    /// Whether LLM chat is available (enabled + at least one provider key set).
    pub fn is_available(&self) -> bool {
        self.config.enabled && self.config.auto_detect_provider().is_some()
    }

    // -----------------------------------------------------------------------
    // Conversation management
    // -----------------------------------------------------------------------

    /// Get or create a conversation for a game, returning a welcome message
    /// the first time.
    pub async fn ensure_conversation(&self, game_id: &str) -> (Conversation, bool) {
        let mut convos = self.conversations.write().await;
        if let Some(c) = convos.get(game_id) {
            return (c.clone(), false);
        }

        let now = Utc::now();
        let welcome = self.generate_welcome();
        let msg = Message {
            id: format!("ai_{}_{}", game_id, now.timestamp_nanos_opt().unwrap_or(0)),
            msg_type: MessageType::Ai,
            content: welcome,
            game_state: None,
            timestamp: now,
        };

        let conversation = Conversation {
            game_id: game_id.to_string(),
            messages: vec![msg],
            context: HashMap::new(),
            created_at: now,
            updated_at: now,
        };

        convos.insert(game_id.to_string(), conversation.clone());
        (conversation, true)
    }

    /// Return the conversation for a game (if any).
    pub async fn get_conversation(&self, game_id: &str) -> Option<Conversation> {
        self.conversations.read().await.get(game_id).cloned()
    }

    /// Clear a conversation.
    pub async fn clear_conversation(&self, game_id: &str) {
        self.conversations.write().await.remove(game_id);
    }

    // -----------------------------------------------------------------------
    // Chat
    // -----------------------------------------------------------------------

    /// Process a user chat message and return an AI response.
    pub async fn chat(&self, input: ChatInput) -> Result<ChatOutput, ChatError> {
        if !self.config.enabled {
            return Err(ChatError::Disabled);
        }

        // Ensure conversation exists.
        let (mut conversation, _new) = self.ensure_conversation(&input.game_id).await;

        // Record user message.
        let user_msg_id = self.add_message(
            &input.game_id,
            MessageType::User,
            &input.message,
            &input.move_data,
        )
        .await;

        // Reload after mutation.
        if let Some(c) = self.get_conversation(&input.game_id).await {
            conversation = c;
        }

        // Build the contextual prompt.
        let prompt =
            Self::build_contextual_message(&input.message, &conversation, &input.move_data);

        // Resolve provider.
        let provider = self.resolve_provider(input.provider.as_deref(), input.api_key.as_deref())?;

        // Ask LLM.
        let raw_response = provider.ask(SYSTEM_PROMPT, &prompt).await?;
        let clean = Self::clean_response(&raw_response);

        // Record AI response.
        let _ai_msg_id =
            self.add_message(&input.game_id, MessageType::Ai, &clean, &None).await;

        let suggestions = Self::generate_suggestions(&input.move_data);

        Ok(ChatOutput {
            message: clean,
            message_id: user_msg_id,
            personality: "friendly_chess_coach".to_string(),
            game_context: Self::build_game_context(&input.move_data),
            suggestions: if suggestions.is_empty() {
                None
            } else {
                Some(suggestions)
            },
            timestamp: Utc::now(),
        })
    }

    /// Generate an AI reaction to a move.
    pub async fn react_to_move(
        &self,
        game_id: &str,
        chess_move: &str,
        move_data: &MoveContext,
        provider_name: Option<&str>,
        api_key: Option<&str>,
    ) -> Result<ChatOutput, ChatError> {
        if !self.config.enabled {
            return Err(ChatError::Disabled);
        }

        // Ensure conversation.
        let (_conversation, _new) = self.ensure_conversation(game_id).await;

        // Build reaction prompt.
        let prompt = Self::build_move_reaction_prompt(chess_move, move_data);

        let provider = self.resolve_provider(provider_name, api_key)?;
        let raw = provider.ask(SYSTEM_PROMPT, &prompt).await?;
        let clean = Self::clean_response(&raw);

        // Record AI reaction.
        let msg_id = self
            .add_message(game_id, MessageType::Ai, &clean, &Some(move_data.clone()))
            .await;

        Ok(ChatOutput {
            message: clean,
            message_id: msg_id,
            personality: "observant_chess_coach".to_string(),
            game_context: Self::build_game_context(&Some(move_data.clone())),
            suggestions: None,
            timestamp: Utc::now(),
        })
    }

    /// General chat (no game context).
    pub async fn general_chat(
        &self,
        message: &str,
        provider_name: Option<&str>,
        api_key: Option<&str>,
    ) -> Result<ChatOutput, ChatError> {
        if !self.config.enabled {
            return Err(ChatError::Disabled);
        }

        let provider = self.resolve_provider(provider_name, api_key)?;
        let raw = provider.ask(SYSTEM_PROMPT, message).await?;
        let clean = Self::clean_response(&raw);

        Ok(ChatOutput {
            message: clean,
            message_id: format!("general_{}", Utc::now().timestamp_millis()),
            personality: "friendly_chess_coach".to_string(),
            game_context: None,
            suggestions: Some(vec![
                "What's the best opening for beginners?".to_string(),
                "Explain the Sicilian Defense".to_string(),
                "How do I improve my endgame?".to_string(),
            ]),
            timestamp: Utc::now(),
        })
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// Resolve the LLM provider to use for a request.
    fn resolve_provider(
        &self,
        requested: Option<&str>,
        api_key: Option<&str>,
    ) -> Result<Box<dyn LlmProvider>, ChatError> {
        // If per-request provider + key are provided, use those.
        if let (Some(name), Some(key)) = (requested, api_key) {
            let mut cfg = self
                .config
                .provider_config(name)
                .cloned()
                .unwrap_or_else(|| crate::config::ProviderConfig {
                    api_key: String::new(),
                    model: String::new(),
                    endpoint: String::new(),
                });
            cfg.api_key = key.to_string();

            // Fill defaults if model/endpoint are empty.
            if cfg.model.is_empty() || cfg.endpoint.is_empty() {
                if let Some(default_cfg) = self.config.provider_config(name) {
                    if cfg.model.is_empty() {
                        cfg.model = default_cfg.model.clone();
                    }
                    if cfg.endpoint.is_empty() {
                        cfg.endpoint = default_cfg.endpoint.clone();
                    }
                }
            }

            return providers::create_provider(name, &cfg);
        }

        // Otherwise use default configured provider (or auto-detect).
        let provider_name = if !self.config.provider.is_empty() {
            // Check if the configured provider actually has a key.
            if let Some(cfg) = self.config.provider_config(&self.config.provider) {
                if !cfg.api_key.is_empty() {
                    self.config.provider.as_str()
                } else {
                    self.config
                        .auto_detect_provider()
                        .ok_or(ChatError::MissingApiKey(self.config.provider.clone()))?
                }
            } else {
                self.config
                    .auto_detect_provider()
                    .ok_or(ChatError::UnsupportedProvider(self.config.provider.clone()))?
            }
        } else {
            self.config
                .auto_detect_provider()
                .ok_or(ChatError::MissingApiKey("(none)".to_string()))?
        };

        let cfg = self
            .config
            .provider_config(provider_name)
            .ok_or_else(|| ChatError::UnsupportedProvider(provider_name.to_string()))?;

        providers::create_provider(provider_name, cfg)
    }

    /// Record a message in a conversation, returning the message ID.
    async fn add_message(
        &self,
        game_id: &str,
        msg_type: MessageType,
        content: &str,
        move_data: &Option<MoveContext>,
    ) -> String {
        let now = Utc::now();
        let msg_id = format!(
            "{}_{}_{}", msg_type.as_str(), game_id, now.timestamp_nanos_opt().unwrap_or(0)
        );

        let game_state = move_data.as_ref().map(|md| {
            let mut map = HashMap::new();
            map.insert(
                "last_move".to_string(),
                serde_json::Value::String(md.last_move.clone()),
            );
            map.insert(
                "move_count".to_string(),
                serde_json::json!(md.move_count),
            );
            map.insert(
                "current_player".to_string(),
                serde_json::Value::String(md.current_player.clone()),
            );
            map.insert(
                "game_status".to_string(),
                serde_json::Value::String(md.game_status.clone()),
            );
            map.insert(
                "position".to_string(),
                serde_json::Value::String(md.position_fen.clone()),
            );
            map
        });

        let message = Message {
            id: msg_id.clone(),
            msg_type,
            content: content.to_string(),
            game_state,
            timestamp: now,
        };

        let mut convos = self.conversations.write().await;
        if let Some(c) = convos.get_mut(game_id) {
            c.messages.push(message);
            c.updated_at = now;
        }

        msg_id
    }

    /// Build a contextual message that includes game state and recent history.
    fn build_contextual_message(
        user_message: &str,
        conversation: &Conversation,
        move_data: &Option<MoveContext>,
    ) -> String {
        let mut ctx = String::new();

        // Game context.
        if let Some(md) = move_data {
            let pos_preview = if md.position_fen.len() > 20 {
                format!("{}...", &md.position_fen[..20])
            } else {
                md.position_fen.clone()
            };
            ctx.push_str(&format!(
                "[Game Context: Move {}, {} to play, Position: {}]",
                md.move_count, md.current_player, pos_preview
            ));
            if !md.last_move.is_empty() {
                ctx.push_str(&format!(" [Last move: {}]", md.last_move));
            }
            ctx.push_str("\n\n");
        }

        // Recent conversation (last 3 exchanges = 6 messages).
        let msgs = &conversation.messages;
        let recent = if msgs.len() > 6 {
            &msgs[msgs.len() - 6..]
        } else {
            msgs
        };

        if !recent.is_empty() {
            ctx.push_str("[Recent conversation:\n");
            for m in recent {
                match m.msg_type {
                    MessageType::User => ctx.push_str(&format!("Human: {}\n", m.content)),
                    MessageType::Ai => ctx.push_str(&format!("Assistant: {}\n", m.content)),
                    MessageType::System => {}
                }
            }
            ctx.push_str("]\n\n");
        }

        ctx.push_str(&format!("Human: {}", user_message));
        ctx
    }

    /// Build a prompt asking the LLM to react to a specific move.
    fn build_move_reaction_prompt(chess_move: &str, md: &MoveContext) -> String {
        format!(
            "[Game Context: Move {}, {} just played {}, Status: {}]\n\n\
             Please give a brief, encouraging reaction to this chess move. Consider:\n\
             - Is this a good opening move, tactical shot, or strategic decision?\n\
             - Should you congratulate, encourage, or gently suggest improvements?\n\
             - Keep it conversational and positive\n\
             - 1-2 sentences maximum\n\n\
             The move played was: {}",
            md.move_count, md.current_player, chess_move, md.game_status, chess_move
        )
    }

    /// Clean up LLM response artifacts.
    fn clean_response(raw: &str) -> String {
        let mut cleaned = raw.trim().to_string();

        // Remove bracketed context that might leak through.
        if cleaned.starts_with('[') {
            if let Some(idx) = cleaned.find(']') {
                if idx < 50 {
                    cleaned = cleaned[idx + 1..].trim().to_string();
                }
            }
        }

        // Remove prefix artifacts.
        for prefix in &["Assistant:", "AI:", "Bot:", "Chatbot:"] {
            if let Some(rest) = cleaned.strip_prefix(prefix) {
                cleaned = rest.trim().to_string();
            }
        }

        // Truncate to 280 chars at sentence boundary.
        if cleaned.len() > 280 {
            let slice = &cleaned[..280];
            let cut = slice
                .rfind('.')
                .filter(|&i| i > 100)
                .or_else(|| slice.rfind('!').filter(|&i| i > 100))
                .or_else(|| slice.rfind('?').filter(|&i| i > 100));
            if let Some(idx) = cut {
                cleaned = cleaned[..=idx].to_string();
            } else {
                cleaned = format!("{}...", &cleaned[..277]);
            }
        }

        cleaned
    }

    /// Generate follow-up suggestions based on game state.
    fn generate_suggestions(move_data: &Option<MoveContext>) -> Vec<String> {
        let mut suggestions = vec![
            "What do you think about this position?".to_string(),
            "Any tips for improvement?".to_string(),
            "What's your favorite opening?".to_string(),
        ];

        if let Some(md) = move_data {
            if md.move_count < 10 {
                suggestions.push("Tell me about this opening".to_string());
            } else if md.move_count > 30 {
                suggestions.push("How's my endgame technique?".to_string());
            } else {
                suggestions.push("Any tactical opportunities here?".to_string());
            }
        }

        suggestions.truncate(3);
        suggestions
    }

    /// Build a game-context map for the API response.
    fn build_game_context(
        move_data: &Option<MoveContext>,
    ) -> Option<HashMap<String, serde_json::Value>> {
        let md = move_data.as_ref()?;

        let mut ctx = HashMap::new();
        ctx.insert(
            "move_count".to_string(),
            serde_json::json!(md.move_count),
        );
        ctx.insert(
            "current_player".to_string(),
            serde_json::Value::String(md.current_player.clone()),
        );
        ctx.insert(
            "game_status".to_string(),
            serde_json::Value::String(md.game_status.clone()),
        );
        ctx.insert(
            "game_phase".to_string(),
            serde_json::Value::String(Self::determine_game_phase(md.move_count).to_string()),
        );
        ctx.insert(
            "position_fen".to_string(),
            serde_json::Value::String(md.position_fen.clone()),
        );

        if !md.last_move.is_empty() {
            ctx.insert(
                "last_move".to_string(),
                serde_json::Value::String(md.last_move.clone()),
            );
        }
        if !md.legal_moves.is_empty() {
            ctx.insert(
                "legal_moves_count".to_string(),
                serde_json::json!(md.legal_moves.len()),
            );
            let sample: Vec<_> = md.legal_moves.iter().take(5).cloned().collect();
            ctx.insert(
                "sample_legal_moves".to_string(),
                serde_json::json!(sample),
            );
        }
        if md.in_check {
            ctx.insert("in_check".to_string(), serde_json::json!(true));
        }
        if let Some(ref cap) = md.captured_piece {
            ctx.insert(
                "captured_piece".to_string(),
                serde_json::Value::String(cap.clone()),
            );
        }

        Some(ctx)
    }

    fn determine_game_phase(move_count: usize) -> &'static str {
        if move_count < 15 {
            "opening"
        } else if move_count < 35 {
            "middlegame"
        } else {
            "endgame"
        }
    }

    fn generate_welcome(&self) -> String {
        let idx = (Utc::now().timestamp() as usize) % WELCOME_MESSAGES.len();
        WELCOME_MESSAGES[idx].to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlmConfig;

    fn test_config() -> LlmConfig {
        let mut cfg = LlmConfig::default();
        cfg.enabled = true;
        cfg.openai.api_key = "sk-test".to_string();
        cfg
    }

    fn disabled_config() -> LlmConfig {
        LlmConfig::default() // enabled = false
    }

    #[test]
    fn service_is_available_when_enabled_and_keyed() {
        let svc = ChatService::new(test_config());
        assert!(svc.is_available());
    }

    #[test]
    fn service_not_available_when_disabled() {
        let svc = ChatService::new(disabled_config());
        assert!(!svc.is_available());
    }

    #[test]
    fn service_not_available_no_keys() {
        let mut cfg = LlmConfig::default();
        cfg.enabled = true;
        let svc = ChatService::new(cfg);
        assert!(!svc.is_available());
    }

    #[tokio::test]
    async fn ensure_conversation_creates_new() {
        let svc = ChatService::new(test_config());
        let (conv, is_new) = svc.ensure_conversation("game-1").await;
        assert!(is_new);
        assert_eq!(conv.game_id, "game-1");
        assert_eq!(conv.messages.len(), 1); // welcome message
        assert_eq!(conv.messages[0].msg_type, MessageType::Ai);
    }

    #[tokio::test]
    async fn ensure_conversation_returns_existing() {
        let svc = ChatService::new(test_config());
        svc.ensure_conversation("game-1").await;
        let (conv, is_new) = svc.ensure_conversation("game-1").await;
        assert!(!is_new);
        assert_eq!(conv.game_id, "game-1");
    }

    #[tokio::test]
    async fn clear_conversation_removes() {
        let svc = ChatService::new(test_config());
        svc.ensure_conversation("game-1").await;
        svc.clear_conversation("game-1").await;
        assert!(svc.get_conversation("game-1").await.is_none());
    }

    #[tokio::test]
    async fn chat_returns_disabled_error() {
        let svc = ChatService::new(disabled_config());
        let result = svc
            .chat(ChatInput {
                game_id: "g1".to_string(),
                message: "hello".to_string(),
                move_data: None,
                provider: None,
                api_key: None,
            })
            .await;
        assert!(matches!(result, Err(ChatError::Disabled)));
    }

    #[tokio::test]
    async fn react_returns_disabled_error() {
        let svc = ChatService::new(disabled_config());
        let md = MoveContext::default();
        let result = svc.react_to_move("g1", "e2e4", &md, None, None).await;
        assert!(matches!(result, Err(ChatError::Disabled)));
    }

    #[tokio::test]
    async fn general_chat_returns_disabled_error() {
        let svc = ChatService::new(disabled_config());
        let result = svc.general_chat("hi", None, None).await;
        assert!(matches!(result, Err(ChatError::Disabled)));
    }

    #[test]
    fn clean_response_trims_whitespace() {
        assert_eq!(ChatService::clean_response("  hello  "), "hello");
    }

    #[test]
    fn clean_response_strips_prefix() {
        assert_eq!(ChatService::clean_response("Assistant: hello"), "hello");
        assert_eq!(ChatService::clean_response("AI: hello"), "hello");
    }

    #[test]
    fn clean_response_removes_bracket_context() {
        assert_eq!(ChatService::clean_response("[ctx] hello"), "hello");
    }

    #[test]
    fn clean_response_truncates_long_text() {
        let long = "x".repeat(300);
        let cleaned = ChatService::clean_response(&long);
        assert!(cleaned.len() <= 280);
        assert!(cleaned.ends_with("..."));
    }

    #[test]
    fn clean_response_truncates_at_sentence_boundary() {
        let mut text = "A".repeat(120);
        text.push_str(". ");
        text.push_str(&"B".repeat(200));
        let cleaned = ChatService::clean_response(&text);
        assert!(cleaned.ends_with('.'));
        assert!(cleaned.len() <= 280);
    }

    #[test]
    fn generate_suggestions_opening() {
        let md = Some(MoveContext {
            move_count: 3,
            ..Default::default()
        });
        let s = ChatService::generate_suggestions(&md);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn generate_suggestions_endgame() {
        let md = Some(MoveContext {
            move_count: 40,
            ..Default::default()
        });
        let s = ChatService::generate_suggestions(&md);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn build_game_context_none() {
        assert!(ChatService::build_game_context(&None).is_none());
    }

    #[test]
    fn build_game_context_some() {
        let md = Some(MoveContext {
            move_count: 10,
            current_player: "white".to_string(),
            game_status: "active".to_string(),
            position_fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
                .to_string(),
            last_move: "e2e4".to_string(),
            legal_moves: vec!["e7e5".to_string(), "d7d5".to_string()],
            in_check: false,
            captured_piece: None,
        });
        let ctx = ChatService::build_game_context(&md).unwrap();
        assert_eq!(ctx["move_count"], serde_json::json!(10));
        assert_eq!(ctx["current_player"], "white");
        assert_eq!(ctx["game_phase"], "opening");
        assert!(ctx.contains_key("last_move"));
        assert!(ctx.contains_key("legal_moves_count"));
    }

    #[test]
    fn determine_game_phase_works() {
        assert_eq!(ChatService::determine_game_phase(5), "opening");
        assert_eq!(ChatService::determine_game_phase(20), "middlegame");
        assert_eq!(ChatService::determine_game_phase(50), "endgame");
    }

    #[test]
    fn build_contextual_message_with_move_data() {
        let conv = Conversation {
            game_id: "g1".to_string(),
            messages: vec![],
            context: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let md = Some(MoveContext {
            last_move: "e2e4".to_string(),
            move_count: 1,
            current_player: "black".to_string(),
            position_fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
                .to_string(),
            ..Default::default()
        });
        let msg = ChatService::build_contextual_message("What's a good response?", &conv, &md);
        assert!(msg.contains("[Game Context:"));
        assert!(msg.contains("[Last move: e2e4]"));
        assert!(msg.contains("Human: What's a good response?"));
    }

    #[test]
    fn build_move_reaction_prompt_includes_move() {
        let md = MoveContext {
            move_count: 5,
            current_player: "white".to_string(),
            game_status: "active".to_string(),
            ..Default::default()
        };
        let prompt = ChatService::build_move_reaction_prompt("Nf3", &md);
        assert!(prompt.contains("Nf3"));
        assert!(prompt.contains("Move 5"));
        assert!(prompt.contains("white"));
    }

    #[test]
    fn welcome_message_is_from_list() {
        let svc = ChatService::new(test_config());
        let welcome = svc.generate_welcome();
        assert!(WELCOME_MESSAGES.contains(&welcome.as_str()));
    }

    #[test]
    fn resolve_provider_returns_error_when_no_keys() {
        let mut cfg = LlmConfig::default();
        cfg.enabled = true;
        let svc = ChatService::new(cfg);
        let result = svc.resolve_provider(None, None);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_provider_with_override() {
        let svc = ChatService::new(test_config());
        let result = svc.resolve_provider(Some("openai"), Some("sk-override"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "openai");
    }

    #[test]
    fn resolve_provider_with_default_config() {
        let svc = ChatService::new(test_config());
        let result = svc.resolve_provider(None, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name(), "openai");
    }
}

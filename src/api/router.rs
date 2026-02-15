use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use super::handlers;
use super::state::SharedState;
use crate::ws;

/// Build the Axum router with all routes and middleware.
pub fn create_router(state: SharedState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // Health check (outside /api prefix)
        .route("/health", get(handlers::health))
        // Game CRUD
        .route(
            "/api/games",
            post(handlers::create_game).get(handlers::list_games),
        )
        .route(
            "/api/games/{id}",
            get(handlers::get_game).delete(handlers::delete_game),
        )
        // Move endpoints
        .route(
            "/api/games/{id}/moves",
            post(handlers::make_move).get(handlers::get_moves),
        )
        .route("/api/games/{id}/undo", post(handlers::undo_move))
        // AI endpoints
        .route("/api/games/{id}/ai-move", post(handlers::ai_move))
        .route("/api/games/{id}/ai-hint", post(handlers::ai_hint))
        // Query endpoints
        .route("/api/games/{id}/legal-moves", get(handlers::legal_moves))
        .route("/api/games/{id}/fen", post(handlers::load_fen))
        .route("/api/games/{id}/pgn", get(handlers::export_pgn))
        .route("/api/games/{id}/analysis", get(handlers::analysis))
        // Chat / LLM endpoints
        .route("/api/games/{id}/chat", post(handlers::chat_with_ai))
        .route("/api/games/{id}/react", post(handlers::react_to_move))
        .route("/api/chat", post(handlers::general_chat))
        .route("/api/chat/status", get(handlers::chat_status))
        // WebSocket — real-time game events
        .route("/ws/games/{id}", get(ws::ws_handler))
        // Middleware
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

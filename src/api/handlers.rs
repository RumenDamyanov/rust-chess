use std::time::Instant;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;

use crate::ai::{AiEngine, MinimaxAi};
use crate::engine::game::Game;
use crate::engine::san::move_to_san;
use crate::engine::types::{Difficulty, Move, Square};

use super::errors::ApiError;
use super::models::*;
use super::state::SharedState;

// =========================================================================
// Health
// =========================================================================

/// GET /health
pub async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    let uptime = state.start_time.elapsed().as_secs();
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        language: "rust".to_string(),
        engine: "rust-chess".to_string(),
        uptime,
    })
}

// =========================================================================
// Create Game
// =========================================================================

/// POST /api/games
pub async fn create_game(
    State(state): State<SharedState>,
    Json(input): Json<CreateGameRequest>,
) -> Result<(StatusCode, Json<GameResponse>), ApiError> {
    let mut game = if let Some(ref fen) = input.fen {
        Game::from_fen(fen).map_err(ApiError::from)?
    } else {
        Game::new()
    };

    game.white_player = input.white_player.unwrap_or_else(|| "White".into());
    game.black_player = input.black_player.unwrap_or_else(|| "Black".into());

    if let Some(ref diff_str) = input.ai_difficulty {
        game.ai_difficulty = Difficulty::from_str_loose(diff_str);
    }

    let response = game_to_response(&game);
    let id = game.id.clone();

    state.games.write().await.insert(id, game);

    Ok((StatusCode::CREATED, Json(response)))
}

// =========================================================================
// List Games
// =========================================================================

/// GET /api/games
pub async fn list_games(
    State(state): State<SharedState>,
    Query(query): Query<ListGamesQuery>,
) -> Json<ListGamesResponse> {
    let games = state.games.read().await;

    let limit = query.limit.unwrap_or(10).min(100);
    let offset = query.offset.unwrap_or(0);

    let mut filtered: Vec<&Game> = games.values().collect();

    // Filter by status if provided.
    if let Some(ref status_filter) = query.status {
        let sf = status_filter.to_lowercase();
        filtered.retain(|g| g.status().as_str() == sf);
    }

    let total = filtered.len();

    // Sort by created_at descending for consistent ordering.
    filtered.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    let page: Vec<GameResponse> = filtered
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(game_to_response)
        .collect();

    Json(ListGamesResponse {
        games: page,
        total,
        limit,
        offset,
    })
}

// =========================================================================
// Get Game
// =========================================================================

/// GET /api/games/:id
pub async fn get_game(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<GameResponse>, ApiError> {
    let games = state.games.read().await;
    let game = games
        .get(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;
    Ok(Json(game_to_response(game)))
}

// =========================================================================
// Delete Game
// =========================================================================

/// DELETE /api/games/:id
pub async fn delete_game(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<DeleteResponse>, ApiError> {
    let mut games = state.games.write().await;
    games
        .remove(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;
    Ok(Json(DeleteResponse {
        success: true,
        message: "Game deleted".to_string(),
    }))
}

// =========================================================================
// Make Move
// =========================================================================

/// POST /api/games/:id/moves
pub async fn make_move(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(input): Json<MoveRequest>,
) -> Result<Json<GameResponse>, ApiError> {
    let mut games = state.games.write().await;
    let game = games
        .get_mut(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

    let mv = resolve_move(game, &input)?;
    game.make_move(mv).map_err(ApiError::from)?;

    Ok(Json(game_to_response(game)))
}

// =========================================================================
// Get Move History
// =========================================================================

/// GET /api/games/:id/moves
pub async fn get_moves(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<MoveListResponse>, ApiError> {
    let games = state.games.read().await;
    let game = games
        .get(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

    let response = game_to_response(game);
    Ok(Json(MoveListResponse {
        moves: response.move_history,
    }))
}

// =========================================================================
// Undo Move
// =========================================================================

/// POST /api/games/:id/undo
pub async fn undo_move(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<GameResponse>, ApiError> {
    let mut games = state.games.write().await;
    let game = games
        .get_mut(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

    game.undo_move().map_err(ApiError::from)?;

    Ok(Json(game_to_response(game)))
}

// =========================================================================
// AI Move
// =========================================================================

/// POST /api/games/:id/ai-move
pub async fn ai_move(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(input): Json<AiMoveRequest>,
) -> Result<Json<AiMoveResponse>, ApiError> {
    // Read the game, clone it for AI computation.
    let (game_clone, difficulty) = {
        let games = state.games.read().await;
        let game = games
            .get(&id)
            .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

        let diff = input
            .difficulty
            .as_deref()
            .and_then(Difficulty::from_str_loose)
            .or(game.ai_difficulty)
            .unwrap_or(Difficulty::Medium);

        (game.clone(), diff)
    };

    // Run AI on a blocking thread to avoid starving the async runtime.
    let ai_result = tokio::task::spawn_blocking(move || {
        let engine = MinimaxAi::new();
        let start = Instant::now();
        let mv = engine.best_move(&game_clone, difficulty)?;
        let elapsed = start.elapsed().as_millis() as u64;
        Ok::<(Move, u64), crate::engine::types::ChessError>((mv, elapsed))
    })
    .await
    .map_err(|e| ApiError::InternalError(format!("AI task panicked: {e}")))?
    .map_err(ApiError::from)?;

    let (ai_mv, thinking_time) = ai_result;

    // Apply the AI move to the game.
    let mut games = state.games.write().await;
    let game = games
        .get_mut(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

    let san = game.make_move(ai_mv).map_err(ApiError::from)?;

    let game_resp = game_to_response(game);
    let ai_last = LastMove {
        from: ai_mv.from.to_algebraic(),
        to: ai_mv.to.to_algebraic(),
        san,
    };

    Ok(Json(AiMoveResponse {
        ai_move: ai_last,
        game: game_resp,
        evaluation: None,
        thinking_time,
    }))
}

// =========================================================================
// AI Hint
// =========================================================================

/// POST /api/games/:id/ai-hint
pub async fn ai_hint(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(input): Json<AiMoveRequest>,
) -> Result<Json<HintResponse>, ApiError> {
    let (game_clone, difficulty) = {
        let games = state.games.read().await;
        let game = games
            .get(&id)
            .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

        let diff = input
            .difficulty
            .as_deref()
            .and_then(Difficulty::from_str_loose)
            .or(game.ai_difficulty)
            .unwrap_or(Difficulty::Medium);

        (game.clone(), diff)
    };

    let legal = game_clone.legal_moves();

    let hint_result = tokio::task::spawn_blocking(move || {
        let engine = MinimaxAi::new();
        let start = Instant::now();
        let mv = engine.best_move(&game_clone, difficulty)?;
        let elapsed = start.elapsed().as_millis() as u64;
        Ok::<(Move, u64), crate::engine::types::ChessError>((mv, elapsed))
    })
    .await
    .map_err(|e| ApiError::InternalError(format!("AI task panicked: {e}")))?
    .map_err(ApiError::from)?;

    let (hint_mv, thinking_time) = hint_result;

    // Generate SAN for the hint move without modifying the game.
    let games = state.games.read().await;
    let game = games
        .get(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;
    let san = move_to_san(game.position(), hint_mv, &legal);

    Ok(Json(HintResponse {
        hint: LastMove {
            from: hint_mv.from.to_algebraic(),
            to: hint_mv.to.to_algebraic(),
            san,
        },
        evaluation: None,
        thinking_time,
    }))
}

// =========================================================================
// Legal Moves
// =========================================================================

/// GET /api/games/:id/legal-moves
pub async fn legal_moves(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Query(query): Query<LegalMovesQuery>,
) -> Result<Json<LegalMovesResponse>, ApiError> {
    let games = state.games.read().await;
    let game = games
        .get(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

    let all_legal = game.legal_moves();

    let moves: Vec<Move> = if let Some(ref from_str) = query.from {
        let sq = Square::from_algebraic(from_str)
            .ok_or_else(|| ApiError::InvalidRequest(format!("invalid square: {from_str}")))?;
        game.legal_moves_from(sq)
    } else {
        all_legal.clone()
    };

    let entries: Vec<LegalMoveEntry> = moves
        .iter()
        .map(|mv| legal_move_entry(game, *mv, &all_legal))
        .collect();

    Ok(Json(LegalMovesResponse { moves: entries }))
}

// =========================================================================
// Load FEN
// =========================================================================

/// POST /api/games/:id/fen
pub async fn load_fen(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(input): Json<FenRequest>,
) -> Result<Json<GameResponse>, ApiError> {
    let mut games = state.games.write().await;
    let game = games
        .get_mut(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

    game.load_fen(&input.fen).map_err(ApiError::from)?;

    Ok(Json(game_to_response(game)))
}

// =========================================================================
// Export PGN
// =========================================================================

/// GET /api/games/:id/pgn
pub async fn export_pgn(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let games = state.games.read().await;
    let game = games
        .get(&id)
        .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

    let pgn = crate::engine::pgn::to_pgn(game);

    Ok((
        StatusCode::OK,
        [("content-type", "text/plain; charset=utf-8")],
        pgn,
    ))
}

// =========================================================================
// Analysis
// =========================================================================

/// GET /api/games/:id/analysis
pub async fn analysis(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Query(query): Query<AnalysisQuery>,
) -> Result<Json<AnalysisResponse>, ApiError> {
    let (game_clone, difficulty, fen_for_analysis) = {
        let games = state.games.read().await;
        let game = games
            .get(&id)
            .ok_or_else(|| ApiError::GameNotFound(id.clone()))?;

        let diff = query
            .difficulty
            .as_deref()
            .and_then(Difficulty::from_str_loose)
            .or(game.ai_difficulty)
            .unwrap_or(Difficulty::Medium);

        (game.clone(), diff, game.to_fen())
    };

    let analysis_result: (Option<Move>, crate::ai::SearchStats, u64, Vec<Move>) =
        tokio::task::spawn_blocking(move || {
            let engine = MinimaxAi::new();
            let depth = difficulty.depth().max(1);
            let legal_for_san = game_clone.legal_moves();
            let mut pos = game_clone.position().clone();
            let start = Instant::now();
            let (best, stats) = engine.search_fixed_depth(&mut pos, depth);
            let elapsed = start.elapsed().as_millis() as u64;
            (best, stats, elapsed, legal_for_san)
        })
        .await
        .map_err(|e| ApiError::InternalError(format!("Analysis task panicked: {e}")))?;

    let (best_mv, stats, thinking_time, legal_for_san) = analysis_result;

    let best_move_entry = best_mv.map(|mv| {
        // Generate SAN using saved legal moves and position FEN.
        let san = if let Ok(pos) = crate::engine::board::Position::from_fen(&fen_for_analysis) {
            move_to_san(&pos, mv, &legal_for_san)
        } else {
            mv.to_string()
        };
        LastMove {
            from: mv.from.to_algebraic(),
            to: mv.to.to_algebraic(),
            san,
        }
    });

    Ok(Json(AnalysisResponse {
        evaluation: stats.score,
        best_move: best_move_entry,
        depth: stats.depth,
        nodes_searched: stats.nodes,
        thinking_time,
    }))
}

// =========================================================================
// Helpers
// =========================================================================

/// Resolve a MoveRequest into an engine Move by matching against legal moves.
fn resolve_move(game: &Game, input: &MoveRequest) -> Result<Move, ApiError> {
    let from = Square::from_algebraic(&input.from)
        .ok_or_else(|| ApiError::InvalidRequest(format!("invalid square: {}", input.from)))?;
    let to = Square::from_algebraic(&input.to)
        .ok_or_else(|| ApiError::InvalidRequest(format!("invalid square: {}", input.to)))?;

    let promotion = input
        .promotion
        .as_deref()
        .map(|p| {
            parse_promotion(p)
                .ok_or_else(|| ApiError::InvalidRequest(format!("invalid promotion: {p}")))
        })
        .transpose()?;

    let legal = game.legal_moves();

    // Find the matching legal move.
    let mv = legal
        .iter()
        .find(|m| m.from == from && m.to == to && m.promotion == promotion)
        .ok_or_else(|| crate::engine::types::ChessError::InvalidMove {
            from: input.from.clone(),
            to: input.to.clone(),
            reason: "not a legal move".into(),
        })?;

    Ok(*mv)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::router::create_router;
    use crate::api::state::AppState;
    use crate::config::AppConfig;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_state() -> SharedState {
        AppState::new(AppConfig::default())
    }

    async fn body_json(response: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn body_string(response: axum::http::Response<Body>) -> String {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    // --- Health ---

    #[tokio::test]
    async fn health_returns_200() {
        let app = create_router(test_state());
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["status"], "ok");
        assert_eq!(json["language"], "rust");
    }

    #[tokio::test]
    async fn not_found_returns_404() {
        let app = create_router(test_state());
        let resp = app
            .oneshot(Request::get("/nonexistent").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn cors_preflight() {
        let app = create_router(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/health")
                    .header("Origin", "http://localhost:3001")
                    .header("Access-Control-Request-Method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get("access-control-allow-origin").is_some());
    }

    // --- Create Game ---

    #[tokio::test]
    async fn create_game_default() {
        let app = create_router(test_state());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_json(resp).await;
        assert!(json["id"].is_string());
        assert_eq!(json["status"], "active");
        assert_eq!(json["currentPlayer"], "white");
        assert!(json["board"].is_array());
        assert_eq!(json["board"].as_array().unwrap().len(), 8);
    }

    #[tokio::test]
    async fn create_game_with_fen() {
        let app = create_router(test_state());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"fen":"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_json(resp).await;
        assert_eq!(json["currentPlayer"], "black");
    }

    #[tokio::test]
    async fn create_game_invalid_fen() {
        let app = create_router(test_state());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"fen":"invalid"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // --- Get / Delete Game ---

    #[tokio::test]
    async fn get_game_not_found() {
        let app = create_router(test_state());
        let resp = app
            .oneshot(
                Request::get("/api/games/nonexistent")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn create_then_get_game() {
        let state = test_state();

        // Create.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let json = body_json(resp).await;
        let id = json["id"].as_str().unwrap().to_string();

        // Get.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::get(format!("/api/games/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["id"], id);
    }

    #[tokio::test]
    async fn create_then_delete_game() {
        let state = test_state();

        // Create.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let json = body_json(resp).await;
        let id = json["id"].as_str().unwrap().to_string();

        // Delete.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::delete(format!("/api/games/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["success"], true);

        // Verify gone.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::get(format!("/api/games/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // --- List Games ---

    #[tokio::test]
    async fn list_games_empty() {
        let app = create_router(test_state());
        let resp = app
            .oneshot(Request::get("/api/games").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["total"], 0);
        assert!(json["games"].as_array().unwrap().is_empty());
    }

    // --- Make Move ---

    #[tokio::test]
    async fn make_move_e2e4() {
        let state = test_state();

        // Create game.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let json = body_json(resp).await;
        let id = json["id"].as_str().unwrap().to_string();

        // Make move e2-e4.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post(format!("/api/games/{id}/moves"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"from":"e2","to":"e4"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["currentPlayer"], "black");
        assert_eq!(json["moveHistory"].as_array().unwrap().len(), 1);
        assert_eq!(json["moveHistory"][0]["san"], "e4");
        assert_eq!(json["lastMove"]["from"], "e2");
        assert_eq!(json["lastMove"]["to"], "e4");
    }

    #[tokio::test]
    async fn make_invalid_move() {
        let state = test_state();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let json = body_json(resp).await;
        let id = json["id"].as_str().unwrap().to_string();

        // Illegal move e2-e5.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post(format!("/api/games/{id}/moves"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"from":"e2","to":"e5"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // --- Undo ---

    #[tokio::test]
    async fn undo_move_works() {
        let state = test_state();

        // Create + make move.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = create_router(state.clone());
        let _ = app
            .oneshot(
                Request::post(format!("/api/games/{id}/moves"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"from":"e2","to":"e4"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Undo.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post(format!("/api/games/{id}/undo"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["currentPlayer"], "white");
        assert!(json["moveHistory"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn undo_nothing_fails() {
        let state = test_state();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post(format!("/api/games/{id}/undo"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // --- Legal Moves ---

    #[tokio::test]
    async fn legal_moves_starting() {
        let state = test_state();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::get(format!("/api/games/{id}/legal-moves"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["moves"].as_array().unwrap().len(), 20);
    }

    #[tokio::test]
    async fn legal_moves_from_square() {
        let state = test_state();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::get(format!("/api/games/{id}/legal-moves?from=e2"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["moves"].as_array().unwrap().len(), 2); // e3, e4
    }

    // --- Load FEN ---

    #[tokio::test]
    async fn load_fen_changes_position() {
        let state = test_state();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post(format!("/api/games/{id}/fen"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"fen":"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["currentPlayer"], "black");
    }

    // --- PGN Export ---

    #[tokio::test]
    async fn export_pgn_returns_text() {
        let state = test_state();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::get(format!("/api/games/{id}/pgn"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let text = body_string(resp).await;
        assert!(text.contains("[Event"));
        assert!(text.contains("[Result"));
    }

    // --- AI Move ---

    #[tokio::test]
    async fn ai_move_returns_valid_move() {
        let state = test_state();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post(format!("/api/games/{id}/ai-move"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"difficulty":"easy"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["move"]["from"].is_string());
        assert!(json["move"]["to"].is_string());
        assert!(json["move"]["san"].is_string());
        assert_eq!(json["currentPlayer"], "black");
    }

    // --- AI Hint ---

    #[tokio::test]
    async fn ai_hint_does_not_modify_game() {
        let state = test_state();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        // Get hint.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post(format!("/api/games/{id}/ai-hint"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"difficulty":"easy"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["hint"]["san"].is_string());

        // Verify game unchanged.
        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::get(format!("/api/games/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let json = body_json(resp).await;
        assert_eq!(json["currentPlayer"], "white");
        assert!(json["moveHistory"].as_array().unwrap().is_empty());
    }

    // --- Analysis ---

    #[tokio::test]
    async fn analysis_returns_evaluation() {
        let state = test_state();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::post("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let id = body_json(resp).await["id"].as_str().unwrap().to_string();

        let app = create_router(state.clone());
        let resp = app
            .oneshot(
                Request::get(format!("/api/games/{id}/analysis?difficulty=easy"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json["evaluation"].is_number());
        assert!(json["depth"].is_number());
        assert!(json["nodesSearched"].is_number());
    }
}

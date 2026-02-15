//! WebSocket upgrade handler — connects a client to a game's live event
//! stream via the `WsManager`.

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures::{SinkExt, StreamExt};
use tracing::debug;

use crate::api::state::SharedState;
use crate::engine::types::GameStatus;

use super::manager::ClientId;
use super::messages::{WsCommand, WsEvent};

/// GET /ws/games/{id} — upgrade to WebSocket.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Path(id): Path<String>,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, id, state))
}

/// Core WebSocket session logic.
async fn handle_socket(socket: WebSocket, game_id: String, state: SharedState) {
    // Validate the game exists — send error and close if not.
    let initial_event = {
        let games = state.games.read().await;
        match games.get(&game_id) {
            Some(game) => {
                let status = game.status();
                let check = matches!(status, GameStatus::Check);
                let player = match game.side_to_move() {
                    crate::engine::types::Color::White => "white",
                    crate::engine::types::Color::Black => "black",
                };
                WsEvent::subscribed(
                    &game_id,
                    &game.to_fen(),
                    status.as_str(),
                    player,
                    game.move_history().len(),
                    check,
                )
            }
            None => {
                let (mut sink, _) = socket.split();
                let err = WsEvent::error(&format!("game not found: {game_id}"));
                let _ = sink.send(Message::Text(err.to_json().into())).await;
                let _ = sink.close().await;
                return;
            }
        }
    };

    // Subscribe this client to the game's broadcast channel.
    let (client_id, mut rx) = state.ws.subscribe(&game_id).await;
    let (mut sink, mut stream) = socket.split();

    // Send the initial subscribed event with the current game state.
    if sink
        .send(Message::Text(initial_event.to_json().into()))
        .await
        .is_err()
    {
        cleanup(&state, &game_id, client_id).await;
        return;
    }

    // Writer task: forward broadcast events from the manager → WS sink.
    let writer_state = state.clone();
    let writer_gid = game_id.clone();
    let mut writer = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if sink
                .send(Message::Text(event.to_json().into()))
                .await
                .is_err()
            {
                break;
            }
        }
        // Try to close gracefully.
        let _ = sink.close().await;
        cleanup(&writer_state, &writer_gid, client_id).await;
    });

    // Reader task: handle client → server commands.
    let reader_state = state.clone();
    let reader_gid = game_id.clone();
    let mut reader = tokio::spawn(async move {
        while let Some(Ok(msg)) = stream.next().await {
            match msg {
                Message::Text(text) => {
                    handle_client_message(&reader_state, &reader_gid, client_id, &text).await;
                }
                Message::Close(_) => break,
                _ => {} // Binary / Ping / Pong handled by Axum
            }
        }
    });

    // Wait for either task to finish, then abort the other.
    tokio::select! {
        _ = &mut writer => { reader.abort(); }
        _ = &mut reader => { writer.abort(); }
    }

    // Final cleanup (idempotent).
    cleanup(&state, &game_id, client_id).await;
}

/// Process a client-sent text message.
async fn handle_client_message(
    state: &SharedState,
    _current_game_id: &str,
    _client_id: ClientId,
    text: &str,
) {
    let cmd = match serde_json::from_str::<WsCommand>(text) {
        Ok(c) => c,
        Err(e) => {
            debug!("invalid WS command: {e}");
            // We could broadcast an error back, but we only hold the manager.
            // The writer task sends events; we don't have a direct sender here.
            // This is fine — unknown commands are silently ignored.
            return;
        }
    };

    match cmd {
        WsCommand::Ping => {
            // Respond with pong. Broadcast to the game so the pong goes
            // out through the writer task. (Only the sender will care, but
            // it's harmless.) For a more precise approach we could use a
            // per-client channel, but broadcasting pong is simple.
            let pong = WsEvent::pong();
            state.ws.broadcast(_current_game_id, pong).await;
        }
        WsCommand::Subscribe { game_id } => {
            debug!(
                game_id,
                "client requested subscribe (already subscribed via URL)"
            );
            // The client is already subscribed by virtue of connecting to
            // /ws/games/{id}. If they send a subscribe command for a
            // *different* game, we could handle multi-game subscriptions
            // here in a future version. For now, acknowledge.
            let games = state.games.read().await;
            if let Some(game) = games.get(&game_id) {
                let status = game.status();
                let check = matches!(status, GameStatus::Check);
                let player = match game.side_to_move() {
                    crate::engine::types::Color::White => "white",
                    crate::engine::types::Color::Black => "black",
                };
                let evt = WsEvent::subscribed(
                    &game_id,
                    &game.to_fen(),
                    status.as_str(),
                    player,
                    game.move_history().len(),
                    check,
                );
                state.ws.broadcast(&game_id, evt).await;
            }
        }
        WsCommand::Unsubscribe { game_id } => {
            debug!(game_id, "client requested unsubscribe");
            // Unsubscribing closes their channel for this game.
            state.ws.unsubscribe(&game_id, _client_id).await;
        }
    }
}

/// Remove client from the manager.
async fn cleanup(state: &SharedState, game_id: &str, client_id: ClientId) {
    state.ws.unsubscribe(game_id, client_id).await;
    debug!(game_id, client_id, "WS session cleaned up");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the handler function signature compiles as an Axum handler.
    #[tokio::test]
    async fn handler_type_check() {
        fn assert_handler<F, Fut, R>(_: F)
        where
            F: FnOnce(WebSocketUpgrade, Path<String>, State<SharedState>) -> Fut,
            Fut: std::future::Future<Output = R>,
            R: IntoResponse,
        {
        }
        assert_handler(ws_handler);
    }
}

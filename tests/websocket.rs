//! Integration tests for WebSocket real-time game events.
//!
//! Spins up an actual HTTP server and connects a WS client to validate the
//! full WebSocket lifecycle: connect → subscribe → receive events → close.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

use rumenx_chess::api::router::create_router;
use rumenx_chess::api::state::AppState;
use rumenx_chess::config::AppConfig;

/// Helper: start the server on an OS-assigned port, return its base URL.
async fn start_server() -> String {
    let state = AppState::new(AppConfig::default());
    let app = create_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://127.0.0.1:{}", addr.port())
}

/// Helper: create a game via REST, return its id.
async fn create_game(base: &str) -> String {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/games"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    body["id"].as_str().unwrap().to_string()
}

/// Helper: connect a WS client to a game, return (write, read) streams.
async fn ws_connect(
    base: &str,
    game_id: &str,
) -> (
    futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        Message,
    >,
    futures_util::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
) {
    let ws_url = base.replace("http://", "ws://");
    let url = format!("{ws_url}/ws/games/{game_id}");
    let (stream, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    stream.split()
}

/// Helper: read the next text message as JSON, with a timeout.
async fn next_json(
    read: &mut futures_util::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
) -> serde_json::Value {
    let msg = tokio::time::timeout(Duration::from_secs(5), read.next())
        .await
        .expect("timed out waiting for WS message")
        .expect("stream ended")
        .expect("WS error");

    match msg {
        Message::Text(text) => serde_json::from_str(&text).expect("invalid JSON"),
        other => panic!("expected Text message, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ws_connect_sends_subscribed_event() {
    let base = start_server().await;
    let game_id = create_game(&base).await;

    let (_write, mut read) = ws_connect(&base, &game_id).await;
    let msg = next_json(&mut read).await;

    assert_eq!(msg["type"], "subscribed");
    assert_eq!(msg["gameId"], game_id);
    assert_eq!(msg["currentPlayer"], "white");
    assert_eq!(msg["status"], "active");
    assert!(msg["fen"].is_string());
}

#[tokio::test]
async fn ws_connect_to_nonexistent_game_sends_error() {
    let base = start_server().await;

    let (_write, mut read) = ws_connect(&base, "nonexistent-id").await;
    let msg = next_json(&mut read).await;

    assert_eq!(msg["type"], "error");
    assert!(msg["message"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn ws_receives_move_made_event() {
    let base = start_server().await;
    let game_id = create_game(&base).await;

    // Connect WS client and consume the subscribed event.
    let (_write, mut read) = ws_connect(&base, &game_id).await;
    let _subscribed = next_json(&mut read).await;

    // Make a move via REST.
    let client = reqwest::Client::new();
    client
        .post(format!("{}/api/games/{}/moves", base, game_id))
        .json(&serde_json::json!({"from": "e2", "to": "e4"}))
        .send()
        .await
        .unwrap();

    // Should receive a move_made event.
    let msg = next_json(&mut read).await;
    assert_eq!(msg["type"], "move_made");
    assert_eq!(msg["gameId"], game_id);
    assert_eq!(msg["san"], "e4");
    assert_eq!(msg["from"], "e2");
    assert_eq!(msg["to"], "e4");
    assert_eq!(msg["player"], "white");
}

#[tokio::test]
async fn ws_receives_undo_game_state_event() {
    let base = start_server().await;
    let game_id = create_game(&base).await;

    let (_write, mut read) = ws_connect(&base, &game_id).await;
    let _subscribed = next_json(&mut read).await;

    // Make a move then undo it.
    let client = reqwest::Client::new();
    client
        .post(format!("{}/api/games/{}/moves", base, game_id))
        .json(&serde_json::json!({"from": "e2", "to": "e4"}))
        .send()
        .await
        .unwrap();
    let _move_evt = next_json(&mut read).await;

    client
        .post(format!("{}/api/games/{}/undo", base, game_id))
        .send()
        .await
        .unwrap();

    let msg = next_json(&mut read).await;
    assert_eq!(msg["type"], "game_state");
    assert_eq!(msg["gameId"], game_id);
    assert_eq!(msg["currentPlayer"], "white"); // back to white after undo
}

#[tokio::test]
async fn ws_receives_fen_load_event() {
    let base = start_server().await;
    let game_id = create_game(&base).await;

    let (_write, mut read) = ws_connect(&base, &game_id).await;
    let _subscribed = next_json(&mut read).await;

    let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    let client = reqwest::Client::new();
    client
        .post(format!("{}/api/games/{}/fen", base, game_id))
        .json(&serde_json::json!({"fen": fen}))
        .send()
        .await
        .unwrap();

    let msg = next_json(&mut read).await;
    assert_eq!(msg["type"], "game_state");
    assert_eq!(msg["gameId"], game_id);
    assert_eq!(msg["currentPlayer"], "black");
}

#[tokio::test]
async fn ws_receives_ai_thinking_and_complete() {
    let base = start_server().await;
    let game_id = create_game(&base).await;

    let (_write, mut read) = ws_connect(&base, &game_id).await;
    let _subscribed = next_json(&mut read).await;

    // Request AI move.
    let client = reqwest::Client::new();
    client
        .post(format!("{}/api/games/{}/ai-move", base, game_id))
        .json(&serde_json::json!({"difficulty": "harmless"}))
        .send()
        .await
        .unwrap();

    // Should receive ai_thinking then ai_move_complete.
    let msg1 = next_json(&mut read).await;
    assert_eq!(msg1["type"], "ai_thinking");
    assert_eq!(msg1["gameId"], game_id);
    assert_eq!(msg1["difficulty"], "harmless");

    let msg2 = next_json(&mut read).await;
    assert_eq!(msg2["type"], "ai_move_complete");
    assert_eq!(msg2["gameId"], game_id);
    assert!(msg2["san"].is_string());
    assert!(msg2["thinkingTimeMs"].is_number());
}

#[tokio::test]
async fn ws_multiple_clients_receive_events() {
    let base = start_server().await;
    let game_id = create_game(&base).await;

    // Connect two WS clients.
    let (_w1, mut r1) = ws_connect(&base, &game_id).await;
    let (_w2, mut r2) = ws_connect(&base, &game_id).await;

    // Consume subscribed events.
    let _s1 = next_json(&mut r1).await;
    let _s2 = next_json(&mut r2).await;

    // Make a move.
    let client = reqwest::Client::new();
    client
        .post(format!("{}/api/games/{}/moves", base, game_id))
        .json(&serde_json::json!({"from": "d2", "to": "d4"}))
        .send()
        .await
        .unwrap();

    // Both should receive the move.
    let m1 = next_json(&mut r1).await;
    let m2 = next_json(&mut r2).await;
    assert_eq!(m1["type"], "move_made");
    assert_eq!(m2["type"], "move_made");
    assert_eq!(m1["san"], m2["san"]);
}

#[tokio::test]
async fn ws_different_games_isolated() {
    let base = start_server().await;
    let game1 = create_game(&base).await;
    let game2 = create_game(&base).await;

    let (_w1, mut r1) = ws_connect(&base, &game1).await;
    let (_w2, mut r2) = ws_connect(&base, &game2).await;

    let _s1 = next_json(&mut r1).await;
    let _s2 = next_json(&mut r2).await;

    // Make a move in game1 only.
    let client = reqwest::Client::new();
    client
        .post(format!("{}/api/games/{}/moves", base, game1))
        .json(&serde_json::json!({"from": "e2", "to": "e4"}))
        .send()
        .await
        .unwrap();

    // game1 client receives the event.
    let m1 = next_json(&mut r1).await;
    assert_eq!(m1["type"], "move_made");

    // game2 client should NOT receive anything (timeout).
    let result = tokio::time::timeout(Duration::from_millis(200), r2.next()).await;
    assert!(
        result.is_err(),
        "game2 client should not receive events from game1"
    );
}

#[tokio::test]
async fn ws_ping_command_returns_pong() {
    let base = start_server().await;
    let game_id = create_game(&base).await;

    let (mut write, mut read) = ws_connect(&base, &game_id).await;
    let _subscribed = next_json(&mut read).await;

    // Send a ping command.
    write
        .send(Message::Text(r#"{"type":"ping"}"#.into()))
        .await
        .unwrap();

    let msg = next_json(&mut read).await;
    assert_eq!(msg["type"], "pong");
    assert!(msg["timestamp"].is_number());
}

#[tokio::test]
async fn ws_subscribe_command_sends_state() {
    let base = start_server().await;
    let game_id = create_game(&base).await;

    let (mut write, mut read) = ws_connect(&base, &game_id).await;
    let _subscribed = next_json(&mut read).await;

    // Send an explicit subscribe command for the same game.
    let cmd = serde_json::json!({"type": "subscribe", "game_id": game_id});
    write
        .send(Message::Text(cmd.to_string().into()))
        .await
        .unwrap();

    let msg = next_json(&mut read).await;
    assert_eq!(msg["type"], "subscribed");
    assert_eq!(msg["gameId"], game_id);
}

#[tokio::test]
async fn ws_game_over_event_on_checkmate() {
    let base = start_server().await;

    // Create game from a position one move from Scholar's mate.
    // Position: Black can be mated with Qxf7#
    let fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4";
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/api/games", base))
        .json(&serde_json::json!({"fen": fen}))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    let game_id = body["id"].as_str().unwrap().to_string();

    let (_write, mut read) = ws_connect(&base, &game_id).await;
    let _subscribed = next_json(&mut read).await;

    // Deliver Scholar's mate: Qxf7#
    client
        .post(format!("{}/api/games/{}/moves", base, game_id))
        .json(&serde_json::json!({"from": "h5", "to": "f7"}))
        .send()
        .await
        .unwrap();

    // Should get move_made then game_over.
    let move_msg = next_json(&mut read).await;
    assert_eq!(move_msg["type"], "move_made");
    assert_eq!(move_msg["status"], "checkmate");

    let go_msg = next_json(&mut read).await;
    assert_eq!(go_msg["type"], "game_over");
    assert_eq!(go_msg["result"], "checkmate");
}

#[tokio::test]
async fn ws_manager_subscriber_counts() {
    let base = start_server().await;
    let game_id = create_game(&base).await;

    // Verify the health endpoint still works (basic sanity).
    let client = reqwest::Client::new();
    let resp = client.get(format!("{base}/health")).send().await.unwrap();
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");

    // Connect two clients.
    let (_w1, mut _r1) = ws_connect(&base, &game_id).await;
    let (_w2, mut _r2) = ws_connect(&base, &game_id).await;

    // Give the server a moment to process both connections.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // There's no public endpoint for subscriber count, but we can
    // verify both receive events (already tested above).
    // This test just ensures no panics with multiple rapid connections.
}

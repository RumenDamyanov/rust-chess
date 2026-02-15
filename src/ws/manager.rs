//! WebSocket connection manager — tracks active connections per game and
//! provides `broadcast()` to push events to all subscribers.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, warn};

use super::messages::WsEvent;

/// Handle for a single WebSocket client.  The handler owns the
/// receiving half; the manager keeps the sending half.
pub type ClientSender = mpsc::UnboundedSender<WsEvent>;

/// A unique ID assigned to each connected WebSocket client.
pub type ClientId = u64;

/// Manages per-game sets of connected clients and provides broadcast.
#[derive(Debug)]
pub struct WsManager {
    /// game_id → { client_id → sender }
    subs: RwLock<HashMap<String, HashMap<ClientId, ClientSender>>>,
    /// Monotonically increasing counter for client IDs.
    next_id: std::sync::atomic::AtomicU64,
}

impl WsManager {
    /// Create a new, empty manager.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            subs: RwLock::new(HashMap::new()),
            next_id: std::sync::atomic::AtomicU64::new(1),
        })
    }

    /// Register a new client for a game, returning (client_id, receiver).
    pub async fn subscribe(&self, game_id: &str) -> (ClientId, mpsc::UnboundedReceiver<WsEvent>) {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();

        let mut subs = self.subs.write().await;
        subs.entry(game_id.to_string()).or_default().insert(id, tx);

        debug!(game_id, client_id = id, "WS client subscribed");
        (id, rx)
    }

    /// Remove a client from a game.
    pub async fn unsubscribe(&self, game_id: &str, client_id: ClientId) {
        let mut subs = self.subs.write().await;
        if let Some(clients) = subs.get_mut(game_id) {
            clients.remove(&client_id);
            if clients.is_empty() {
                subs.remove(game_id);
            }
        }
        debug!(game_id, client_id, "WS client unsubscribed");
    }

    /// Broadcast an event to all subscribers of a game.
    pub async fn broadcast(&self, game_id: &str, event: WsEvent) {
        let subs = self.subs.read().await;
        if let Some(clients) = subs.get(game_id) {
            let mut stale: Vec<ClientId> = Vec::new();
            for (&cid, tx) in clients {
                if tx.send(event.clone()).is_err() {
                    stale.push(cid);
                }
            }
            drop(subs); // release read lock before write

            if !stale.is_empty() {
                let mut subs = self.subs.write().await;
                if let Some(clients) = subs.get_mut(game_id) {
                    for cid in &stale {
                        clients.remove(cid);
                        warn!(game_id, client_id = cid, "removed stale WS client");
                    }
                    if clients.is_empty() {
                        subs.remove(game_id);
                    }
                }
            }
        }
    }

    /// Number of subscribers for a game.
    pub async fn subscriber_count(&self, game_id: &str) -> usize {
        let subs = self.subs.read().await;
        subs.get(game_id).map_or(0, |c| c.len())
    }

    /// Total number of active connections across all games.
    pub async fn total_connections(&self) -> usize {
        let subs = self.subs.read().await;
        subs.values().map(|c| c.len()).sum()
    }

    /// List game IDs with active subscribers.
    pub async fn active_games(&self) -> Vec<String> {
        let subs = self.subs.read().await;
        subs.keys().cloned().collect()
    }
}

impl Default for WsManager {
    fn default() -> Self {
        Self {
            subs: RwLock::new(HashMap::new()),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn subscribe_returns_unique_ids() {
        let mgr = WsManager::new();
        let (id1, _rx1) = mgr.subscribe("g1").await;
        let (id2, _rx2) = mgr.subscribe("g1").await;
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn subscriber_count_tracks_correctly() {
        let mgr = WsManager::new();
        assert_eq!(mgr.subscriber_count("g1").await, 0);

        let (id1, _rx1) = mgr.subscribe("g1").await;
        assert_eq!(mgr.subscriber_count("g1").await, 1);

        let (_id2, _rx2) = mgr.subscribe("g1").await;
        assert_eq!(mgr.subscriber_count("g1").await, 2);

        mgr.unsubscribe("g1", id1).await;
        assert_eq!(mgr.subscriber_count("g1").await, 1);
    }

    #[tokio::test]
    async fn unsubscribe_removes_empty_game() {
        let mgr = WsManager::new();
        let (id1, _rx1) = mgr.subscribe("g1").await;
        mgr.unsubscribe("g1", id1).await;
        assert_eq!(mgr.subscriber_count("g1").await, 0);
        assert!(mgr.active_games().await.is_empty());
    }

    #[tokio::test]
    async fn broadcast_delivers_to_all_subscribers() {
        let mgr = WsManager::new();
        let (_id1, mut rx1) = mgr.subscribe("g1").await;
        let (_id2, mut rx2) = mgr.subscribe("g1").await;

        let evt = WsEvent::game_state("g1", "startfen", "active", "white", 0, false);
        mgr.broadcast("g1", evt).await;

        let msg1 = rx1.recv().await.unwrap();
        let msg2 = rx2.recv().await.unwrap();
        assert_eq!(msg1.to_json(), msg2.to_json());
    }

    #[tokio::test]
    async fn broadcast_does_not_cross_games() {
        let mgr = WsManager::new();
        let (_id1, mut rx1) = mgr.subscribe("g1").await;
        let (_id2, mut rx2) = mgr.subscribe("g2").await;

        let evt = WsEvent::game_state("g1", "fen", "active", "white", 0, false);
        mgr.broadcast("g1", evt).await;

        // g1 client gets it
        assert!(rx1.recv().await.is_some());
        // g2 client does not
        assert!(rx2.try_recv().is_err());
    }

    #[tokio::test]
    async fn broadcast_removes_stale_clients() {
        let mgr = WsManager::new();
        let (_id1, rx1) = mgr.subscribe("g1").await;
        let (_id2, _rx2) = mgr.subscribe("g1").await;

        // Drop rx1 to simulate disconnected client
        drop(rx1);

        let evt = WsEvent::pong();
        mgr.broadcast("g1", evt).await;

        // Stale client should be cleaned up
        assert_eq!(mgr.subscriber_count("g1").await, 1);
    }

    #[tokio::test]
    async fn total_connections_across_games() {
        let mgr = WsManager::new();
        let (_id1, _rx1) = mgr.subscribe("g1").await;
        let (_id2, _rx2) = mgr.subscribe("g1").await;
        let (_id3, _rx3) = mgr.subscribe("g2").await;
        assert_eq!(mgr.total_connections().await, 3);
    }

    #[tokio::test]
    async fn active_games_lists_games() {
        let mgr = WsManager::new();
        let (_id1, _rx1) = mgr.subscribe("g1").await;
        let (_id2, _rx2) = mgr.subscribe("g2").await;
        let games = mgr.active_games().await;
        assert_eq!(games.len(), 2);
        assert!(games.contains(&"g1".to_string()));
        assert!(games.contains(&"g2".to_string()));
    }

    #[tokio::test]
    async fn broadcast_to_nonexistent_game_is_noop() {
        let mgr = WsManager::new();
        // Should not panic
        mgr.broadcast("nonexistent", WsEvent::pong()).await;
    }

    #[tokio::test]
    async fn unsubscribe_nonexistent_is_noop() {
        let mgr = WsManager::new();
        // Should not panic
        mgr.unsubscribe("g1", 999).await;
    }
}

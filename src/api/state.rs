use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::AppConfig;
use crate::engine::game::Game;

/// Games stored by UUID.
pub type GameStore = RwLock<HashMap<String, Game>>;

/// Shared application state passed to all handlers via Axum's State extractor.
pub struct AppState {
    pub games: GameStore,
    pub config: AppConfig,
    pub start_time: std::time::Instant,
}

pub type SharedState = Arc<AppState>;

impl AppState {
    pub fn new(config: AppConfig) -> SharedState {
        Arc::new(AppState {
            games: RwLock::new(HashMap::new()),
            config,
            start_time: std::time::Instant::now(),
        })
    }
}

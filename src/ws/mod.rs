//! WebSocket module — real-time game event streaming.
//!
//! - [`messages`]: Typed event/command envelopes.
//! - [`manager`]: Per-game connection tracking and broadcast.
//! - [`handler`]: Axum WebSocket upgrade handler.

pub mod handler;
pub mod manager;
pub mod messages;

pub use handler::ws_handler;
pub use manager::WsManager;
pub use messages::WsEvent;

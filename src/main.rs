use rust_chess::api::router::create_router;
use rust_chess::api::state::AppState;
use rust_chess::config::AppConfig;

#[tokio::main]
async fn main() {
    // Handle --health-check flag for Docker HEALTHCHECK (works in scratch image).
    if std::env::args().any(|a| a == "--health-check") {
        match health_check().await {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("Health check failed: {e}");
                std::process::exit(1);
            }
        }
    }

    // Initialize tracing (structured logging).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rust_chess=info,tower_http=info".into()),
        )
        .init();

    let config = AppConfig::from_env();
    let bind_addr = config.bind_addr();
    let state = AppState::new(config);

    let app = create_router(state);

    tracing::info!(
        "rust-chess v{} starting on {bind_addr}",
        env!("CARGO_PKG_VERSION")
    );

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app).await.expect("Server error");
}

/// Lightweight health check: send raw HTTP/1.1 request and check for 200 OK.
async fn health_check() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::var("PORT").unwrap_or_else(|_| "8082".to_string());
    let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}")).await?;
    let request =
        format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    stream.write_all(request.as_bytes()).await?;
    let mut buf = vec![0u8; 1024];
    let n = stream.read(&mut buf).await?;
    let response = String::from_utf8_lossy(&buf[..n]);
    if response.starts_with("HTTP/1.1 200") {
        Ok(())
    } else {
        Err(format!(
            "Unexpected response: {}",
            response.lines().next().unwrap_or("")
        )
        .into())
    }
}

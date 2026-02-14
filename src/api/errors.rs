use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::engine::ChessError;

/// Structured API error that serializes to JSON.
#[derive(Debug)]
pub enum ApiError {
    GameNotFound(String),
    InvalidMove(ChessError),
    InvalidFen(ChessError),
    InvalidRequest(String),
    GameOver(String),
    NothingToUndo,
    InternalError(String),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorDetail {
    code: String,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            ApiError::GameNotFound(id) => (
                StatusCode::NOT_FOUND,
                "GAME_NOT_FOUND",
                format!("Game not found: {id}"),
            ),
            ApiError::InvalidMove(err) => {
                (StatusCode::BAD_REQUEST, "INVALID_MOVE", err.to_string())
            }
            ApiError::InvalidFen(err) => (StatusCode::BAD_REQUEST, "INVALID_FEN", err.to_string()),
            ApiError::InvalidRequest(msg) => (StatusCode::BAD_REQUEST, "INVALID_REQUEST", msg),
            ApiError::GameOver(msg) => (
                StatusCode::BAD_REQUEST,
                "GAME_OVER",
                format!("Game is already over: {msg}"),
            ),
            ApiError::NothingToUndo => (
                StatusCode::BAD_REQUEST,
                "NOTHING_TO_UNDO",
                "No moves to undo".to_string(),
            ),
            ApiError::InternalError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", msg)
            }
        };

        let body = ErrorResponse {
            error: ErrorDetail {
                code: code.to_string(),
                message,
            },
        };

        (status, Json(body)).into_response()
    }
}

impl From<ChessError> for ApiError {
    fn from(err: ChessError) -> Self {
        match &err {
            ChessError::InvalidMove { .. } => ApiError::InvalidMove(err),
            ChessError::InvalidFen(_) => ApiError::InvalidFen(err),
            ChessError::InvalidSquare(_) => ApiError::InvalidRequest(err.to_string()),
            ChessError::GameOver(_) => ApiError::GameOver(err.to_string()),
            ChessError::InvalidPromotion(_) => ApiError::InvalidRequest(err.to_string()),
            ChessError::NothingToUndo => ApiError::NothingToUndo,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    async fn error_to_json(err: ApiError) -> (StatusCode, serde_json::Value) {
        let response = err.into_response();
        let status = response.status();
        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, json)
    }

    #[tokio::test]
    async fn game_not_found_returns_404() {
        let (status, json) = error_to_json(ApiError::GameNotFound("abc".into())).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json["error"]["code"], "GAME_NOT_FOUND");
    }

    #[tokio::test]
    async fn invalid_request_returns_400() {
        let (status, json) = error_to_json(ApiError::InvalidRequest("bad input".into())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "INVALID_REQUEST");
    }

    #[tokio::test]
    async fn nothing_to_undo_returns_400() {
        let (status, json) = error_to_json(ApiError::NothingToUndo).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "NOTHING_TO_UNDO");
    }

    #[tokio::test]
    async fn internal_error_returns_500() {
        let (status, json) = error_to_json(ApiError::InternalError("oops".into())).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(json["error"]["code"], "INTERNAL_ERROR");
    }

    #[tokio::test]
    async fn chess_error_converts_to_api_error() {
        let chess_err = ChessError::InvalidFen("bad fen".into());
        let api_err: ApiError = chess_err.into();
        let (status, json) = error_to_json(api_err).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "INVALID_FEN");
    }
}

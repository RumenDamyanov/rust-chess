use serde::{Deserialize, Serialize};

use crate::engine::game::Game;
use crate::engine::san::move_to_san;
use crate::engine::types::{Color, Move, PieceType, Square};

// ---------------------------------------------------------------------------
// Request models
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateGameRequest {
    pub fen: Option<String>,
    pub ai_enabled: Option<bool>,
    pub ai_difficulty: Option<String>,
    pub ai_color: Option<String>,
    pub white_player: Option<String>,
    pub black_player: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveRequest {
    pub from: String,
    pub to: String,
    pub promotion: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiMoveRequest {
    pub difficulty: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FenRequest {
    pub fen: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListGamesQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegalMovesQuery {
    pub from: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisQuery {
    pub difficulty: Option<String>,
}

// ---------------------------------------------------------------------------
// Response models
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub language: String,
    pub engine: String,
    pub uptime: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GameResponse {
    pub id: String,
    pub board: Vec<Vec<Option<String>>>,
    pub fen: String,
    pub status: String,
    pub current_player: String,
    pub move_history: Vec<MoveHistoryEntry>,
    pub captured_pieces: CapturedPieces,
    pub check: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_move: Option<LastMove>,
    pub players: Players,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveHistoryEntry {
    pub from: String,
    pub to: String,
    pub piece: PieceInfo,
    pub captured: Option<PieceInfo>,
    pub promotion: Option<String>,
    pub san: String,
    pub fen: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PieceInfo {
    #[serde(rename = "type")]
    pub piece_type: String,
    pub color: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapturedPieces {
    pub white: Vec<String>,
    pub black: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct LastMove {
    pub from: String,
    pub to: String,
    pub san: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Players {
    pub white: String,
    pub black: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListGamesResponse {
    pub games: Vec<GameResponse>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveListResponse {
    pub moves: Vec<MoveHistoryEntry>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LegalMoveEntry {
    pub from: String,
    pub to: String,
    pub san: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub promotion: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LegalMovesResponse {
    pub moves: Vec<LegalMoveEntry>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiMoveResponse {
    #[serde(rename = "move")]
    pub ai_move: LastMove,
    #[serde(flatten)]
    pub game: GameResponse,
    pub evaluation: Option<i32>,
    pub thinking_time: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HintResponse {
    pub hint: LastMove,
    pub evaluation: Option<i32>,
    pub thinking_time: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisResponse {
    pub evaluation: i32,
    pub best_move: Option<LastMove>,
    pub depth: u32,
    pub nodes_searched: u64,
    pub thinking_time: u64,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn piece_type_name(pt: PieceType) -> &'static str {
    match pt {
        PieceType::Pawn => "pawn",
        PieceType::Knight => "knight",
        PieceType::Bishop => "bishop",
        PieceType::Rook => "rook",
        PieceType::Queen => "queen",
        PieceType::King => "king",
    }
}

fn color_name(c: Color) -> &'static str {
    match c {
        Color::White => "white",
        Color::Black => "black",
    }
}

/// Promotion piece type character from string like "queen", "rook", etc.
pub fn parse_promotion(s: &str) -> Option<PieceType> {
    match s.to_lowercase().as_str() {
        "queen" | "q" => Some(PieceType::Queen),
        "rook" | "r" => Some(PieceType::Rook),
        "bishop" | "b" => Some(PieceType::Bishop),
        "knight" | "n" => Some(PieceType::Knight),
        _ => None,
    }
}

fn promotion_name(pt: PieceType) -> &'static str {
    match pt {
        PieceType::Queen => "queen",
        PieceType::Rook => "rook",
        PieceType::Bishop => "bishop",
        PieceType::Knight => "knight",
        _ => "queen",
    }
}

/// Build the 8×8 board array for the API response.
/// Row 0 = rank 8 (top), row 7 = rank 1 (bottom).
/// Pieces: uppercase for White ("R"), lowercase for Black ("r").
/// Empty = None.
pub fn board_to_api(game: &Game) -> Vec<Vec<Option<String>>> {
    let pos = game.position();
    let mut rows = Vec::with_capacity(8);
    for rank in (0..8u8).rev() {
        let mut row = Vec::with_capacity(8);
        for file in 0..8u8 {
            let sq = Square::from_file_rank(file, rank);
            match pos.piece_at(sq) {
                Some((color, pt)) => {
                    let ch = pt.to_char(color);
                    row.push(Some(ch.to_string()));
                }
                None => row.push(None),
            }
        }
        rows.push(row);
    }
    rows
}

/// Build captured pieces from move history by replaying captures.
pub fn captured_pieces(game: &Game) -> CapturedPieces {
    let mut white_captured: Vec<String> = Vec::new(); // pieces captured BY white (black pieces)
    let mut black_captured: Vec<String> = Vec::new(); // pieces captured BY black (white pieces)

    // We need to track captures. The MoveRecord has the move with CAPTURE flag.
    // We need to know what was captured. We'll replay from starting position.
    // Simpler: read captured info from moves. But MoveRecord only stores mv and san.
    // We'll need to reconstruct. For now, track from the position difference.

    // Alternative: compare piece counts from starting position vs current.
    // This is simpler and always correct.
    let starting_counts = if game.started_from_fen() {
        // Re-parse the starting FEN to get original piece counts.
        if let Ok(start) = crate::engine::board::Position::from_fen(game.starting_fen()) {
            count_pieces(&start)
        } else {
            starting_piece_counts()
        }
    } else {
        starting_piece_counts()
    };

    let current = count_pieces(game.position());

    // White pieces captured (by Black) = starting white - current white
    for pt_idx in 0..6 {
        let diff = starting_counts[0][pt_idx] as i32 - current[0][pt_idx] as i32;
        let pt = PieceType::ALL[pt_idx];
        for _ in 0..diff.max(0) {
            white_captured.push(pt.to_char(Color::White).to_string());
        }
    }

    // Black pieces captured (by White) = starting black - current black
    for pt_idx in 0..6 {
        let diff = starting_counts[1][pt_idx] as i32 - current[1][pt_idx] as i32;
        let pt = PieceType::ALL[pt_idx];
        for _ in 0..diff.max(0) {
            black_captured.push(pt.to_char(Color::Black).to_string());
        }
    }

    CapturedPieces {
        white: black_captured, // pieces lost by white (captured by black)
        black: white_captured, // pieces lost by black (captured by white)
    }
}

fn count_pieces(pos: &crate::engine::board::Position) -> [[u32; 6]; 2] {
    let mut counts = [[0u32; 6]; 2];
    for (color, row) in counts.iter_mut().enumerate() {
        for (pt, cell) in row.iter_mut().enumerate() {
            *cell = pos.pieces[color][pt].pop_count();
        }
    }
    counts
}

fn starting_piece_counts() -> [[u32; 6]; 2] {
    // Standard starting position: P=8, N=2, B=2, R=2, Q=1, K=1
    [[8, 2, 2, 2, 1, 1], [8, 2, 2, 2, 1, 1]]
}

/// Convert internal Game to full API GameResponse.
pub fn game_to_response(game: &Game) -> GameResponse {
    let status = game.status();
    let is_check = matches!(status, crate::engine::types::GameStatus::Check);

    let last_move = game.move_history().last().map(|rec| LastMove {
        from: rec.mv.from.to_algebraic(),
        to: rec.mv.to.to_algebraic(),
        san: rec.san.clone(),
    });

    GameResponse {
        id: game.id.clone(),
        board: board_to_api(game),
        fen: game.to_fen(),
        status: status.as_str().to_string(),
        current_player: color_name(game.side_to_move()).to_string(),
        move_history: build_move_history(game),
        captured_pieces: captured_pieces(game),
        check: is_check,
        last_move,
        players: Players {
            white: game.white_player.clone(),
            black: game.black_player.clone(),
        },
        created_at: game.created_at.to_rfc3339(),
    }
}

/// Build the move history entries by replaying the game from the starting
/// position to reconstruct per-move FEN and captured piece info.
fn build_move_history(game: &Game) -> Vec<MoveHistoryEntry> {
    let history = game.move_history();
    if history.is_empty() {
        return Vec::new();
    }

    // Replay from starting FEN.
    let mut replay = if game.started_from_fen() {
        match Game::from_fen(game.starting_fen()) {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        }
    } else {
        Game::new()
    };

    let mut entries = Vec::with_capacity(history.len());

    for rec in history {
        let mv = rec.mv;
        let pos = replay.position();
        let side = pos.side_to_move;

        // Determine piece being moved.
        let piece_info = pos
            .piece_at(mv.from)
            .map(|(c, pt)| PieceInfo {
                piece_type: piece_type_name(pt).to_string(),
                color: color_name(c).to_string(),
            })
            .unwrap_or(PieceInfo {
                piece_type: "pawn".to_string(),
                color: color_name(side).to_string(),
            });

        // Determine captured piece (if any).
        let captured = if mv.flags.is_capture() {
            if mv.flags.is_en_passant() {
                Some(PieceInfo {
                    piece_type: "pawn".to_string(),
                    color: color_name(!side).to_string(),
                })
            } else {
                pos.piece_at(mv.to).map(|(c, pt)| PieceInfo {
                    piece_type: piece_type_name(pt).to_string(),
                    color: color_name(c).to_string(),
                })
            }
        } else {
            None
        };

        let promotion = mv.promotion.map(|pt| promotion_name(pt).to_string());

        // Make the move to get the resulting FEN.
        let _ = replay.make_move(mv);
        let fen_after = replay.to_fen();

        entries.push(MoveHistoryEntry {
            from: mv.from.to_algebraic(),
            to: mv.to.to_algebraic(),
            piece: piece_info,
            captured,
            promotion,
            san: rec.san.clone(),
            fen: fen_after,
        });
    }

    entries
}

/// Build a legal move entry with SAN.
pub fn legal_move_entry(game: &Game, mv: Move, legal: &[Move]) -> LegalMoveEntry {
    let san = move_to_san(game.position(), mv, legal);
    LegalMoveEntry {
        from: mv.from.to_algebraic(),
        to: mv.to.to_algebraic(),
        san,
        promotion: mv.promotion.map(|pt| promotion_name(pt).to_string()),
    }
}

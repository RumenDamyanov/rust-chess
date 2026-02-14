//! Stateful game controller wrapping Position.
//!
//! `Game` manages move history, undo stack, repetition tracking, and game
//! status detection (checkmate, stalemate, draws). It is the primary type
//! the API layer interacts with.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::engine::board::{Position, UndoInfo};
use crate::engine::movegen;
use crate::engine::types::{
    ChessError, Color, Difficulty, DrawReason, GameStatus, Move, PieceType, Square,
};

// =========================================================================
// MoveRecord
// =========================================================================

/// A recorded move in the game history.
#[derive(Clone, Debug)]
pub struct MoveRecord {
    /// The move that was played.
    pub mv: Move,
    /// The SAN notation for the move (computed at move time).
    pub san: String,
    /// What game status resulted from this move.
    pub status_after: GameStatus,
}

// =========================================================================
// Game
// =========================================================================

/// A complete chess game with history, undo, and status tracking.
#[derive(Clone, Debug)]
pub struct Game {
    // Core state
    position: Position,
    move_history: Vec<MoveRecord>,
    undo_stack: Vec<UndoInfo>,
    /// Zobrist hashes of all positions reached (for threefold repetition).
    /// Includes the current position.
    position_hashes: Vec<u64>,

    // Status
    status: GameStatus,

    // Metadata
    pub id: String,
    pub white_player: String,
    pub black_player: String,
    pub created_at: DateTime<Utc>,
    pub ai_difficulty: Option<Difficulty>,

    // FEN tracking
    started_from_fen: bool,
    starting_fen: String,
}

impl Game {
    // -----------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------

    /// Create a new game from the standard starting position.
    pub fn new() -> Self {
        let pos = Position::starting();
        let hash = pos.zobrist_hash;
        let fen = pos.to_fen();
        Self {
            position: pos,
            move_history: Vec::new(),
            undo_stack: Vec::new(),
            position_hashes: vec![hash],
            status: GameStatus::Active,
            id: Uuid::new_v4().to_string(),
            white_player: "Player".into(),
            black_player: "Player".into(),
            created_at: Utc::now(),
            ai_difficulty: None,
            started_from_fen: false,
            starting_fen: fen,
        }
    }

    /// Create a game from a FEN string.
    pub fn from_fen(fen: &str) -> Result<Self, ChessError> {
        let pos = Position::from_fen(fen)?;
        let hash = pos.zobrist_hash;
        let mut game = Self {
            position: pos,
            move_history: Vec::new(),
            undo_stack: Vec::new(),
            position_hashes: vec![hash],
            status: GameStatus::Active,
            id: Uuid::new_v4().to_string(),
            white_player: "Player".into(),
            black_player: "Player".into(),
            created_at: Utc::now(),
            ai_difficulty: None,
            started_from_fen: true,
            starting_fen: fen.to_string(),
        };
        game.status = game.compute_status();
        Ok(game)
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    /// Current board position.
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Current game status.
    pub fn status(&self) -> &GameStatus {
        &self.status
    }

    /// Side to move.
    pub fn side_to_move(&self) -> Color {
        self.position.side_to_move
    }

    /// Completed move history.
    pub fn move_history(&self) -> &[MoveRecord] {
        &self.move_history
    }

    /// All legal moves in the current position.
    pub fn legal_moves(&self) -> Vec<Move> {
        movegen::legal_moves(&self.position)
    }

    /// Legal moves from a specific square.
    pub fn legal_moves_from(&self, sq: Square) -> Vec<Move> {
        movegen::legal_moves_from(&self.position, sq)
    }

    /// Whether the game is over.
    pub fn is_game_over(&self) -> bool {
        self.status.is_game_over()
    }

    /// Current position as FEN.
    pub fn to_fen(&self) -> String {
        self.position.to_fen()
    }

    /// Whether the game was started from a custom FEN.
    pub fn started_from_fen(&self) -> bool {
        self.started_from_fen
    }

    /// The starting FEN.
    pub fn starting_fen(&self) -> &str {
        &self.starting_fen
    }

    /// Fullmove number.
    pub fn fullmove_number(&self) -> u16 {
        self.position.fullmove_number
    }

    /// Halfmove clock (for 50-move rule).
    pub fn halfmove_clock(&self) -> u16 {
        self.position.halfmove_clock
    }

    // -----------------------------------------------------------------
    // Make move
    // -----------------------------------------------------------------

    /// Play a move. Returns the SAN notation of the move played.
    ///
    /// The move must be legal. Returns `ChessError::GameOver` if the game
    /// is already finished, or `ChessError::InvalidMove` if not legal.
    pub fn make_move(&mut self, mv: Move) -> Result<String, ChessError> {
        if self.status.is_game_over() {
            return Err(ChessError::GameOver(format!(
                "game is over: {}",
                self.status
            )));
        }

        // Validate legality.
        let legal = self.legal_moves();
        if !legal.contains(&mv) {
            return Err(ChessError::InvalidMove {
                from: mv.from.to_algebraic(),
                to: mv.to.to_algebraic(),
                reason: "not a legal move".into(),
            });
        }

        // Generate SAN before making the move (needs current position context).
        let san = crate::engine::san::move_to_san(&self.position, mv, &legal);

        // Apply the move.
        let undo = self.position.make_move(mv);
        self.undo_stack.push(undo);
        self.position_hashes.push(self.position.zobrist_hash);

        // Compute new status.
        let status = self.compute_status();
        self.status = status.clone();

        // Append check/mate suffix to SAN.
        let san = match &status {
            GameStatus::Checkmate => format!("{san}#"),
            GameStatus::Check => format!("{san}+"),
            _ => san,
        };

        self.move_history.push(MoveRecord {
            mv,
            san: san.clone(),
            status_after: status,
        });

        Ok(san)
    }

    // -----------------------------------------------------------------
    // Undo move
    // -----------------------------------------------------------------

    /// Undo the last move. Returns the move that was undone.
    pub fn undo_move(&mut self) -> Result<Move, ChessError> {
        let record = self.move_history.pop().ok_or(ChessError::NothingToUndo)?;
        let undo = self.undo_stack.pop().unwrap();
        self.position_hashes.pop();

        self.position.undo_move(record.mv, &undo);
        self.status = self.compute_status();

        Ok(record.mv)
    }

    // -----------------------------------------------------------------
    // Load a new FEN into an existing game (reset).
    // -----------------------------------------------------------------

    /// Load a FEN position, resetting all history.
    pub fn load_fen(&mut self, fen: &str) -> Result<(), ChessError> {
        let pos = Position::from_fen(fen)?;
        self.position = pos;
        self.move_history.clear();
        self.undo_stack.clear();
        self.position_hashes.clear();
        self.position_hashes.push(self.position.zobrist_hash);
        self.started_from_fen = true;
        self.starting_fen = fen.to_string();
        self.status = self.compute_status();
        Ok(())
    }

    // -----------------------------------------------------------------
    // Status detection
    // -----------------------------------------------------------------

    fn compute_status(&self) -> GameStatus {
        let legal = movegen::legal_moves(&self.position);
        let in_check = self.position.is_in_check();

        if legal.is_empty() {
            if in_check {
                return GameStatus::Checkmate;
            } else {
                return GameStatus::Stalemate;
            }
        }

        // Draw conditions (checked in order of cheapness).
        if self.position.halfmove_clock >= 100 {
            return GameStatus::Draw(DrawReason::FiftyMoveRule);
        }

        if self.is_threefold_repetition() {
            return GameStatus::Draw(DrawReason::ThreefoldRepetition);
        }

        if self.is_insufficient_material() {
            return GameStatus::Draw(DrawReason::InsufficientMaterial);
        }

        if in_check {
            GameStatus::Check
        } else {
            GameStatus::Active
        }
    }

    /// Threefold repetition: current position hash has appeared 3+ times.
    fn is_threefold_repetition(&self) -> bool {
        let current = self.position.zobrist_hash;
        let count = self
            .position_hashes
            .iter()
            .filter(|&&h| h == current)
            .count();
        count >= 3
    }

    /// Insufficient material detection.
    ///
    /// Draws: K vs K, K+B vs K, K+N vs K, K+B vs K+B (same color bishops).
    fn is_insufficient_material(&self) -> bool {
        let pos = &self.position;

        // Any pawns, rooks, or queens → sufficient.
        for color_idx in 0..2 {
            if pos.pieces[color_idx][PieceType::Pawn.index()].is_not_empty() {
                return false;
            }
            if pos.pieces[color_idx][PieceType::Rook.index()].is_not_empty() {
                return false;
            }
            if pos.pieces[color_idx][PieceType::Queen.index()].is_not_empty() {
                return false;
            }
        }

        let w_knights = pos.pieces[Color::White.index()][PieceType::Knight.index()].pop_count();
        let b_knights = pos.pieces[Color::Black.index()][PieceType::Knight.index()].pop_count();
        let w_bishops = pos.pieces[Color::White.index()][PieceType::Bishop.index()].pop_count();
        let b_bishops = pos.pieces[Color::Black.index()][PieceType::Bishop.index()].pop_count();

        let w_minor = w_knights + w_bishops;
        let b_minor = b_knights + b_bishops;

        // K vs K
        if w_minor == 0 && b_minor == 0 {
            return true;
        }

        // K+minor vs K
        if (w_minor == 1 && b_minor == 0) || (w_minor == 0 && b_minor == 1) {
            return true;
        }

        // K+B vs K+B with bishops on same color squares
        if w_minor == 1 && b_minor == 1 && w_bishops == 1 && b_bishops == 1 {
            let w_bsq = pos.pieces[Color::White.index()][PieceType::Bishop.index()]
                .lsb()
                .unwrap()
                .0;
            let b_bsq = pos.pieces[Color::Black.index()][PieceType::Bishop.index()]
                .lsb()
                .unwrap()
                .0;
            let w_sq_color = (w_bsq / 8 + w_bsq % 8) & 1;
            let b_sq_color = (b_bsq / 8 + b_bsq % 8) & 1;
            if w_sq_color == b_sq_color {
                return true;
            }
        }

        false
    }

    // -----------------------------------------------------------------
    // Board array (for API responses)
    // -----------------------------------------------------------------

    /// Generate an 8×8 board array (row-major, rank 8 first → rank 1 last).
    /// Empty squares are empty strings. Pieces are like "wP", "bK", etc.
    pub fn board_array(&self) -> [[String; 8]; 8] {
        let mut board = std::array::from_fn(|_| std::array::from_fn(|_| String::new()));
        for rank in 0..8u8 {
            for file in 0..8u8 {
                let sq = Square::from_file_rank(file, 7 - rank);
                if let Some((color, piece)) = self.position.piece_at(sq) {
                    let c = match color {
                        Color::White => 'w',
                        Color::Black => 'b',
                    };
                    let p = match piece {
                        PieceType::Pawn => 'P',
                        PieceType::Knight => 'N',
                        PieceType::Bishop => 'B',
                        PieceType::Rook => 'R',
                        PieceType::Queen => 'Q',
                        PieceType::King => 'K',
                    };
                    board[rank as usize][file as usize] = format!("{c}{p}");
                }
            }
        }
        board
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::MoveFlags;

    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    #[test]
    fn new_game_is_active() {
        let g = Game::new();
        assert_eq!(*g.status(), GameStatus::Active);
        assert!(!g.is_game_over());
        assert_eq!(g.side_to_move(), Color::White);
        assert_eq!(g.fullmove_number(), 1);
    }

    #[test]
    fn game_from_fen() {
        let g =
            Game::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        assert_eq!(g.side_to_move(), Color::Black);
        assert!(g.started_from_fen());
    }

    #[test]
    fn game_from_invalid_fen() {
        assert!(Game::from_fen("invalid").is_err());
    }

    // -----------------------------------------------------------------
    // Making moves
    // -----------------------------------------------------------------

    #[test]
    fn make_move_e2e4() {
        let mut g = Game::new();
        let mv = Move::new(
            Square::from_algebraic("e2").unwrap(),
            Square::from_algebraic("e4").unwrap(),
        );
        let mv = Move::with_flags(mv.from, mv.to, MoveFlags::DOUBLE_PUSH);
        let san = g.make_move(mv).unwrap();
        assert_eq!(san, "e4");
        assert_eq!(g.side_to_move(), Color::Black);
        assert_eq!(g.move_history().len(), 1);
    }

    #[test]
    fn make_illegal_move_errors() {
        let mut g = Game::new();
        let mv = Move::new(
            Square::from_algebraic("e2").unwrap(),
            Square::from_algebraic("e5").unwrap(),
        );
        assert!(g.make_move(mv).is_err());
    }

    #[test]
    fn make_move_on_finished_game_errors() {
        // Fool's mate.
        let mut g =
            Game::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        // Play fool's mate: 1. f3 e5 2. g4 Qh4#
        play(&mut g, "f2", "f3", MoveFlags::NONE);
        play(&mut g, "e7", "e5", MoveFlags::DOUBLE_PUSH);
        play(&mut g, "g2", "g4", MoveFlags::DOUBLE_PUSH);
        play(&mut g, "d8", "h4", MoveFlags::NONE);
        assert_eq!(*g.status(), GameStatus::Checkmate);
        assert!(g.is_game_over());

        // Try another move.
        let mv = Move::new(
            Square::from_algebraic("e2").unwrap(),
            Square::from_algebraic("e4").unwrap(),
        );
        assert!(g.make_move(mv).is_err());
    }

    // -----------------------------------------------------------------
    // Undo
    // -----------------------------------------------------------------

    #[test]
    fn undo_single_move() {
        let mut g = Game::new();
        let original_fen = g.to_fen();
        let mv = Move::with_flags(
            Square::from_algebraic("e2").unwrap(),
            Square::from_algebraic("e4").unwrap(),
            MoveFlags::DOUBLE_PUSH,
        );
        g.make_move(mv).unwrap();
        g.undo_move().unwrap();
        assert_eq!(g.to_fen(), original_fen);
        assert_eq!(g.move_history().len(), 0);
    }

    #[test]
    fn undo_nothing_errors() {
        let mut g = Game::new();
        assert!(g.undo_move().is_err());
    }

    // -----------------------------------------------------------------
    // Status detection: checkmate
    // -----------------------------------------------------------------

    #[test]
    fn scholars_mate() {
        // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#
        let mut g = Game::new();
        play(&mut g, "e2", "e4", MoveFlags::DOUBLE_PUSH);
        play(&mut g, "e7", "e5", MoveFlags::DOUBLE_PUSH);
        play(&mut g, "f1", "c4", MoveFlags::NONE);
        play(&mut g, "b8", "c6", MoveFlags::NONE);
        play(&mut g, "d1", "h5", MoveFlags::NONE);
        play(&mut g, "g8", "f6", MoveFlags::NONE);
        play(&mut g, "h5", "f7", MoveFlags::CAPTURE);
        assert_eq!(*g.status(), GameStatus::Checkmate);
        assert!(g.is_game_over());
    }

    // -----------------------------------------------------------------
    // Status detection: stalemate
    // -----------------------------------------------------------------

    #[test]
    fn stalemate_detection() {
        // Black king on a8, white king on c7, white queen on b6.
        // Black to move — no legal moves but not in check.
        let g = Game::from_fen("k7/2K5/1Q6/8/8/8/8/8 b - - 0 1").unwrap();
        assert_eq!(*g.status(), GameStatus::Stalemate);
    }

    // -----------------------------------------------------------------
    // Status detection: fifty-move rule
    // -----------------------------------------------------------------

    #[test]
    fn fifty_move_rule_detection() {
        let g = Game::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 80").unwrap();
        assert_eq!(*g.status(), GameStatus::Draw(DrawReason::FiftyMoveRule));
    }

    // -----------------------------------------------------------------
    // Status detection: insufficient material
    // -----------------------------------------------------------------

    #[test]
    fn insufficient_material_k_vs_k() {
        let g = Game::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            *g.status(),
            GameStatus::Draw(DrawReason::InsufficientMaterial)
        );
    }

    #[test]
    fn insufficient_material_k_plus_bishop_vs_k() {
        let g = Game::from_fen("4k3/8/8/8/8/8/8/4KB2 w - - 0 1").unwrap();
        assert_eq!(
            *g.status(),
            GameStatus::Draw(DrawReason::InsufficientMaterial)
        );
    }

    #[test]
    fn insufficient_material_k_plus_knight_vs_k() {
        let g = Game::from_fen("4k3/8/8/8/8/8/8/4KN2 w - - 0 1").unwrap();
        assert_eq!(
            *g.status(),
            GameStatus::Draw(DrawReason::InsufficientMaterial)
        );
    }

    #[test]
    fn insufficient_material_kb_vs_kb_same_color() {
        // Both bishops on light squares (c1=dark, no: c1 file=2 rank=0, 2+0=even=dark).
        // f1 (file=5 rank=0) = 5+0=odd=light. f8 = 5+7=12=even=dark. Not same color.
        // Let me pick: B on c1 (dark) and b on f8 (dark) => same.
        // c1 = sq(2,0): 0+2=2 (even) = dark. f8 = sq(5,7): 7+5=12 (even) = dark. Same!
        let g = Game::from_fen("4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1").unwrap();
        assert_eq!(
            *g.status(),
            GameStatus::Draw(DrawReason::InsufficientMaterial)
        );
    }

    #[test]
    fn sufficient_material_kb_vs_kb_diff_color() {
        // B on c1 (dark), b on c8 (light: 2+7=9=odd). Different colors → sufficient.
        let g = Game::from_fen("2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1").unwrap();
        assert_ne!(
            *g.status(),
            GameStatus::Draw(DrawReason::InsufficientMaterial)
        );
    }

    #[test]
    fn sufficient_material_with_pawns() {
        let g = Game::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        assert_eq!(*g.status(), GameStatus::Active);
    }

    // -----------------------------------------------------------------
    // Threefold repetition
    // -----------------------------------------------------------------

    #[test]
    fn threefold_repetition() {
        let mut g = Game::new();
        // Shuffle knights back and forth 3 times.
        // Ng1-f3, Ng8-f6, Nf3-g1, Nf6-g8, Ng1-f3, Ng8-f6, Nf3-g1, Nf6-g8
        // After these 8 half-moves we return to the starting position 3 times.
        for _ in 0..2 {
            play(&mut g, "g1", "f3", MoveFlags::NONE);
            play(&mut g, "g8", "f6", MoveFlags::NONE);
            play(&mut g, "f3", "g1", MoveFlags::NONE);
            play(&mut g, "f6", "g8", MoveFlags::NONE);
        }
        assert_eq!(
            *g.status(),
            GameStatus::Draw(DrawReason::ThreefoldRepetition)
        );
    }

    // -----------------------------------------------------------------
    // Board array
    // -----------------------------------------------------------------

    #[test]
    fn board_array_starting_position() {
        let g = Game::new();
        let board = g.board_array();
        // Rank 8 = row 0: rook on a8.
        assert_eq!(board[0][0], "bR");
        // Rank 1 = row 7: king on e1.
        assert_eq!(board[7][4], "wK");
        // Rank 5 = row 3: empty.
        assert_eq!(board[3][0], "");
    }

    // -----------------------------------------------------------------
    // Load FEN
    // -----------------------------------------------------------------

    #[test]
    fn load_fen_resets_game() {
        let mut g = Game::new();
        play(&mut g, "e2", "e4", MoveFlags::DOUBLE_PUSH);
        g.load_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(g.move_history().len(), 0);
        assert!(g.started_from_fen());
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    fn play(g: &mut Game, from: &str, to: &str, flags: MoveFlags) {
        let mv = Move::with_flags(
            Square::from_algebraic(from).unwrap(),
            Square::from_algebraic(to).unwrap(),
            flags,
        );
        g.make_move(mv).unwrap();
    }
}

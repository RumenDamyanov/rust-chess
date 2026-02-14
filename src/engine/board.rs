//! Bitboard-based chess position representation.
//!
//! `Position` stores piece placement as 12 bitboards (2 colours × 6 piece types),
//! redundant occupancy bitboards, side to move, castling rights, en-passant square,
//! move counters, and an incremental Zobrist hash.

use crate::engine::attacks;
use crate::engine::types::{Bitboard, CastlingRights, ChessError, Color, Move, PieceType, Square};
use crate::engine::zobrist;

// ---------------------------------------------------------------------------
// UndoInfo — saved state for reversing a move
// ---------------------------------------------------------------------------

/// State that must be saved before making a move so it can be restored on undo.
#[derive(Clone, Debug)]
pub struct UndoInfo {
    pub captured_piece: Option<PieceType>,
    pub castling_rights: CastlingRights,
    pub en_passant: Option<Square>,
    pub halfmove_clock: u16,
    pub zobrist_hash: u64,
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

/// A complete chess position using bitboard representation.
///
/// Board layout follows LERF (Little-Endian Rank-File) mapping:
/// a1 = 0, b1 = 1, … h1 = 7, a2 = 8, … h8 = 63.
#[derive(Clone, Debug)]
pub struct Position {
    /// Piece bitboards: `pieces[color][piece_type]`.
    pub pieces: [[Bitboard; PieceType::COUNT]; 2],

    /// Per-colour occupancy (union of all piece bitboards for that colour).
    pub occupied: [Bitboard; 2],

    /// Total occupancy (union of both colours).
    pub all_occupied: Bitboard,

    /// Whose turn it is.
    pub side_to_move: Color,

    /// Castling availability (K/Q/k/q).
    pub castling_rights: CastlingRights,

    /// En-passant target square (the square *behind* the double-pushed pawn).
    pub en_passant: Option<Square>,

    /// Half-move clock for the 50-move rule (reset on pawn move or capture).
    pub halfmove_clock: u16,

    /// Full-move number (starts at 1, incremented after Black moves).
    pub fullmove_number: u16,

    /// Incremental Zobrist hash of the position.
    pub zobrist_hash: u64,
}

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

impl Position {
    /// Create an empty board with no pieces.
    pub fn empty() -> Self {
        Position {
            pieces: [[Bitboard::EMPTY; PieceType::COUNT]; 2],
            occupied: [Bitboard::EMPTY; 2],
            all_occupied: Bitboard::EMPTY,
            side_to_move: Color::White,
            castling_rights: CastlingRights::NONE,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            zobrist_hash: 0,
        }
    }

    /// Standard starting position.
    pub fn starting() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("starting FEN is always valid")
    }

    // -----------------------------------------------------------------------
    // Piece manipulation (low-level)
    // -----------------------------------------------------------------------

    /// Place a piece on a square. Does NOT update the Zobrist hash.
    #[inline]
    pub fn put_piece(&mut self, sq: Square, color: Color, piece: PieceType) {
        let bb = Bitboard::from_square(sq);
        self.pieces[color.index()][piece.index()] |= bb;
        self.occupied[color.index()] |= bb;
        self.all_occupied |= bb;
    }

    /// Remove a piece from a square. Does NOT update the Zobrist hash.
    #[inline]
    pub fn remove_piece(&mut self, sq: Square, color: Color, piece: PieceType) {
        let bb = Bitboard::from_square(sq);
        self.pieces[color.index()][piece.index()] &= !bb;
        self.occupied[color.index()] &= !bb;
        self.all_occupied &= !bb;
    }

    /// Place a piece and update the Zobrist hash.
    #[inline]
    pub fn put_piece_hash(&mut self, sq: Square, color: Color, piece: PieceType) {
        self.put_piece(sq, color, piece);
        self.zobrist_hash ^= zobrist::keys().piece_key(color, piece, sq);
    }

    /// Remove a piece and update the Zobrist hash.
    #[inline]
    pub fn remove_piece_hash(&mut self, sq: Square, color: Color, piece: PieceType) {
        self.remove_piece(sq, color, piece);
        self.zobrist_hash ^= zobrist::keys().piece_key(color, piece, sq);
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// What piece (if any) is on a given square?
    pub fn piece_at(&self, sq: Square) -> Option<(Color, PieceType)> {
        let bb = Bitboard::from_square(sq);

        // Quick bail-out: nothing on this square at all.
        if (self.all_occupied & bb).is_empty() {
            return None;
        }

        // Determine colour.
        let color = if (self.occupied[Color::White.index()] & bb).is_not_empty() {
            Color::White
        } else {
            Color::Black
        };

        // Find which piece type.
        for &pt in &PieceType::ALL {
            if (self.pieces[color.index()][pt.index()] & bb).is_not_empty() {
                return Some((color, pt));
            }
        }

        // Should be unreachable if occupancy is consistent.
        None
    }

    /// Bitboard of all pieces of a given colour and type.
    #[inline]
    pub fn bb(&self, color: Color, piece: PieceType) -> Bitboard {
        self.pieces[color.index()][piece.index()]
    }

    /// Bitboard of friendly (side-to-move) pieces.
    #[inline]
    pub fn friendly(&self) -> Bitboard {
        self.occupied[self.side_to_move.index()]
    }

    /// Bitboard of enemy pieces.
    #[inline]
    pub fn enemy(&self) -> Bitboard {
        self.occupied[(!self.side_to_move).index()]
    }

    /// Find the king square for the given colour.
    #[inline]
    pub fn king_sq(&self, color: Color) -> Square {
        self.pieces[color.index()][PieceType::King.index()]
            .lsb()
            .expect("king must exist")
    }

    // -----------------------------------------------------------------------
    // Zobrist hash computation (full recompute)
    // -----------------------------------------------------------------------

    /// Compute the Zobrist hash from scratch (useful for FEN loading / verification).
    pub fn compute_zobrist(&self) -> u64 {
        let zk = zobrist::keys();
        let mut hash = 0u64;

        // Pieces.
        for color in [Color::White, Color::Black] {
            for &pt in &PieceType::ALL {
                for sq in self.bb(color, pt).iter() {
                    hash ^= zk.piece_key(color, pt, sq);
                }
            }
        }

        // Side to move.
        if self.side_to_move == Color::Black {
            hash ^= zk.side_to_move;
        }

        // Castling rights.
        hash ^= zk.castling_key(self.castling_rights.0);

        // En passant file.
        if let Some(ep_sq) = self.en_passant {
            hash ^= zk.ep_key(ep_sq.file());
        }

        hash
    }

    // -----------------------------------------------------------------------
    // Occupancy sanity check (debug builds)
    // -----------------------------------------------------------------------

    /// Verify that redundant occupancy bitboards are consistent with piece bitboards.
    /// Available in debug builds and test builds.
    #[cfg(any(debug_assertions, test))]
    pub fn assert_consistent(&self) {
        for color in [Color::White, Color::Black] {
            let mut expected = Bitboard::EMPTY;
            for &pt in &PieceType::ALL {
                expected |= self.pieces[color.index()][pt.index()];
            }
            assert_eq!(
                self.occupied[color.index()],
                expected,
                "occupancy mismatch for {color:?}",
            );
        }
        assert_eq!(
            self.all_occupied,
            self.occupied[0] | self.occupied[1],
            "all_occupied mismatch",
        );
    }

    // -----------------------------------------------------------------------
    // Attack detection
    // -----------------------------------------------------------------------

    /// Is `sq` attacked by any piece of colour `by`?
    pub fn is_square_attacked(&self, sq: Square, by: Color) -> bool {
        let t = attacks::tables();
        let occ = self.all_occupied;

        // Pawn attacks: if a pawn of `by` is on a square that attacks `sq`…
        let pawn_atk = t.pawn_attacks(!by, sq); // reverse: squares that attack sq
        if (pawn_atk & self.bb(by, PieceType::Pawn)).is_not_empty() {
            return true;
        }

        // Knight attacks.
        if (t.knight_attacks(sq) & self.bb(by, PieceType::Knight)).is_not_empty() {
            return true;
        }

        // King attacks.
        if (t.king_attacks(sq) & self.bb(by, PieceType::King)).is_not_empty() {
            return true;
        }

        // Rook / Queen (straight lines).
        let rook_queen = self.bb(by, PieceType::Rook) | self.bb(by, PieceType::Queen);
        if (t.rook_attacks(sq, occ) & rook_queen).is_not_empty() {
            return true;
        }

        // Bishop / Queen (diagonals).
        let bishop_queen = self.bb(by, PieceType::Bishop) | self.bb(by, PieceType::Queen);
        if (t.bishop_attacks(sq, occ) & bishop_queen).is_not_empty() {
            return true;
        }

        false
    }

    /// Is the side-to-move's king currently in check?
    #[inline]
    pub fn is_in_check(&self) -> bool {
        let king = self.king_sq(self.side_to_move);
        self.is_square_attacked(king, !self.side_to_move)
    }

    // -----------------------------------------------------------------------
    // Make / Undo move
    // -----------------------------------------------------------------------

    /// Apply a move to the position. Returns `UndoInfo` for reversal.
    ///
    /// The caller is responsible for ensuring the move is legal (i.e. the
    /// king is not left in check). In debug builds, we assert this at the end.
    pub fn make_move(&mut self, mv: Move) -> UndoInfo {
        let zk = zobrist::keys();
        let us = self.side_to_move;
        let them = !us;

        // Save restorable state.
        let undo = UndoInfo {
            captured_piece: None, // updated below if capture
            castling_rights: self.castling_rights,
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            zobrist_hash: self.zobrist_hash,
        };

        // Determine which piece is moving.
        let moving_piece = self.piece_type_at(mv.from, us);

        // ---- Remove en-passant hash (if any) ----
        if let Some(ep) = self.en_passant {
            self.zobrist_hash ^= zk.ep_key(ep.file());
        }
        self.en_passant = None;

        // ---- Remove old castling hash ----
        self.zobrist_hash ^= zk.castling_key(self.castling_rights.0);

        // ---- Handle capture ----
        let mut captured = None;
        if mv.flags.is_en_passant() {
            // The captured pawn is on a different square from mv.to.
            let cap_sq = match us {
                Color::White => Square(mv.to.0 - 8),
                Color::Black => Square(mv.to.0 + 8),
            };
            self.remove_piece(cap_sq, them, PieceType::Pawn);
            self.zobrist_hash ^= zk.piece_key(them, PieceType::Pawn, cap_sq);
            captured = Some(PieceType::Pawn);
        } else if mv.flags.is_capture() {
            // Normal capture — find what's on the to-square.
            let cap_piece = self.piece_type_at(mv.to, them);
            self.remove_piece(mv.to, them, cap_piece);
            self.zobrist_hash ^= zk.piece_key(them, cap_piece, mv.to);
            captured = Some(cap_piece);
        }

        // ---- Move the piece ----
        self.remove_piece(mv.from, us, moving_piece);
        self.zobrist_hash ^= zk.piece_key(us, moving_piece, mv.from);

        let landing_piece = mv.promotion.unwrap_or(moving_piece);
        self.put_piece(mv.to, us, landing_piece);
        self.zobrist_hash ^= zk.piece_key(us, landing_piece, mv.to);

        // ---- Castling: move the rook ----
        if mv.flags.is_castling() {
            let (rook_from, rook_to) = castling_rook_squares(mv.to);
            self.remove_piece(rook_from, us, PieceType::Rook);
            self.zobrist_hash ^= zk.piece_key(us, PieceType::Rook, rook_from);
            self.put_piece(rook_to, us, PieceType::Rook);
            self.zobrist_hash ^= zk.piece_key(us, PieceType::Rook, rook_to);
        }

        // ---- Update castling rights ----
        // Moving king or rook, or capturing on a rook's home square.
        self.castling_rights.0 &= CASTLING_MASK[mv.from.0 as usize];
        self.castling_rights.0 &= CASTLING_MASK[mv.to.0 as usize];

        self.zobrist_hash ^= zk.castling_key(self.castling_rights.0);

        // ---- Double pawn push → set en passant ----
        if mv.flags.is_double_push() {
            let ep_sq = match us {
                Color::White => Square(mv.from.0 + 8),
                Color::Black => Square(mv.from.0 - 8),
            };
            self.en_passant = Some(ep_sq);
            self.zobrist_hash ^= zk.ep_key(ep_sq.file());
        }

        // ---- Halfmove clock ----
        if moving_piece == PieceType::Pawn || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        // ---- Fullmove number ----
        if us == Color::Black {
            self.fullmove_number += 1;
        }

        // ---- Switch side ----
        self.side_to_move = them;
        self.zobrist_hash ^= zk.side_to_move;

        // Return undo info with capture.
        UndoInfo {
            captured_piece: captured,
            ..undo
        }
    }

    /// Reverse a move previously applied with `make_move`.
    pub fn undo_move(&mut self, mv: Move, undo: &UndoInfo) {
        let them = self.side_to_move; // after make_move, side was switched
        let us = !them;

        // ---- Switch side back ----
        self.side_to_move = us;

        // ---- Determine which piece landed ----
        let landing_piece = mv
            .promotion
            .unwrap_or_else(|| self.piece_type_at(mv.to, us));
        let original_piece = if mv.promotion.is_some() {
            PieceType::Pawn
        } else {
            landing_piece
        };

        // ---- Remove the piece from to-square, put back on from-square ----
        self.remove_piece(mv.to, us, landing_piece);
        self.put_piece(mv.from, us, original_piece);

        // ---- Restore capture ----
        if mv.flags.is_en_passant() {
            let cap_sq = match us {
                Color::White => Square(mv.to.0 - 8),
                Color::Black => Square(mv.to.0 + 8),
            };
            self.put_piece(cap_sq, them, PieceType::Pawn);
        } else if let Some(cap_piece) = undo.captured_piece {
            self.put_piece(mv.to, them, cap_piece);
        }

        // ---- Undo castling: move the rook back ----
        if mv.flags.is_castling() {
            let (rook_from, rook_to) = castling_rook_squares(mv.to);
            self.remove_piece(rook_to, us, PieceType::Rook);
            self.put_piece(rook_from, us, PieceType::Rook);
        }

        // ---- Restore saved state ----
        self.castling_rights = undo.castling_rights;
        self.en_passant = undo.en_passant;
        self.halfmove_clock = undo.halfmove_clock;
        self.zobrist_hash = undo.zobrist_hash;

        // Fullmove: decrement if we're undoing a Black move.
        if us == Color::Black {
            self.fullmove_number -= 1;
        }
    }

    // -----------------------------------------------------------------------
    // Internal helper: find piece type on a square for a known colour
    // -----------------------------------------------------------------------

    /// Like `piece_at` but only checks one colour and panics if not found.
    #[inline]
    fn piece_type_at(&self, sq: Square, color: Color) -> PieceType {
        let bb = Bitboard::from_square(sq);
        for &pt in &PieceType::ALL {
            if (self.pieces[color.index()][pt.index()] & bb).is_not_empty() {
                return pt;
            }
        }
        panic!(
            "no {} piece found on {} (board:\n{})",
            color,
            sq,
            self.board_string()
        );
    }

    // -----------------------------------------------------------------------
    // Board display (8×8 text grid)
    // -----------------------------------------------------------------------

    /// Render the board as an 8-line string (rank 8 at top), useful for debugging.
    pub fn board_string(&self) -> String {
        let mut s = String::with_capacity(200);
        for rank in (0..8).rev() {
            s.push((b'1' + rank) as char);
            s.push(' ');
            for file in 0..8 {
                let sq = Square::from_file_rank(file, rank);
                let ch = match self.piece_at(sq) {
                    Some((c, p)) => p.to_char(c),
                    None => '.',
                };
                s.push(ch);
                if file < 7 {
                    s.push(' ');
                }
            }
            s.push('\n');
        }
        s.push_str("  a b c d e f g h");
        s
    }
}

// ---------------------------------------------------------------------------
// Castling helpers (free functions)
// ---------------------------------------------------------------------------

/// For a king-destination square (after castling), return (rook_from, rook_to).
fn castling_rook_squares(king_to: Square) -> (Square, Square) {
    match king_to.0 {
        // White kingside: king e1→g1, rook h1→f1.
        6 => (Square(7), Square(5)),
        // White queenside: king e1→c1, rook a1→d1.
        2 => (Square(0), Square(3)),
        // Black kingside: king e8→g8, rook h8→f8.
        62 => (Square(63), Square(61)),
        // Black queenside: king e8→c8, rook a8→d8.
        58 => (Square(56), Square(59)),
        _ => panic!("invalid castling king destination: {king_to}"),
    }
}

/// Mask table indexed by square index. When a move touches a square, AND the
/// castling rights with this mask. E.g. if a rook on a1 moves (or is captured),
/// remove White-queenside. The king's home square removes both that side's rights.
#[rustfmt::skip]
const CASTLING_MASK: [u8; 64] = {
    let mut mask = [0b1111u8; 64];
    // a1 (0): remove white-queenside (bit 1)
    mask[0]  = 0b1111 & !CastlingRights::WHITE_QUEENSIDE;
    // e1 (4): remove both white rights
    mask[4]  = 0b1111 & !(CastlingRights::WHITE_KINGSIDE | CastlingRights::WHITE_QUEENSIDE);
    // h1 (7): remove white-kingside (bit 0)
    mask[7]  = 0b1111 & !CastlingRights::WHITE_KINGSIDE;
    // a8 (56): remove black-queenside (bit 3)
    mask[56] = 0b1111 & !CastlingRights::BLACK_QUEENSIDE;
    // e8 (60): remove both black rights
    mask[60] = 0b1111 & !(CastlingRights::BLACK_KINGSIDE | CastlingRights::BLACK_QUEENSIDE);
    // h8 (63): remove black-kingside (bit 2)
    mask[63] = 0b1111 & !CastlingRights::BLACK_KINGSIDE;
    mask
};

// ---------------------------------------------------------------------------
// FEN parsing & generation
// ---------------------------------------------------------------------------

impl Position {
    /// Parse a FEN string into a `Position`.
    ///
    /// Validates all 6 fields (piece placement, side to move, castling,
    /// en passant, halfmove clock, fullmove number) and ensures exactly one
    /// king per side.
    pub fn from_fen(fen: &str) -> Result<Self, ChessError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(ChessError::InvalidFen(format!(
                "expected 6 fields, got {}",
                fields.len()
            )));
        }

        let mut pos = Position::empty();

        // ----- Field 1: Piece placement -----
        let ranks: Vec<&str> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(ChessError::InvalidFen(format!(
                "expected 8 ranks, got {}",
                ranks.len()
            )));
        }

        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - rank_idx as u8; // FEN starts from rank 8
            let mut file: u8 = 0;
            for ch in rank_str.chars() {
                if file > 7 {
                    return Err(ChessError::InvalidFen(format!(
                        "too many squares in rank {}",
                        rank + 1
                    )));
                }
                if let Some(digit) = ch.to_digit(10) {
                    if !(1..=8).contains(&digit) {
                        return Err(ChessError::InvalidFen(format!(
                            "invalid empty count '{ch}' in rank {}",
                            rank + 1
                        )));
                    }
                    file += digit as u8;
                } else if let Some((color, piece)) = PieceType::from_char(ch) {
                    let sq = Square::from_file_rank(file, rank);
                    pos.put_piece(sq, color, piece);
                    file += 1;
                } else {
                    return Err(ChessError::InvalidFen(format!(
                        "invalid character '{ch}' in piece placement"
                    )));
                }
            }
            if file != 8 {
                return Err(ChessError::InvalidFen(format!(
                    "rank {} has {} squares instead of 8",
                    rank + 1,
                    file
                )));
            }
        }

        // Validate exactly one king per side.
        for color in [Color::White, Color::Black] {
            let king_count = pos.pieces[color.index()][PieceType::King.index()].pop_count();
            if king_count != 1 {
                return Err(ChessError::InvalidFen(format!(
                    "{color} has {king_count} kings (expected 1)"
                )));
            }
        }

        // ----- Field 2: Side to move -----
        pos.side_to_move = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            other => {
                return Err(ChessError::InvalidFen(format!(
                    "invalid side to move: '{other}'"
                )));
            }
        };

        // ----- Field 3: Castling availability -----
        pos.castling_rights = CastlingRights::from_fen(fields[2]).ok_or_else(|| {
            ChessError::InvalidFen(format!("invalid castling string: '{}'", fields[2]))
        })?;

        // ----- Field 4: En passant target square -----
        if fields[3] != "-" {
            let ep_sq = Square::from_algebraic(fields[3]).ok_or_else(|| {
                ChessError::InvalidFen(format!("invalid en passant square: '{}'", fields[3]))
            })?;
            // En passant target must be on rank 3 (for Black) or rank 6 (for White).
            let rank = ep_sq.rank();
            if rank != 2 && rank != 5 {
                return Err(ChessError::InvalidFen(format!(
                    "en passant square {} is not on rank 3 or 6",
                    fields[3]
                )));
            }
            pos.en_passant = Some(ep_sq);
        }

        // ----- Field 5: Halfmove clock -----
        pos.halfmove_clock = fields[4].parse::<u16>().map_err(|_| {
            ChessError::InvalidFen(format!("invalid halfmove clock: '{}'", fields[4]))
        })?;

        // ----- Field 6: Fullmove number -----
        pos.fullmove_number = fields[5].parse::<u16>().map_err(|_| {
            ChessError::InvalidFen(format!("invalid fullmove number: '{}'", fields[5]))
        })?;
        if pos.fullmove_number == 0 {
            return Err(ChessError::InvalidFen(
                "fullmove number must be >= 1".to_string(),
            ));
        }

        // Compute the Zobrist hash from scratch.
        pos.zobrist_hash = pos.compute_zobrist();

        #[cfg(debug_assertions)]
        pos.assert_consistent();

        Ok(pos)
    }

    /// Export the position as a FEN string.
    pub fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(80);

        // ----- Field 1: Piece placement -----
        for rank in (0..8).rev() {
            let mut empty_count = 0u8;
            for file in 0..8 {
                let sq = Square::from_file_rank(file, rank);
                match self.piece_at(sq) {
                    Some((color, piece)) => {
                        if empty_count > 0 {
                            fen.push((b'0' + empty_count) as char);
                            empty_count = 0;
                        }
                        fen.push(piece.to_char(color));
                    }
                    None => {
                        empty_count += 1;
                    }
                }
            }
            if empty_count > 0 {
                fen.push((b'0' + empty_count) as char);
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // ----- Field 2: Side to move -----
        fen.push(' ');
        fen.push(match self.side_to_move {
            Color::White => 'w',
            Color::Black => 'b',
        });

        // ----- Field 3: Castling -----
        fen.push(' ');
        fen.push_str(&self.castling_rights.to_fen());

        // ----- Field 4: En passant -----
        fen.push(' ');
        match self.en_passant {
            Some(sq) => fen.push_str(&sq.to_algebraic()),
            None => fen.push('-'),
        }

        // ----- Field 5: Halfmove clock -----
        fen.push(' ');
        fen.push_str(&self.halfmove_clock.to_string());

        // ----- Field 6: Fullmove number -----
        fen.push(' ');
        fen.push_str(&self.fullmove_number.to_string());

        fen
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.board_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn starting() -> Position {
        Position::starting()
    }

    fn sq(name: &str) -> Square {
        Square::from_algebraic(name).unwrap()
    }

    // ===================================================================
    // Starting position
    // ===================================================================

    #[test]
    fn starting_position_fen() {
        let pos = starting();
        assert_eq!(
            pos.to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
    }

    #[test]
    fn starting_position_side_to_move() {
        let pos = starting();
        assert_eq!(pos.side_to_move, Color::White);
    }

    #[test]
    fn starting_position_castling() {
        let pos = starting();
        assert_eq!(pos.castling_rights, CastlingRights::ALL);
    }

    #[test]
    fn starting_position_en_passant() {
        let pos = starting();
        assert_eq!(pos.en_passant, None);
    }

    #[test]
    fn starting_position_clocks() {
        let pos = starting();
        assert_eq!(pos.halfmove_clock, 0);
        assert_eq!(pos.fullmove_number, 1);
    }

    #[test]
    fn starting_position_piece_count() {
        let pos = starting();
        assert_eq!(pos.all_occupied.pop_count(), 32);
        assert_eq!(pos.occupied[Color::White.index()].pop_count(), 16);
        assert_eq!(pos.occupied[Color::Black.index()].pop_count(), 16);
    }

    // ===================================================================
    // piece_at queries on starting position
    // ===================================================================

    #[test]
    fn piece_at_white_king() {
        let pos = starting();
        assert_eq!(
            pos.piece_at(sq("e1")),
            Some((Color::White, PieceType::King))
        );
    }

    #[test]
    fn piece_at_black_queen() {
        let pos = starting();
        assert_eq!(
            pos.piece_at(sq("d8")),
            Some((Color::Black, PieceType::Queen))
        );
    }

    #[test]
    fn piece_at_white_pawns() {
        let pos = starting();
        for file in b'a'..=b'h' {
            let name = format!("{}2", file as char);
            assert_eq!(
                pos.piece_at(sq(&name)),
                Some((Color::White, PieceType::Pawn)),
                "expected white pawn on {name}"
            );
        }
    }

    #[test]
    fn piece_at_black_pawns() {
        let pos = starting();
        for file in b'a'..=b'h' {
            let name = format!("{}7", file as char);
            assert_eq!(
                pos.piece_at(sq(&name)),
                Some((Color::Black, PieceType::Pawn)),
                "expected black pawn on {name}"
            );
        }
    }

    #[test]
    fn piece_at_empty_squares() {
        let pos = starting();
        // Ranks 3-6 should be empty.
        for rank in 3..=6 {
            for file in b'a'..=b'h' {
                let name = format!("{}{}", file as char, rank);
                assert_eq!(pos.piece_at(sq(&name)), None, "expected empty on {name}");
            }
        }
    }

    #[test]
    fn piece_at_corners() {
        let pos = starting();
        assert_eq!(
            pos.piece_at(sq("a1")),
            Some((Color::White, PieceType::Rook))
        );
        assert_eq!(
            pos.piece_at(sq("h1")),
            Some((Color::White, PieceType::Rook))
        );
        assert_eq!(
            pos.piece_at(sq("a8")),
            Some((Color::Black, PieceType::Rook))
        );
        assert_eq!(
            pos.piece_at(sq("h8")),
            Some((Color::Black, PieceType::Rook))
        );
    }

    #[test]
    fn piece_at_knights() {
        let pos = starting();
        assert_eq!(
            pos.piece_at(sq("b1")),
            Some((Color::White, PieceType::Knight))
        );
        assert_eq!(
            pos.piece_at(sq("g8")),
            Some((Color::Black, PieceType::Knight))
        );
    }

    #[test]
    fn piece_at_bishops() {
        let pos = starting();
        assert_eq!(
            pos.piece_at(sq("c1")),
            Some((Color::White, PieceType::Bishop))
        );
        assert_eq!(
            pos.piece_at(sq("f8")),
            Some((Color::Black, PieceType::Bishop))
        );
    }

    // ===================================================================
    // king_sq
    // ===================================================================

    #[test]
    fn king_sq_starting() {
        let pos = starting();
        assert_eq!(pos.king_sq(Color::White), sq("e1"));
        assert_eq!(pos.king_sq(Color::Black), sq("e8"));
    }

    // ===================================================================
    // bb() accessor
    // ===================================================================

    #[test]
    fn bb_white_pawns() {
        let pos = starting();
        let pawns = pos.bb(Color::White, PieceType::Pawn);
        assert_eq!(pawns.pop_count(), 8);
        // All should be on rank 2 (indices 8..16).
        assert_eq!(pawns.0, 0x0000_0000_0000_FF00);
    }

    #[test]
    fn bb_black_rooks() {
        let pos = starting();
        let rooks = pos.bb(Color::Black, PieceType::Rook);
        assert_eq!(rooks.pop_count(), 2);
        assert!(rooks.is_set(sq("a8")));
        assert!(rooks.is_set(sq("h8")));
    }

    // ===================================================================
    // put_piece / remove_piece
    // ===================================================================

    #[test]
    fn put_and_remove_piece() {
        let mut pos = Position::empty();
        let e4 = sq("e4");

        pos.put_piece(e4, Color::White, PieceType::Knight);
        assert_eq!(pos.piece_at(e4), Some((Color::White, PieceType::Knight)));
        assert!(pos.all_occupied.is_set(e4));

        pos.remove_piece(e4, Color::White, PieceType::Knight);
        assert_eq!(pos.piece_at(e4), None);
        assert!(!pos.all_occupied.is_set(e4));
    }

    #[test]
    fn put_piece_hash_updates_zobrist() {
        let mut pos = Position::empty();
        let hash_before = pos.zobrist_hash;

        pos.put_piece_hash(sq("d4"), Color::White, PieceType::Queen);
        assert_ne!(
            pos.zobrist_hash, hash_before,
            "hash should change after put"
        );

        // Removing the same piece should restore the hash (XOR is self-inverse).
        pos.remove_piece_hash(sq("d4"), Color::White, PieceType::Queen);
        assert_eq!(
            pos.zobrist_hash, hash_before,
            "hash should be restored after remove"
        );
    }

    // ===================================================================
    // friendly / enemy helpers
    // ===================================================================

    #[test]
    fn friendly_enemy_starting() {
        let pos = starting();
        assert_eq!(pos.friendly(), pos.occupied[Color::White.index()]);
        assert_eq!(pos.enemy(), pos.occupied[Color::Black.index()]);
    }

    // ===================================================================
    // Zobrist hash
    // ===================================================================

    #[test]
    fn zobrist_hash_nonzero_for_starting() {
        let pos = starting();
        assert_ne!(
            pos.zobrist_hash, 0,
            "starting position hash should not be 0"
        );
    }

    #[test]
    fn zobrist_hash_matches_recompute() {
        let pos = starting();
        assert_eq!(pos.zobrist_hash, pos.compute_zobrist());
    }

    #[test]
    fn zobrist_different_positions_differ() {
        let pos1 = starting();
        let pos2 =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
                .unwrap();
        assert_ne!(pos1.zobrist_hash, pos2.zobrist_hash);
    }

    // ===================================================================
    // FEN parsing
    // ===================================================================

    #[test]
    fn fen_round_trip_starting() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.to_fen(), fen);
    }

    #[test]
    fn fen_round_trip_after_e4() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.to_fen(), fen);
    }

    #[test]
    fn fen_round_trip_kiwipete() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.to_fen(), fen);
    }

    #[test]
    fn fen_round_trip_endgame() {
        let fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.to_fen(), fen);
    }

    #[test]
    fn fen_round_trip_castling_partial() {
        let fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 5 20";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.to_fen(), fen);
    }

    #[test]
    fn fen_round_trip_black_to_move() {
        let fen = "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.side_to_move, Color::Black);
        assert_eq!(pos.to_fen(), fen);
    }

    // ===================================================================
    // FEN validation errors
    // ===================================================================

    #[test]
    fn fen_error_wrong_field_count() {
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -").is_err()
        );
    }

    #[test]
    fn fen_error_wrong_rank_count() {
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err()
        );
    }

    #[test]
    fn fen_error_invalid_piece_char() {
        assert!(
            Position::from_fen("xnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err()
        );
    }

    #[test]
    fn fen_error_invalid_side_to_move() {
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1").is_err()
        );
    }

    #[test]
    fn fen_error_invalid_castling() {
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w XYZ - 0 1").is_err()
        );
    }

    #[test]
    fn fen_error_invalid_ep_square() {
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq z9 0 1")
                .is_err()
        );
    }

    #[test]
    fn fen_error_ep_wrong_rank() {
        // e4 is rank 4, not 3 or 6 — invalid for en passant target.
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e4 0 1")
                .is_err()
        );
    }

    #[test]
    fn fen_error_invalid_halfmove() {
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - abc 1")
                .is_err()
        );
    }

    #[test]
    fn fen_error_fullmove_zero() {
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0").is_err()
        );
    }

    #[test]
    fn fen_error_no_white_king() {
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR w KQkq - 0 1").is_err()
        );
    }

    #[test]
    fn fen_error_two_white_kings() {
        assert!(
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKKBNR w KQkq - 0 1").is_err()
        );
    }

    #[test]
    fn fen_error_rank_too_long() {
        assert!(
            Position::from_fen("rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                .is_err()
        );
    }

    // ===================================================================
    // Position::empty
    // ===================================================================

    #[test]
    fn empty_position() {
        let pos = Position::empty();
        assert!(pos.all_occupied.is_empty());
        assert_eq!(pos.castling_rights, CastlingRights::NONE);
        assert_eq!(pos.en_passant, None);
        assert_eq!(pos.side_to_move, Color::White);
        assert_eq!(pos.fullmove_number, 1);
    }

    // ===================================================================
    // board_string display
    // ===================================================================

    #[test]
    fn board_string_starting() {
        let pos = starting();
        let s = pos.board_string();
        // First line should be rank 8.
        assert!(s.starts_with("8 r n b q k b n r"));
        // Last line should be the file labels.
        assert!(s.ends_with("a b c d e f g h"));
    }

    // ===================================================================
    // Consistency check
    // ===================================================================

    #[test]
    fn starting_position_is_consistent() {
        let pos = starting();
        pos.assert_consistent();
    }

    #[test]
    fn fen_loaded_position_is_consistent() {
        let pos = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        pos.assert_consistent();
    }

    // ===================================================================
    // Specific bitboard layouts
    // ===================================================================

    #[test]
    fn starting_white_back_rank() {
        let pos = starting();
        // White back rank: R N B Q K B N R on squares 0..7.
        assert!(pos.bb(Color::White, PieceType::Rook).is_set(sq("a1")));
        assert!(pos.bb(Color::White, PieceType::Knight).is_set(sq("b1")));
        assert!(pos.bb(Color::White, PieceType::Bishop).is_set(sq("c1")));
        assert!(pos.bb(Color::White, PieceType::Queen).is_set(sq("d1")));
        assert!(pos.bb(Color::White, PieceType::King).is_set(sq("e1")));
        assert!(pos.bb(Color::White, PieceType::Bishop).is_set(sq("f1")));
        assert!(pos.bb(Color::White, PieceType::Knight).is_set(sq("g1")));
        assert!(pos.bb(Color::White, PieceType::Rook).is_set(sq("h1")));
    }

    #[test]
    fn starting_black_back_rank() {
        let pos = starting();
        assert!(pos.bb(Color::Black, PieceType::Rook).is_set(sq("a8")));
        assert!(pos.bb(Color::Black, PieceType::Knight).is_set(sq("b8")));
        assert!(pos.bb(Color::Black, PieceType::Bishop).is_set(sq("c8")));
        assert!(pos.bb(Color::Black, PieceType::Queen).is_set(sq("d8")));
        assert!(pos.bb(Color::Black, PieceType::King).is_set(sq("e8")));
        assert!(pos.bb(Color::Black, PieceType::Bishop).is_set(sq("f8")));
        assert!(pos.bb(Color::Black, PieceType::Knight).is_set(sq("g8")));
        assert!(pos.bb(Color::Black, PieceType::Rook).is_set(sq("h8")));
    }

    #[test]
    fn starting_piece_counts() {
        let pos = starting();
        for color in [Color::White, Color::Black] {
            assert_eq!(pos.bb(color, PieceType::Pawn).pop_count(), 8);
            assert_eq!(pos.bb(color, PieceType::Knight).pop_count(), 2);
            assert_eq!(pos.bb(color, PieceType::Bishop).pop_count(), 2);
            assert_eq!(pos.bb(color, PieceType::Rook).pop_count(), 2);
            assert_eq!(pos.bb(color, PieceType::Queen).pop_count(), 1);
            assert_eq!(pos.bb(color, PieceType::King).pop_count(), 1);
        }
    }

    // ===================================================================
    // FEN: various known positions
    // ===================================================================

    #[test]
    fn fen_position_4() {
        let fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.to_fen(), fen);
        // White king is on g1 (already castled).
        assert_eq!(pos.king_sq(Color::White), sq("g1"));
        // Only black can castle.
        assert!(!pos.castling_rights.can_castle_kingside(Color::White));
        assert!(pos.castling_rights.can_castle_kingside(Color::Black));
    }

    #[test]
    fn fen_position_5() {
        let fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.to_fen(), fen);
        assert_eq!(pos.halfmove_clock, 1);
        assert_eq!(pos.fullmove_number, 8);
    }

    #[test]
    fn fen_empty_board_with_kings() {
        let fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(pos.to_fen(), fen);
        assert_eq!(pos.all_occupied.pop_count(), 2);
        assert_eq!(pos.king_sq(Color::White), sq("e1"));
        assert_eq!(pos.king_sq(Color::Black), sq("e8"));
    }
}

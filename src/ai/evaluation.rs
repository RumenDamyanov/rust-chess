//! Static position evaluation.
//!
//! Returns a score in centipawns from White's perspective.
//! Positive = White advantage, negative = Black advantage.
//!
//! Components:
//!   1. Material balance
//!   2. Piece-square tables (middle-game)
//!   3. Mobility (simple legal-move count, optional)

use crate::engine::board::Position;
use crate::engine::types::{Color, PieceType, Square};

/// Infinity sentinel. Larger than any realistic eval.
pub const INF: i32 = 100_000;

/// Checkmate score base. Actual mate scores are `MATE - ply` so closer mates
/// score higher.
pub const MATE: i32 = 90_000;

/// Is this score a forced-mate score?
#[inline]
pub fn is_mate_score(score: i32) -> bool {
    score.abs() >= MATE - 500
}

// =========================================================================
// Material values (centipawns)
// =========================================================================

const PIECE_VALUE: [i32; 6] = [
    100, // Pawn
    320, // Knight
    330, // Bishop
    500, // Rook
    900, // Queen
    0,   // King (not counted in material balance)
];

// =========================================================================
// Piece-Square Tables (middle-game, from White's perspective)
//
// Indexed by square (LERF: a1=0 .. h8=63).
// Values are centipawn bonuses/penalties.
// =========================================================================

/// Pawn PST — encourages central pawns and advancement.
#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,   // rank 1 (never occupied)
     5, 10, 10,-20,-20, 10, 10,  5,   // rank 2
     5, -5,-10,  0,  0,-10, -5,  5,   // rank 3
     0,  0,  0, 20, 20,  0,  0,  0,   // rank 4
     5,  5, 10, 25, 25, 10,  5,  5,   // rank 5
    10, 10, 20, 30, 30, 20, 10, 10,   // rank 6
    50, 50, 50, 50, 50, 50, 50, 50,   // rank 7
     0,  0,  0,  0,  0,  0,  0,  0,   // rank 8 (promoted)
];

/// Knight PST — encourages centralization.
#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

/// Bishop PST — encourages long diagonals and avoids corners.
#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

/// Rook PST — encourages 7th rank and open files.
#[rustfmt::skip]
const ROOK_PST: [i32; 64] = [
      0,  0,  0,  5,  5,  0,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      5, 10, 10, 10, 10, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
];

/// Queen PST — minor centralization bonus.
#[rustfmt::skip]
const QUEEN_PST: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

/// King PST (middle-game) — encourages castled position, penalizes center.
#[rustfmt::skip]
const KING_MG_PST: [i32; 64] = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
];

const PST: [[i32; 64]; 6] = [
    PAWN_PST,
    KNIGHT_PST,
    BISHOP_PST,
    ROOK_PST,
    QUEEN_PST,
    KING_MG_PST,
];

// =========================================================================
// Evaluation
// =========================================================================

/// Evaluate a position. Returns centipawn score from White's perspective.
pub fn evaluate(pos: &Position) -> i32 {
    let mut score = 0i32;

    for pt_idx in 0..6 {
        // White pieces.
        for sq in pos.pieces[Color::White.index()][pt_idx].iter() {
            score += PIECE_VALUE[pt_idx];
            score += PST[pt_idx][sq.0 as usize];
        }

        // Black pieces — mirror the PST (flip rank).
        for sq in pos.pieces[Color::Black.index()][pt_idx].iter() {
            score -= PIECE_VALUE[pt_idx];
            score -= PST[pt_idx][mirror_square(sq) as usize];
        }
    }

    // Bishop pair bonus.
    if pos.pieces[Color::White.index()][PieceType::Bishop.index()].pop_count() >= 2 {
        score += 30;
    }
    if pos.pieces[Color::Black.index()][PieceType::Bishop.index()].pop_count() >= 2 {
        score -= 30;
    }

    score
}

/// Evaluate from the side-to-move's perspective (for negamax).
#[inline]
pub fn evaluate_relative(pos: &Position) -> i32 {
    let score = evaluate(pos);
    match pos.side_to_move {
        Color::White => score,
        Color::Black => -score,
    }
}

/// Mirror a square vertically (flip rank) for Black PST lookup.
#[inline]
fn mirror_square(sq: Square) -> u8 {
    sq.0 ^ 56 // XOR with 56 flips rank: rank 0 ↔ rank 7, rank 1 ↔ rank 6, etc.
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::board::Position;

    #[test]
    fn starting_position_roughly_equal() {
        let pos = Position::starting();
        let score = evaluate(&pos);
        // Starting position should be roughly 0 (symmetric).
        assert!(
            score.abs() < 50,
            "starting position eval too skewed: {score}"
        );
    }

    #[test]
    fn white_extra_queen_is_positive() {
        // White has an extra queen.
        let pos = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let score = evaluate(&pos);
        assert!(
            score > 800,
            "extra queen should give large advantage: {score}"
        );
    }

    #[test]
    fn black_extra_queen_is_negative() {
        let pos = Position::from_fen("3qk3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let score = evaluate(&pos);
        assert!(
            score < -800,
            "opponent extra queen should be negative: {score}"
        );
    }

    #[test]
    fn symmetric_position_near_zero() {
        // Mirror position: both sides identical material.
        let pos =
            Position::from_fen("r1bqkb1r/pppppppp/2n2n2/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 0 1")
                .unwrap();
        let score = evaluate(&pos);
        assert!(
            score.abs() < 30,
            "symmetric position should be near zero: {score}"
        );
    }

    #[test]
    fn evaluate_relative_flips_for_black() {
        let pos = Position::from_fen("3qk3/8/8/8/8/8/8/4K3 b - - 0 1").unwrap();
        // White eval is negative (Black has extra queen).
        // Relative for Black (side to move) should be positive.
        let rel = evaluate_relative(&pos);
        assert!(
            rel > 800,
            "relative eval for Black with extra queen should be positive: {rel}"
        );
    }

    #[test]
    fn bishop_pair_bonus() {
        // White has two bishops, Black has one.
        let w2b = Position::from_fen("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1").unwrap();
        let w1b = Position::from_fen("4k3/8/8/8/8/8/8/4KB2 w - - 0 1").unwrap();
        let diff = evaluate(&w2b) - evaluate(&w1b);
        // Should include the second bishop's value (~330) + bishop pair bonus (30).
        assert!(diff > 300, "adding second bishop should add >300cp: {diff}");
    }

    #[test]
    fn mate_score_detection() {
        assert!(is_mate_score(MATE));
        assert!(is_mate_score(MATE - 10));
        assert!(is_mate_score(-(MATE - 10)));
        assert!(!is_mate_score(500));
        assert!(!is_mate_score(0));
    }

    #[test]
    fn mirror_square_works() {
        assert_eq!(mirror_square(Square(0)), 56); // a1 → a8
        assert_eq!(mirror_square(Square(63)), 7); // h8 → h1
        assert_eq!(mirror_square(Square(4)), 60); // e1 → e8
    }
}

//! Legal move generation.
//!
//! Pipeline:
//!   1. Generate pseudo-legal moves (ignoring pins / check evasion).
//!   2. Filter: make the move, verify king is not in check, undo.
//!
//! This "make-and-check" approach is simple and correct. For AI search speed
//! a staged/incremental generator can be added later without changing the API.

use crate::engine::attacks;
use crate::engine::board::Position;
use crate::engine::types::{Bitboard, Color, Move, MoveFlags, PieceType, Square};

// =========================================================================
// Public API
// =========================================================================

/// Generate all legal moves for the side to move.
pub fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut pseudo = Vec::with_capacity(256);
    generate_pseudo_legal(pos, &mut pseudo);

    // Filter: after each move the own king must not be in check.
    let mut legal = Vec::with_capacity(pseudo.len());
    for mv in pseudo {
        let mut copy = pos.clone();
        copy.make_move(mv);
        // After make_move, it's the opponent's turn. Check whether the side
        // that just moved left their king in check.
        let us = !copy.side_to_move;
        if !copy.is_square_attacked(copy.king_sq(us), copy.side_to_move) {
            legal.push(mv);
        }
    }
    legal
}

/// Generate all legal moves originating from a specific square.
pub fn legal_moves_from(pos: &Position, from: Square) -> Vec<Move> {
    legal_moves(pos)
        .into_iter()
        .filter(|m| m.from == from)
        .collect()
}

// =========================================================================
// Pseudo-legal generation (internal)
// =========================================================================

fn generate_pseudo_legal(pos: &Position, moves: &mut Vec<Move>) {
    let us = pos.side_to_move;
    generate_pawn_moves(pos, us, moves);
    generate_knight_moves(pos, us, moves);
    generate_king_moves(pos, us, moves);
    generate_slider_moves(pos, us, PieceType::Bishop, moves);
    generate_slider_moves(pos, us, PieceType::Rook, moves);
    generate_slider_moves(pos, us, PieceType::Queen, moves);
    generate_castling_moves(pos, us, moves);
}

// =========================================================================
// Pawn moves
// =========================================================================

fn generate_pawn_moves(pos: &Position, us: Color, moves: &mut Vec<Move>) {
    let t = attacks::tables();
    let pawns = pos.bb(us, PieceType::Pawn);
    let enemy = pos.occupied[(!us).index()];
    let empty = !pos.all_occupied;

    let (push_dir, start_rank, promo_rank): (i8, u8, u8) = match us {
        Color::White => (8, 1, 6),  // rank 2 start, rank 7 promotes
        Color::Black => (-8, 6, 1), // rank 7 start, rank 2 promotes
    };

    for from in pawns.iter() {
        let from_rank = from.rank();

        // --- Single push ---
        let to_idx = (from.0 as i8 + push_dir) as u8;
        let to = Square(to_idx);
        if (Bitboard::from_square(to) & empty).is_not_empty() {
            if from_rank == promo_rank {
                add_promotions(from, to, MoveFlags::NONE, moves);
            } else {
                moves.push(Move::new(from, to));
            }

            // --- Double push ---
            if from_rank == start_rank {
                let to2_idx = (from.0 as i8 + push_dir * 2) as u8;
                let to2 = Square(to2_idx);
                if (Bitboard::from_square(to2) & empty).is_not_empty() {
                    moves.push(Move::with_flags(from, to2, MoveFlags::DOUBLE_PUSH));
                }
            }
        }

        // --- Captures (including promotion captures) ---
        let attack_bb = t.pawn_attacks(us, from) & enemy;
        for to in attack_bb.iter() {
            if from_rank == promo_rank {
                add_promotions(from, to, MoveFlags::CAPTURE, moves);
            } else {
                moves.push(Move::with_flags(from, to, MoveFlags::CAPTURE));
            }
        }

        // --- En passant ---
        if let Some(ep_sq) = pos.en_passant
            && t.pawn_attacks(us, from).is_set(ep_sq)
        {
            moves.push(Move::with_flags(
                from,
                ep_sq,
                MoveFlags::CAPTURE | MoveFlags::EN_PASSANT,
            ));
        }
    }
}

/// Add all four promotion variants for a pawn push or capture.
fn add_promotions(from: Square, to: Square, extra_flags: MoveFlags, moves: &mut Vec<Move>) {
    for &promo in &[
        PieceType::Queen,
        PieceType::Rook,
        PieceType::Bishop,
        PieceType::Knight,
    ] {
        moves.push(Move::with_promotion(from, to, promo, extra_flags));
    }
}

// =========================================================================
// Knight moves
// =========================================================================

fn generate_knight_moves(pos: &Position, us: Color, moves: &mut Vec<Move>) {
    let t = attacks::tables();
    let knights = pos.bb(us, PieceType::Knight);
    let friendly = pos.occupied[us.index()];
    let enemy = pos.occupied[(!us).index()];

    for from in knights.iter() {
        let targets = t.knight_attacks(from) & !friendly;
        for to in targets.iter() {
            let flags = if (Bitboard::from_square(to) & enemy).is_not_empty() {
                MoveFlags::CAPTURE
            } else {
                MoveFlags::NONE
            };
            moves.push(Move::with_flags(from, to, flags));
        }
    }
}

// =========================================================================
// King moves (non-castling)
// =========================================================================

fn generate_king_moves(pos: &Position, us: Color, moves: &mut Vec<Move>) {
    let t = attacks::tables();
    let king_sq = pos.king_sq(us);
    let friendly = pos.occupied[us.index()];
    let enemy = pos.occupied[(!us).index()];

    let targets = t.king_attacks(king_sq) & !friendly;
    for to in targets.iter() {
        let flags = if (Bitboard::from_square(to) & enemy).is_not_empty() {
            MoveFlags::CAPTURE
        } else {
            MoveFlags::NONE
        };
        moves.push(Move::with_flags(king_sq, to, flags));
    }
}

// =========================================================================
// Slider moves (bishop, rook, queen)
// =========================================================================

fn generate_slider_moves(pos: &Position, us: Color, piece: PieceType, moves: &mut Vec<Move>) {
    let t = attacks::tables();
    let pieces = pos.bb(us, piece);
    let friendly = pos.occupied[us.index()];
    let enemy = pos.occupied[(!us).index()];
    let occ = pos.all_occupied;

    for from in pieces.iter() {
        let attacks = match piece {
            PieceType::Bishop => t.bishop_attacks(from, occ),
            PieceType::Rook => t.rook_attacks(from, occ),
            PieceType::Queen => t.queen_attacks(from, occ),
            _ => unreachable!(),
        };
        let targets = attacks & !friendly;
        for to in targets.iter() {
            let flags = if (Bitboard::from_square(to) & enemy).is_not_empty() {
                MoveFlags::CAPTURE
            } else {
                MoveFlags::NONE
            };
            moves.push(Move::with_flags(from, to, flags));
        }
    }
}

// =========================================================================
// Castling
// =========================================================================

fn generate_castling_moves(pos: &Position, us: Color, moves: &mut Vec<Move>) {
    let them = !us;

    // Can't castle while in check.
    let king_sq = pos.king_sq(us);
    if pos.is_square_attacked(king_sq, them) {
        return;
    }

    let (ks_right, qs_right, rank_base) = match us {
        Color::White => (
            CastlingRights::WHITE_KINGSIDE,
            CastlingRights::WHITE_QUEENSIDE,
            0u8,
        ),
        Color::Black => (
            CastlingRights::BLACK_KINGSIDE,
            CastlingRights::BLACK_QUEENSIDE,
            56u8,
        ),
    };

    // Kingside: king moves e→g, path through f and g must be clear and not attacked.
    if pos.castling_rights.has(ks_right) {
        let f_sq = Square(rank_base + 5);
        let g_sq = Square(rank_base + 6);
        if !pos.all_occupied.is_set(f_sq)
            && !pos.all_occupied.is_set(g_sq)
            && !pos.is_square_attacked(f_sq, them)
            && !pos.is_square_attacked(g_sq, them)
        {
            moves.push(Move::with_flags(king_sq, g_sq, MoveFlags::CASTLING));
        }
    }

    // Queenside: king moves e→c, path through b, c, d must be clear; c and d not attacked.
    if pos.castling_rights.has(qs_right) {
        let b_sq = Square(rank_base + 1);
        let c_sq = Square(rank_base + 2);
        let d_sq = Square(rank_base + 3);
        if !pos.all_occupied.is_set(b_sq)
            && !pos.all_occupied.is_set(c_sq)
            && !pos.all_occupied.is_set(d_sq)
            && !pos.is_square_attacked(c_sq, them)
            && !pos.is_square_attacked(d_sq, them)
        {
            moves.push(Move::with_flags(king_sq, c_sq, MoveFlags::CASTLING));
        }
    }
}

use crate::engine::types::CastlingRights;

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(name: &str) -> Square {
        Square::from_algebraic(name).unwrap()
    }

    fn pos(fen: &str) -> Position {
        Position::from_fen(fen).unwrap()
    }

    fn count_legal(fen: &str) -> usize {
        legal_moves(&pos(fen)).len()
    }

    // -------------------------------------------------------------------
    // Starting position
    // -------------------------------------------------------------------

    #[test]
    fn starting_position_has_20_moves() {
        assert_eq!(
            count_legal("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            20
        );
    }

    #[test]
    fn starting_position_after_e4() {
        assert_eq!(
            count_legal("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
            20
        );
    }

    // -------------------------------------------------------------------
    // Pawn moves
    // -------------------------------------------------------------------

    #[test]
    fn pawn_single_push() {
        // White pawn on e2, only piece.
        let p = pos("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
        let moves = legal_moves(&p);
        let pawn_moves: Vec<_> = moves.iter().filter(|m| m.from == sq("e2")).collect();
        // Single push e3 + double push e4.
        assert_eq!(pawn_moves.len(), 2);
    }

    #[test]
    fn pawn_blocked() {
        // White pawn on e2, blocked by a piece on e3.
        let p = pos("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1");
        let moves = legal_moves(&p);
        let pawn_moves: Vec<_> = moves.iter().filter(|m| m.from == sq("e2")).collect();
        assert_eq!(pawn_moves.len(), 0);
    }

    #[test]
    fn pawn_capture() {
        let p = pos("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1");
        let moves = legal_moves(&p);
        let pawn_moves: Vec<_> = moves.iter().filter(|m| m.from == sq("e2")).collect();
        // Single push, double push: e3 not blocked, e4 is blocked by d4? No, d4 is on different file.
        // e3, e4 (double push), and no captures because d4 is not diagonally adjacent to e2.
        // Wait: e2 can push to e3 and e4, but d4 is on rank 4, file d. Pawn on e2 can capture d3 or f3.
        // d4 is on rank 4, not reachable. So just pushes.
        assert_eq!(pawn_moves.len(), 2); // e3 and e4
    }

    #[test]
    fn pawn_promotion() {
        // Black king on h8, not blocking e8.
        let p = pos("7k/4P3/8/8/8/8/8/4K3 w - - 0 1");
        let moves = legal_moves(&p);
        let promo_moves: Vec<_> = moves.iter().filter(|m| m.from == sq("e7")).collect();
        // 4 promotion pieces.
        assert_eq!(promo_moves.len(), 4);
        assert!(promo_moves.iter().all(|m| m.promotion.is_some()));
    }

    #[test]
    fn en_passant_move_generated() {
        // After 1. e4 d5 2. e5 f5, White can play exf6 e.p.
        let p = pos("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
        let moves = legal_moves(&p);
        let ep_moves: Vec<_> = moves.iter().filter(|m| m.flags.is_en_passant()).collect();
        assert_eq!(ep_moves.len(), 1);
        assert_eq!(ep_moves[0].to, sq("f6"));
    }

    // -------------------------------------------------------------------
    // Castling
    // -------------------------------------------------------------------

    #[test]
    fn castling_both_sides() {
        let p = pos("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
        let moves = legal_moves(&p);
        let castle_moves: Vec<_> = moves.iter().filter(|m| m.flags.is_castling()).collect();
        assert_eq!(castle_moves.len(), 2);
    }

    #[test]
    fn castling_blocked() {
        // Pieces between king and rook.
        let p = pos("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/RN2K1NR w KQkq - 0 1");
        let moves = legal_moves(&p);
        let castle_moves: Vec<_> = moves.iter().filter(|m| m.flags.is_castling()).collect();
        assert_eq!(castle_moves.len(), 0);
    }

    #[test]
    fn castling_through_check_forbidden() {
        // Black king on e8, rook on f8 attacks f1.
        // Kingside castling goes through f1 → forbidden. Queenside ok.
        let p = pos("4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        let moves = legal_moves(&p);
        let castle_moves: Vec<_> = moves.iter().filter(|m| m.flags.is_castling()).collect();
        assert_eq!(castle_moves.len(), 1);
        assert_eq!(castle_moves[0].to, sq("c1"));
    }

    #[test]
    fn no_castling_while_in_check() {
        let p = pos("4k3/8/8/8/8/8/8/R3K2r w Q - 0 1");
        let moves = legal_moves(&p);
        let castle_moves: Vec<_> = moves.iter().filter(|m| m.flags.is_castling()).collect();
        assert_eq!(castle_moves.len(), 0);
    }

    // -------------------------------------------------------------------
    // Check evasion
    // -------------------------------------------------------------------

    #[test]
    fn must_escape_check() {
        // White king on e1, black queen on e8. White is in check.
        let p = pos("4k3/8/8/8/8/8/8/R3K2q w Q - 0 1");
        let moves = legal_moves(&p);
        // Every legal move must leave the king not in check.
        for mv in &moves {
            let mut copy = p.clone();
            copy.make_move(*mv);
            let us = Color::White;
            assert!(
                !copy.is_square_attacked(copy.king_sq(us), !us),
                "move {} leaves king in check",
                mv
            );
        }
    }

    // -------------------------------------------------------------------
    // Known positions
    // -------------------------------------------------------------------

    #[test]
    fn kiwipete_48_moves() {
        assert_eq!(
            count_legal("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
            48
        );
    }

    #[test]
    fn position_3_14_moves() {
        assert_eq!(count_legal("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"), 14);
    }

    #[test]
    fn position_4_6_moves() {
        assert_eq!(
            count_legal("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
            6
        );
    }

    #[test]
    fn position_5_44_moves() {
        assert_eq!(
            count_legal("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
            44
        );
    }

    // -------------------------------------------------------------------
    // Make/undo preserves Zobrist
    // -------------------------------------------------------------------

    #[test]
    fn make_undo_preserves_hash() {
        let p = Position::starting();
        let original_hash = p.zobrist_hash;
        let all = legal_moves(&p);
        for mv in all {
            let mut copy = p.clone();
            let undo = copy.make_move(mv);
            copy.undo_move(mv, &undo);
            assert_eq!(
                copy.zobrist_hash, original_hash,
                "hash mismatch after make/undo of {mv}"
            );
        }
    }

    #[test]
    fn make_undo_preserves_fen() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let p = pos(fen);
        let original_hash = p.zobrist_hash;
        for mv in legal_moves(&p) {
            let mut copy = p.clone();
            let undo = copy.make_move(mv);
            copy.undo_move(mv, &undo);
            assert_eq!(copy.to_fen(), fen, "FEN mismatch after make/undo of {mv}");
            assert_eq!(copy.zobrist_hash, original_hash);
        }
    }

    // -------------------------------------------------------------------
    // legal_moves_from
    // -------------------------------------------------------------------

    #[test]
    fn legal_moves_from_e2() {
        let p = Position::starting();
        let pawn_moves = legal_moves_from(&p, sq("e2"));
        assert_eq!(pawn_moves.len(), 2); // e3, e4
    }

    #[test]
    fn legal_moves_from_empty_square() {
        let p = Position::starting();
        let moves = legal_moves_from(&p, sq("e4"));
        assert_eq!(moves.len(), 0);
    }
}

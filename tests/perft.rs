//! Perft (PERFormance Test) — exhaustive move-generation correctness suite.
//!
//! Each test verifies that the number of leaf nodes at a given depth matches
//! known-correct values for standard positions.  If perft is wrong at any
//! depth, there is a bug in move generation, make/undo, or legality filtering.
//!
//! Reference: <https://www.chessprogramming.org/Perft_Results>

use rust_chess::engine::board::Position;
use rust_chess::engine::movegen::legal_moves;

/// Recursive perft: count leaf nodes at `depth`.
fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = legal_moves(pos);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0u64;
    for mv in moves {
        let mut child = pos.clone();
        child.make_move(mv);
        nodes += perft(&child, depth - 1);
    }
    nodes
}

// =====================================================================
// Position 1 — Starting position
// =====================================================================

#[test]
fn perft_start_depth_1() {
    let pos = Position::starting();
    assert_eq!(perft(&pos, 1), 20);
}

#[test]
fn perft_start_depth_2() {
    let pos = Position::starting();
    assert_eq!(perft(&pos, 2), 400);
}

#[test]
fn perft_start_depth_3() {
    let pos = Position::starting();
    assert_eq!(perft(&pos, 3), 8_902);
}

#[test]
fn perft_start_depth_4() {
    let pos = Position::starting();
    assert_eq!(perft(&pos, 4), 197_281);
}

// =====================================================================
// Position 2 — "Kiwipete" (tricky: castling, EP, pins, promotions)
// =====================================================================

fn kiwipete() -> Position {
    Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        .unwrap()
}

#[test]
fn perft_kiwipete_depth_1() {
    assert_eq!(perft(&kiwipete(), 1), 48);
}

#[test]
fn perft_kiwipete_depth_2() {
    assert_eq!(perft(&kiwipete(), 2), 2_039);
}

#[test]
fn perft_kiwipete_depth_3() {
    assert_eq!(perft(&kiwipete(), 3), 97_862);
}

// =====================================================================
// Position 3
// =====================================================================

fn position_3() -> Position {
    Position::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap()
}

#[test]
fn perft_pos3_depth_1() {
    assert_eq!(perft(&position_3(), 1), 14);
}

#[test]
fn perft_pos3_depth_2() {
    assert_eq!(perft(&position_3(), 2), 191);
}

#[test]
fn perft_pos3_depth_3() {
    assert_eq!(perft(&position_3(), 3), 2_812);
}

#[test]
fn perft_pos3_depth_4() {
    assert_eq!(perft(&position_3(), 4), 43_238);
}

// =====================================================================
// Position 4
// =====================================================================

fn position_4() -> Position {
    Position::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1").unwrap()
}

#[test]
fn perft_pos4_depth_1() {
    assert_eq!(perft(&position_4(), 1), 6);
}

#[test]
fn perft_pos4_depth_2() {
    assert_eq!(perft(&position_4(), 2), 264);
}

#[test]
fn perft_pos4_depth_3() {
    assert_eq!(perft(&position_4(), 3), 9_467);
}

// =====================================================================
// Position 5
// =====================================================================

fn position_5() -> Position {
    Position::from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8").unwrap()
}

#[test]
fn perft_pos5_depth_1() {
    assert_eq!(perft(&position_5(), 1), 44);
}

#[test]
fn perft_pos5_depth_2() {
    assert_eq!(perft(&position_5(), 2), 1_486);
}

#[test]
fn perft_pos5_depth_3() {
    assert_eq!(perft(&position_5(), 3), 62_379);
}

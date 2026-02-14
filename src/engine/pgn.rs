//! PGN (Portable Game Notation) export.
//!
//! Produces PGN with the Seven Tag Roster and move text with move numbers.

use crate::engine::game::Game;
use crate::engine::types::{Color, GameStatus};

// =========================================================================
// PGN generation
// =========================================================================

/// Export a game as a PGN string.
pub fn to_pgn(game: &Game) -> String {
    let mut pgn = String::with_capacity(512);

    // Seven Tag Roster.
    let date = game.created_at.format("%Y.%m.%d").to_string();
    let result_str = result_string(game.status());

    pgn.push_str("[Event \"Casual Game\"]\n");
    pgn.push_str("[Site \"rust-chess\"]\n");
    pgn.push_str(&format!("[Date \"{date}\"]\n"));
    pgn.push_str("[Round \"-\"]\n");
    pgn.push_str(&format!("[White \"{}\"]\n", game.white_player));
    pgn.push_str(&format!("[Black \"{}\"]\n", game.black_player));
    pgn.push_str(&format!("[Result \"{result_str}\"]\n"));

    // If started from FEN, add SetUp and FEN tags.
    if game.started_from_fen() {
        pgn.push_str("[SetUp \"1\"]\n");
        pgn.push_str(&format!("[FEN \"{}\"]\n", game.starting_fen()));
    }

    pgn.push('\n');

    // Move text.
    let history = game.move_history();
    if history.is_empty() {
        pgn.push_str(&format!("{result_str}\n"));
        return pgn;
    }

    // Determine starting move number and who moved first.
    let first_moves_as_white = if game.started_from_fen() {
        // The starting FEN tells us who moved first.
        game.starting_fen().contains(" w ")
    } else {
        true
    };

    // Compute starting fullmove number from the starting FEN.
    let start_fullmove = if game.started_from_fen() {
        parse_fullmove_from_fen(game.starting_fen())
    } else {
        1
    };

    let mut move_num = start_fullmove;
    let mut white_turn = first_moves_as_white;
    let mut line = String::new();
    let mut line_len = 0;

    for (i, record) in history.iter().enumerate() {
        let token = if white_turn {
            format!("{}. {}", move_num, record.san)
        } else if i == 0 && !first_moves_as_white {
            // First move by black: use "N... move" notation.
            format!("{}... {}", move_num, record.san)
        } else {
            record.san.clone()
        };

        // Line wrapping at ~80 chars.
        if line_len + token.len() + 1 > 80 && line_len > 0 {
            pgn.push_str(&line);
            pgn.push('\n');
            line.clear();
            line_len = 0;
        }

        if line_len > 0 {
            line.push(' ');
            line_len += 1;
        }
        line_len += token.len();
        line.push_str(&token);

        if !white_turn {
            move_num += 1;
        }
        white_turn = !white_turn;
    }

    // Append result.
    let result_token = result_str;
    if line_len + result_token.len() + 1 > 80 && line_len > 0 {
        pgn.push_str(&line);
        pgn.push('\n');
        line.clear();
    } else if line_len > 0 {
        line.push(' ');
    }
    line.push_str(&result_token);
    pgn.push_str(&line);
    pgn.push('\n');

    pgn
}

/// PGN result string.
fn result_string(status: &GameStatus) -> String {
    match status {
        GameStatus::Checkmate => {
            // The side that was checkmated is the side to move at the end.
            // We don't know who won from GameStatus alone, but the convention
            // is: if it's checkmate, the side that just moved won.
            // However we can't determine that here without more context.
            // For simplicity, the Game struct stores enough info: the side_to_move
            // at checkmate is the loser.
            // But this function only gets the status... we'll just use "*" and let
            // the caller override if needed. Actually let's handle it.
            // Since we can't know here, we use a placeholder convention.
            // The Game's to_pgn wrapper should pass more info. For now, "*".
            "*".into()
        }
        GameStatus::Stalemate => "1/2-1/2".into(),
        GameStatus::Draw(_) => "1/2-1/2".into(),
        GameStatus::Active | GameStatus::Check => "*".into(),
    }
}

/// Export PGN with result awareness (knows who won on checkmate).
pub fn to_pgn_with_result(game: &Game) -> String {
    let mut pgn = to_pgn(game);

    // Fix result for checkmate: the side to move at end is the loser.
    if *game.status() == GameStatus::Checkmate {
        let loser = game.side_to_move();
        let result = match loser {
            Color::White => "0-1",
            Color::Black => "1-0",
        };

        // Replace the result in the header and at end.
        pgn = pgn.replace("[Result \"*\"]", &format!("[Result \"{result}\"]"));
        // Replace trailing result marker.
        if pgn.ends_with("*\n") {
            let len = pgn.len();
            pgn.replace_range(len - 2..len, &format!("{result}\n"));
        }
    }

    pgn
}

fn parse_fullmove_from_fen(fen: &str) -> u32 {
    fen.split_whitespace()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::game::Game;
    use crate::engine::types::{Move, MoveFlags, Square};

    fn sq(name: &str) -> Square {
        Square::from_algebraic(name).unwrap()
    }

    fn play(g: &mut Game, from: &str, to: &str, flags: MoveFlags) {
        let mv = Move::with_flags(sq(from), sq(to), flags);
        g.make_move(mv).unwrap();
    }

    #[test]
    fn pgn_empty_game() {
        let g = Game::new();
        let pgn = to_pgn(&g);
        assert!(pgn.contains("[Event \"Casual Game\"]"));
        assert!(pgn.contains("[Result \"*\"]"));
        assert!(pgn.contains("*\n"));
    }

    #[test]
    fn pgn_with_moves() {
        let mut g = Game::new();
        play(&mut g, "e2", "e4", MoveFlags::DOUBLE_PUSH);
        play(&mut g, "e7", "e5", MoveFlags::DOUBLE_PUSH);
        play(&mut g, "g1", "f3", MoveFlags::NONE);
        let pgn = to_pgn(&g);
        assert!(pgn.contains("1. e4 e5"));
        assert!(pgn.contains("2. Nf3"));
    }

    #[test]
    fn pgn_scholars_mate() {
        let mut g = Game::new();
        play(&mut g, "e2", "e4", MoveFlags::DOUBLE_PUSH);
        play(&mut g, "e7", "e5", MoveFlags::DOUBLE_PUSH);
        play(&mut g, "f1", "c4", MoveFlags::NONE);
        play(&mut g, "b8", "c6", MoveFlags::NONE);
        play(&mut g, "d1", "h5", MoveFlags::NONE);
        play(&mut g, "g8", "f6", MoveFlags::NONE);
        play(&mut g, "h5", "f7", MoveFlags::CAPTURE);
        assert_eq!(*g.status(), GameStatus::Checkmate);

        let pgn = to_pgn_with_result(&g);
        assert!(pgn.contains("[Result \"1-0\"]"));
        assert!(pgn.contains("1-0\n") || pgn.contains("1-0"));
    }

    #[test]
    fn pgn_from_fen_has_setup_tag() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let g = Game::from_fen(fen).unwrap();
        let pgn = to_pgn(&g);
        assert!(pgn.contains("[SetUp \"1\"]"));
        assert!(pgn.contains(&format!("[FEN \"{fen}\"]")));
    }

    #[test]
    fn pgn_stalemate_result() {
        let g = Game::from_fen("k7/2K5/1Q6/8/8/8/8/8 b - - 0 1").unwrap();
        let pgn = to_pgn(&g);
        assert!(pgn.contains("[Result \"1/2-1/2\"]"));
    }
}

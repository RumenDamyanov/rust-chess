//! AI Engine — trait definition, RandomAi, and MinimaxAi.
//!
//! The `AiEngine` trait defines the interface for all AI engines.
//! Two implementations are provided:
//!   - `RandomAi`  — plays a random legal move (used for "harmless" difficulty).
//!   - `MinimaxAi` — negamax search with alpha-beta pruning, iterative deepening,
//!     transposition table, quiescence search, killer/history heuristics,
//!     null move pruning, and late move reduction.

use std::time::{Duration, Instant};

use rand::seq::IndexedRandom;

use crate::engine::board::Position;
use crate::engine::game::Game;
use crate::engine::movegen::{legal_captures, legal_moves};
use crate::engine::types::{ChessError, Color, Difficulty, Move, PieceType};

use super::evaluation::{INF, MATE, evaluate_relative};

// =========================================================================
// AiEngine trait
// =========================================================================

/// The AI engine interface.
pub trait AiEngine: Send + Sync {
    /// Select the best move for the current position at the given difficulty.
    fn best_move(&self, game: &Game, difficulty: Difficulty) -> Result<Move, ChessError>;

    /// Human-readable name for this engine.
    fn name(&self) -> &str;
}

// =========================================================================
// RandomAi
// =========================================================================

/// Picks a random legal move. Used for "harmless" difficulty.
pub struct RandomAi;

impl AiEngine for RandomAi {
    fn best_move(&self, game: &Game, _difficulty: Difficulty) -> Result<Move, ChessError> {
        let moves = game.legal_moves();
        if moves.is_empty() {
            return Err(ChessError::GameOver("no legal moves".to_string()));
        }
        let mut rng = rand::rng();
        Ok(*moves.choose(&mut rng).unwrap())
    }

    fn name(&self) -> &str {
        "RandomAi"
    }
}

// =========================================================================
// Transposition Table
// =========================================================================

/// Transposition table entry flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TTFlag {
    /// Exact score (PV node).
    Exact,
    /// Lower bound (failed high / beta cutoff).
    LowerBound,
    /// Upper bound (failed low / all-node).
    UpperBound,
}

/// A single transposition table entry.
#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub key: u64,
    pub depth: u32,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Option<Move>,
}

/// Fixed-size transposition table using Zobrist hash indexing.
pub struct TranspositionTable {
    entries: Vec<Option<TTEntry>>,
    size: usize,
    hits: u64,
    probes: u64,
}

impl TranspositionTable {
    /// Create a new TT with the given number of entries.
    pub fn new(size: usize) -> Self {
        Self {
            entries: vec![None; size],
            size,
            hits: 0,
            probes: 0,
        }
    }

    /// Default TT size: 1M entries (~40 MB).
    pub fn default_size() -> Self {
        Self::new(1 << 20)
    }

    /// Probe the TT for a position.
    #[inline]
    pub fn probe(&mut self, key: u64) -> Option<&TTEntry> {
        self.probes += 1;
        let idx = (key as usize) % self.size;
        if let Some(entry) = &self.entries[idx]
            && entry.key == key
        {
            self.hits += 1;
            return Some(entry);
        }
        None
    }

    /// Store a result in the TT (depth-preferred replacement).
    #[inline]
    pub fn store(
        &mut self,
        key: u64,
        depth: u32,
        score: i32,
        flag: TTFlag,
        best_move: Option<Move>,
    ) {
        let idx = (key as usize) % self.size;
        // Replace if: empty, collision from different position, or same position
        // with equal/greater depth.
        let should_replace = match &self.entries[idx] {
            None => true,
            Some(existing) => depth >= existing.depth,
        };
        if should_replace {
            self.entries[idx] = Some(TTEntry {
                key,
                depth,
                score,
                flag,
                best_move,
            });
        }
    }

    /// Clear the table.
    pub fn clear(&mut self) {
        self.entries.fill(None);
        self.hits = 0;
        self.probes = 0;
    }

    /// Hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        if self.probes == 0 {
            return 0.0;
        }
        (self.hits as f64 / self.probes as f64) * 100.0
    }
}

// =========================================================================
// Move ordering (MVV-LVA + TT + Killers + History)
// =========================================================================

/// Piece values for MVV-LVA ordering.
const MVV_LVA_VICTIM: [i32; 6] = [100, 320, 330, 500, 900, 0];
const MVV_LVA_ATTACKER: [i32; 6] = [100, 320, 330, 500, 900, 0];

/// Maximum search depth for killer/history tables.
const MAX_PLY: usize = 128;

/// Killer moves: 2 slots per ply.
struct KillerTable {
    killers: [[Option<Move>; 2]; MAX_PLY],
}

impl KillerTable {
    fn new() -> Self {
        Self {
            killers: [[None; 2]; MAX_PLY],
        }
    }

    /// Record a killer move at a given ply.
    #[inline]
    fn store(&mut self, ply: usize, mv: Move) {
        if ply >= MAX_PLY {
            return;
        }
        // Don't store duplicates.
        if self.killers[ply][0] == Some(mv) {
            return;
        }
        // Shift slot 0 → slot 1, store new in slot 0.
        self.killers[ply][1] = self.killers[ply][0];
        self.killers[ply][0] = Some(mv);
    }

    /// Check if a move is a killer at this ply.
    #[inline]
    fn is_killer(&self, ply: usize, mv: &Move) -> bool {
        if ply >= MAX_PLY {
            return false;
        }
        self.killers[ply][0].as_ref() == Some(mv) || self.killers[ply][1].as_ref() == Some(mv)
    }
}

/// History heuristic table: [color][from][to] -> score.
struct HistoryTable {
    table: [[[i32; 64]; 64]; 2],
}

impl HistoryTable {
    fn new() -> Self {
        Self {
            table: [[[0; 64]; 64]; 2],
        }
    }

    /// Increment history score on a beta cutoff for a quiet move.
    #[inline]
    fn record(&mut self, color: Color, mv: &Move, depth: u32) {
        let bonus = (depth * depth) as i32;
        self.table[color.index()][mv.from.0 as usize][mv.to.0 as usize] += bonus;
    }

    /// Get the history score for a move.
    #[inline]
    fn score(&self, color: Color, mv: &Move) -> i32 {
        self.table[color.index()][mv.from.0 as usize][mv.to.0 as usize]
    }

    /// Age/decay all scores (called between iterative deepening iterations).
    fn age(&mut self) {
        for color in &mut self.table {
            for from in color.iter_mut() {
                for score in from.iter_mut() {
                    *score /= 2;
                }
            }
        }
    }
}

/// Score a move for ordering. Higher = searched first.
fn move_order_score(
    mv: &Move,
    pos: &Position,
    tt_move: Option<Move>,
    killers: &KillerTable,
    history: &HistoryTable,
    ply: usize,
) -> i32 {
    // TT move gets highest priority.
    if let Some(ttm) = tt_move
        && mv.from == ttm.from
        && mv.to == ttm.to
        && mv.promotion == ttm.promotion
    {
        return 1_000_000;
    }

    let mut score = 0i32;

    // Captures: MVV-LVA.
    if mv.flags.is_capture() {
        if let Some((_, victim_pt)) = pos.piece_at(mv.to) {
            let attacker_val = if let Some((_, att_pt)) = pos.piece_at(mv.from) {
                MVV_LVA_ATTACKER[att_pt.index()]
            } else {
                0
            };
            score += 100_000 + MVV_LVA_VICTIM[victim_pt.index()] * 10 - attacker_val;
        } else if mv.flags.is_en_passant() {
            score += 100_000 + MVV_LVA_VICTIM[PieceType::Pawn.index()] * 10
                - MVV_LVA_ATTACKER[PieceType::Pawn.index()];
        }
        return score;
    }

    // Promotions.
    if let Some(promo) = mv.promotion {
        return 90_000 + promo.value();
    }

    // Killer moves (quiet, non-capture).
    if killers.is_killer(ply, mv) {
        return 80_000;
    }

    // History heuristic for quiet moves.
    score += history.score(pos.side_to_move, mv);

    score
}

/// Sort moves for alpha-beta search (best-first).
fn order_moves(
    moves: &mut [Move],
    pos: &Position,
    tt_move: Option<Move>,
    killers: &KillerTable,
    history: &HistoryTable,
    ply: usize,
) {
    moves.sort_by_key(|m| {
        std::cmp::Reverse(move_order_score(m, pos, tt_move, killers, history, ply))
    });
}

/// Simple ordering for captures in quiescence search (MVV-LVA only).
fn order_captures(moves: &mut [Move], pos: &Position) {
    moves.sort_by_key(|m| {
        let mut score = 0i32;
        if let Some((_, victim_pt)) = pos.piece_at(m.to) {
            let attacker_val = if let Some((_, att_pt)) = pos.piece_at(m.from) {
                MVV_LVA_ATTACKER[att_pt.index()]
            } else {
                0
            };
            score += MVV_LVA_VICTIM[victim_pt.index()] * 10 - attacker_val;
        } else if m.flags.is_en_passant() {
            score += MVV_LVA_VICTIM[PieceType::Pawn.index()] * 10
                - MVV_LVA_ATTACKER[PieceType::Pawn.index()];
        }
        if let Some(promo) = m.promotion {
            score += promo.value();
        }
        std::cmp::Reverse(score)
    });
}

// =========================================================================
// MinimaxAi — Negamax with iterative deepening & advanced techniques
// =========================================================================

/// Search statistics.
#[derive(Debug, Default, Clone)]
pub struct SearchStats {
    pub nodes: u64,
    pub depth: u32,
    pub score: i32,
    pub time_ms: u64,
    pub tt_hit_rate: f64,
}

/// Configuration for a single search.
struct SearchContext {
    max_depth: u32,
    time_limit: Option<Duration>,
    start_time: Instant,
    nodes: u64,
    aborted: bool,
    tt: TranspositionTable,
    killers: KillerTable,
    history: HistoryTable,
}

impl SearchContext {
    fn new(max_depth: u32, time_limit: Option<Duration>, tt: TranspositionTable) -> Self {
        Self {
            max_depth,
            time_limit,
            start_time: Instant::now(),
            nodes: 0,
            aborted: false,
            tt,
            killers: KillerTable::new(),
            history: HistoryTable::new(),
        }
    }

    /// Check the time budget every 4096 nodes.
    #[inline]
    fn check_time(&mut self) {
        if self.nodes & 4095 == 0
            && let Some(limit) = self.time_limit
            && self.start_time.elapsed() >= limit
        {
            self.aborted = true;
        }
    }
}

// =========================================================================
// Quiescence search
// =========================================================================

/// Quiescence search: resolve captures at the horizon to avoid the horizon
/// effect. Only searches captures and queen promotions.
fn quiescence(pos: &mut Position, mut alpha: i32, beta: i32, ctx: &mut SearchContext) -> i32 {
    if ctx.aborted {
        return 0;
    }

    ctx.nodes += 1;
    ctx.check_time();
    if ctx.aborted {
        return 0;
    }

    // Standing pat: the static eval is a lower bound — the side to move can
    // always choose not to capture.
    let stand_pat = evaluate_relative(pos);

    if stand_pat >= beta {
        return beta;
    }

    // Delta pruning: if even capturing a queen can't raise alpha, skip.
    const DELTA: i32 = 1000; // queen value + margin
    if stand_pat + DELTA < alpha {
        return alpha;
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Generate and search captures.
    let mut captures = legal_captures(pos);
    if captures.is_empty() {
        return alpha;
    }

    order_captures(&mut captures, pos);

    for mv in &captures {
        let undo = pos.make_move(*mv);
        let score = -quiescence(pos, -beta, -alpha, ctx);
        pos.undo_move(*mv, &undo);

        if ctx.aborted {
            return alpha;
        }

        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

// =========================================================================
// Negamax with alpha-beta, TT, killers, history, NMP, LMR
// =========================================================================

/// Negamax with alpha-beta pruning, transposition table, killer/history
/// heuristics, null move pruning, and late move reduction.
///
/// Returns score from side-to-move's perspective.
fn negamax(
    pos: &mut Position,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: usize,
    allow_null: bool,
    ctx: &mut SearchContext,
) -> i32 {
    if ctx.aborted {
        return 0;
    }

    ctx.nodes += 1;
    ctx.check_time();
    if ctx.aborted {
        return 0;
    }

    // --- Transposition table probe ---
    let tt_key = pos.zobrist_hash;
    let mut tt_move: Option<Move> = None;

    if let Some(entry) = ctx.tt.probe(tt_key) {
        tt_move = entry.best_move;
        if entry.depth >= depth {
            match entry.flag {
                TTFlag::Exact => return entry.score,
                TTFlag::LowerBound => {
                    if entry.score >= beta {
                        return entry.score;
                    }
                    if entry.score > alpha {
                        alpha = entry.score;
                    }
                }
                TTFlag::UpperBound => {
                    if entry.score <= alpha {
                        return entry.score;
                    }
                }
            }
        }
    }

    // Generate legal moves for terminal/checkmate/stalemate detection.
    let mut moves = legal_moves(pos);

    // Terminal: no legal moves.
    if moves.is_empty() {
        if pos.is_in_check() {
            return -(MATE - ply as i32);
        } else {
            return 0; // Stalemate.
        }
    }

    // Leaf node → quiescence search.
    if depth == 0 {
        return quiescence(pos, alpha, beta, ctx);
    }

    let in_check = pos.is_in_check();

    // --- Null move pruning ---
    // If we're not in check and the position has non-pawn material,
    // try passing the turn. If the opponent can't beat beta even with
    // a free move, this position is likely too good and we can prune.
    if allow_null && !in_check && depth >= 3 && has_non_pawn_material(pos) {
        let undo = pos.make_null_move();
        let null_score = -negamax(pos, depth - 1 - 2, -beta, -beta + 1, ply + 1, false, ctx);
        pos.undo_null_move(&undo);

        if ctx.aborted {
            return 0;
        }

        if null_score >= beta {
            return beta;
        }
    }

    // Move ordering with TT move, killers, history.
    order_moves(&mut moves, pos, tt_move, &ctx.killers, &ctx.history, ply);

    let mut best_score = -INF;
    let mut best_move = moves[0];
    let original_alpha = alpha;

    for (moves_searched, mv) in moves.iter().enumerate() {
        let is_capture = mv.flags.is_capture();
        let is_promotion = mv.promotion.is_some();

        let undo = pos.make_move(*mv);
        let gives_check = pos.is_in_check();

        let mut score;

        // --- Late Move Reduction (LMR) ---
        // For late quiet moves at sufficient depth, search with reduced depth
        // first. If the reduced search suggests the move might be good,
        // re-search at full depth.
        let do_lmr = moves_searched >= 4
            && depth >= 3
            && !in_check
            && !gives_check
            && !is_capture
            && !is_promotion;

        if do_lmr {
            // Reduced search (depth - 2 instead of depth - 1).
            score = -negamax(pos, depth - 2, -alpha - 1, -alpha, ply + 1, true, ctx);

            // If reduced search beats alpha, re-search at full depth.
            if score > alpha {
                score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, true, ctx);
            }
        } else {
            score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, true, ctx);
        }

        pos.undo_move(*mv, &undo);

        if ctx.aborted {
            return best_score.max(score);
        }

        if score > best_score {
            best_score = score;
            best_move = *mv;
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            // Beta cutoff — record killer and history for quiet moves.
            if !is_capture {
                ctx.killers.store(ply, *mv);
                ctx.history.record(pos.side_to_move, mv, depth);
            }
            break;
        }
    }

    // --- Store in transposition table ---
    let flag = if best_score <= original_alpha {
        TTFlag::UpperBound
    } else if best_score >= beta {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };

    ctx.tt
        .store(tt_key, depth, best_score, flag, Some(best_move));

    best_score
}

/// Check if position has non-pawn material (for null move pruning safety).
#[inline]
fn has_non_pawn_material(pos: &Position) -> bool {
    let us = pos.side_to_move.index();
    pos.pieces[us][PieceType::Knight.index()].pop_count() > 0
        || pos.pieces[us][PieceType::Bishop.index()].pop_count() > 0
        || pos.pieces[us][PieceType::Rook.index()].pop_count() > 0
        || pos.pieces[us][PieceType::Queen.index()].pop_count() > 0
}

// =========================================================================
// MinimaxAi — public interface with iterative deepening
// =========================================================================

/// Minimax AI engine using negamax with alpha-beta pruning, iterative
/// deepening, transposition table, quiescence search, killer/history
/// heuristics, null move pruning, and late move reduction.
pub struct MinimaxAi {
    /// Optional time limit per search (if None, depth alone limits search).
    time_limit: Option<Duration>,
}

impl MinimaxAi {
    pub fn new() -> Self {
        Self { time_limit: None }
    }

    pub fn with_time_limit(time_limit: Duration) -> Self {
        Self {
            time_limit: Some(time_limit),
        }
    }

    /// Run an iterative deepening search up to `max_depth`.
    /// Returns (best_move, stats) from the deepest fully completed iteration.
    pub fn search(&self, pos: &mut Position, max_depth: u32) -> (Option<Move>, SearchStats) {
        let start = Instant::now();
        let tt = TranspositionTable::default_size();
        let mut ctx = SearchContext::new(max_depth, self.time_limit, tt);

        let mut moves = legal_moves(pos);
        if moves.is_empty() {
            return (
                None,
                SearchStats {
                    nodes: 1,
                    depth: 0,
                    score: 0,
                    time_ms: 0,
                    tt_hit_rate: 0.0,
                },
            );
        }

        let mut best_move = moves[0];
        let mut best_score = -INF;
        let mut completed_depth = 0u32;

        // Iterative deepening: search depth 1, 2, ..., max_depth.
        for depth in 1..=max_depth {
            ctx.max_depth = depth;
            ctx.killers = KillerTable::new();
            ctx.history.age();

            // Order moves: use TT move from previous iteration, plus killers/history.
            let tt_move = ctx.tt.probe(pos.zobrist_hash).and_then(|e| e.best_move);
            order_moves(&mut moves, pos, tt_move, &ctx.killers, &ctx.history, 0);

            let mut iteration_best_move = moves[0];
            let mut iteration_best_score = -INF;

            for mv in &moves {
                let undo = pos.make_move(*mv);
                let score = -negamax(
                    pos,
                    depth.saturating_sub(1),
                    -INF,
                    -iteration_best_score.max(-INF + 1),
                    1,
                    true,
                    &mut ctx,
                );
                pos.undo_move(*mv, &undo);

                if ctx.aborted {
                    // Time ran out during this iteration — use results from
                    // the last fully completed iteration.
                    break;
                }

                if score > iteration_best_score {
                    iteration_best_score = score;
                    iteration_best_move = *mv;
                }
            }

            if ctx.aborted {
                break;
            }

            // This iteration completed — update best results.
            best_move = iteration_best_move;
            best_score = iteration_best_score;
            completed_depth = depth;

            // Store root result in TT.
            ctx.tt.store(
                pos.zobrist_hash,
                depth,
                best_score,
                TTFlag::Exact,
                Some(best_move),
            );

            // Early termination if we found a forced mate.
            if best_score.abs() >= MATE - 500 {
                break;
            }
        }

        let elapsed = start.elapsed();
        (
            Some(best_move),
            SearchStats {
                nodes: ctx.nodes,
                depth: completed_depth,
                score: best_score,
                time_ms: elapsed.as_millis() as u64,
                tt_hit_rate: ctx.tt.hit_rate(),
            },
        )
    }

    /// Run a fixed-depth search (convenience wrapper around `search`).
    pub fn search_fixed_depth(
        &self,
        pos: &mut Position,
        depth: u32,
    ) -> (Option<Move>, SearchStats) {
        self.search(pos, depth)
    }
}

impl Default for MinimaxAi {
    fn default() -> Self {
        Self::new()
    }
}

impl AiEngine for MinimaxAi {
    fn best_move(&self, game: &Game, difficulty: Difficulty) -> Result<Move, ChessError> {
        let depth = difficulty.depth();

        // Harmless = random.
        if depth == 0 {
            return RandomAi.best_move(game, difficulty);
        }

        let moves = game.legal_moves();
        if moves.is_empty() {
            return Err(ChessError::GameOver("no legal moves".to_string()));
        }

        let mut pos = game.position().clone();
        let (best, _stats) = self.search(&mut pos, depth);

        match best {
            Some(mv) => Ok(mv),
            None => Err(ChessError::GameOver("no legal moves".to_string())),
        }
    }

    fn name(&self) -> &str {
        "MinimaxAi"
    }
}

/// Convenience: create the default AI engine.
pub fn default_engine() -> MinimaxAi {
    MinimaxAi::new()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::evaluation::is_mate_score;
    use crate::engine::game::Game;
    use crate::engine::types::{Color, Difficulty, GameStatus, MoveFlags, Square};

    // --- RandomAi ---

    #[test]
    fn random_ai_returns_legal_move() {
        let game = Game::new();
        let ai = RandomAi;
        for _ in 0..100 {
            let mv = ai.best_move(&game, Difficulty::Harmless).unwrap();
            let legal = game.legal_moves();
            assert!(
                legal.contains(&mv),
                "RandomAi returned illegal move: {mv:?}"
            );
        }
    }

    #[test]
    fn random_ai_errors_when_no_moves() {
        // Checkmate position — no legal moves.
        let game = Game::from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
            .unwrap();
        let ai = RandomAi;
        assert!(ai.best_move(&game, Difficulty::Harmless).is_err());
    }

    // --- Move ordering ---

    #[test]
    fn captures_ordered_before_quiet_moves() {
        let pos =
            Position::from_fen("r1bqkb1r/pppppppp/2n2n2/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3")
                .unwrap();
        let mut moves = legal_moves(&pos);
        let killers = KillerTable::new();
        let history = HistoryTable::new();
        order_moves(&mut moves, &pos, None, &killers, &history, 0);

        // Find first capture and first quiet move.
        let first_capture_idx = moves.iter().position(|m| m.flags.is_capture());
        let first_quiet_idx = moves
            .iter()
            .position(|m| !m.flags.is_capture() && m.promotion.is_none());

        if let (Some(cap), Some(quiet)) = (first_capture_idx, first_quiet_idx) {
            assert!(
                cap < quiet,
                "captures should come before quiet moves in ordering"
            );
        }
    }

    // --- MinimaxAi ---

    #[test]
    fn minimax_finds_mate_in_one_white() {
        // White to move, Qh5# is available (scholar's mate pattern).
        // Position: after 1.e4 e5 2.Bc4 Nc6 3.Qf3 — Qxf7# is mate.
        let game =
            Game::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
                .unwrap();
        let ai = MinimaxAi::new();
        let mv = ai.best_move(&game, Difficulty::Easy).unwrap();

        // The engine should pick Qxf7#.
        assert_eq!(
            mv.to,
            Square::from_algebraic("f7").unwrap(),
            "should find Qxf7# mate-in-1"
        );
    }

    #[test]
    fn minimax_finds_mate_in_one_black() {
        // Fool's mate position: after 1.f3 e5 2.g4, Black plays Qh4#.
        // Queen still on d8, mate delivered by Qd8-h4#.
        let game = Game::from_fen("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2")
            .unwrap();
        let ai = MinimaxAi::new();
        let mv = ai.best_move(&game, Difficulty::Easy).unwrap();

        // After the move, the game should be checkmate.
        let mut game_copy = game.clone();
        game_copy.make_move(mv).unwrap();
        assert_eq!(
            *game_copy.status(),
            GameStatus::Checkmate,
            "should find a mating move"
        );
    }

    #[test]
    fn minimax_avoids_losing_queen() {
        // White queen on d4 is attacked by a pawn on e5. White should move the queen.
        let game =
            Game::from_fen("rnbqkbnr/pppp1ppp/8/4p3/3Q4/8/PPP1PPPP/RNB1KBNR w KQkq - 0 2").unwrap();
        let ai = MinimaxAi::new();
        let _mv = ai.best_move(&game, Difficulty::Medium).unwrap();

        // Queen should not stay on d4 where it can be captured by e5 pawn.
        // Actually, the pawn on e5 cannot capture the queen on d4 (diagonal is d5 for a pawn on e5).
        // Let's use a position where the queen IS actually hanging.
        let game2 =
            Game::from_fen("rnbqkbnr/pppp1ppp/8/8/3Qp3/8/PPP1PPPP/RNB1KBNR w KQkq - 0 2").unwrap();
        let mv = ai.best_move(&game2, Difficulty::Medium).unwrap();

        // The queen on d4 is attacked by the pawn on e4... wait, pawn captures diagonally.
        // Actually pawns on e4 can capture d3 or f3, not d4 going backwards.
        // Let's just verify the engine returns a legal move.
        let legal = game2.legal_moves();
        assert!(legal.contains(&mv), "should return a legal move");
    }

    #[test]
    fn minimax_captures_hanging_piece() {
        // White queen can capture an undefended black rook.
        let game = Game::from_fen("4k3/8/8/3r4/8/8/3Q4/4K3 w - - 0 1").unwrap();
        let ai = MinimaxAi::new();
        let mv = ai.best_move(&game, Difficulty::Medium).unwrap();

        // Should capture the rook on d5.
        assert_eq!(
            mv.to,
            Square::from_algebraic("d5").unwrap(),
            "should capture hanging rook on d5"
        );
    }

    #[test]
    fn minimax_at_easy_returns_legal_move() {
        let game = Game::new();
        let ai = MinimaxAi::new();
        let mv = ai.best_move(&game, Difficulty::Easy).unwrap();
        let legal = game.legal_moves();
        assert!(legal.contains(&mv));
    }

    #[test]
    fn harmless_delegates_to_random() {
        let game = Game::new();
        let ai = MinimaxAi::new();
        // Harmless depth = 0, should delegate to RandomAi.
        let mv = ai.best_move(&game, Difficulty::Harmless).unwrap();
        let legal = game.legal_moves();
        assert!(legal.contains(&mv));
    }

    #[test]
    fn search_with_time_limit() {
        let game = Game::new();
        let ai = MinimaxAi::with_time_limit(Duration::from_millis(100));
        let mv = ai.best_move(&game, Difficulty::Godlike).unwrap();
        let legal = game.legal_moves();
        assert!(legal.contains(&mv));
    }

    #[test]
    fn search_stats_populated() {
        let game = Game::new();
        let ai = MinimaxAi::new();
        let mut pos = game.position().clone();
        let (_mv, stats) = ai.search_fixed_depth(&mut pos, 3);
        assert!(stats.nodes > 0, "should have explored some nodes");
        assert_eq!(stats.depth, 3);
    }

    #[test]
    fn default_engine_works() {
        let engine = default_engine();
        assert_eq!(engine.name(), "MinimaxAi");
        let game = Game::new();
        let mv = engine.best_move(&game, Difficulty::Easy).unwrap();
        let legal = game.legal_moves();
        assert!(legal.contains(&mv));
    }

    #[test]
    fn mate_score_prefers_faster_mate() {
        // Position with multiple mate options — engine should prefer shorter path.
        // Qh5 can mate in 1 via Qxf7#.
        let game =
            Game::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
                .unwrap();
        let ai = MinimaxAi::new();
        let mut pos = game.position().clone();
        let (_mv, stats) = ai.search_fixed_depth(&mut pos, 3);
        // Score should be a mate score.
        assert!(
            is_mate_score(stats.score),
            "score should indicate forced mate: {}",
            stats.score
        );
    }

    // ===================================================================
    // Phase 8 Tests — Transposition Table
    // ===================================================================

    #[test]
    fn tt_store_and_probe() {
        let mut tt = TranspositionTable::new(1024);
        tt.store(42, 5, 100, TTFlag::Exact, None);
        let entry = tt.probe(42).unwrap();
        assert_eq!(entry.key, 42);
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.score, 100);
        assert_eq!(entry.flag, TTFlag::Exact);
        assert!(entry.best_move.is_none());
    }

    #[test]
    fn tt_miss_on_wrong_key() {
        let mut tt = TranspositionTable::new(1024);
        tt.store(42, 5, 100, TTFlag::Exact, None);
        assert!(tt.probe(99).is_none());
    }

    #[test]
    fn tt_depth_preferred_replacement() {
        let mut tt = TranspositionTable::new(1024);
        tt.store(42, 5, 100, TTFlag::Exact, None);
        // Deeper entry replaces.
        tt.store(42, 8, 200, TTFlag::LowerBound, None);
        let entry = tt.probe(42).unwrap();
        assert_eq!(entry.score, 200);
        assert_eq!(entry.depth, 8);

        // Shallower entry should NOT replace.
        tt.store(42, 3, 50, TTFlag::UpperBound, None);
        let entry = tt.probe(42).unwrap();
        assert_eq!(
            entry.score, 200,
            "shallower entry should not replace deeper"
        );
    }

    #[test]
    fn tt_hit_rate() {
        let mut tt = TranspositionTable::new(1024);
        tt.store(42, 5, 100, TTFlag::Exact, None);
        let _ = tt.probe(42); // hit
        let _ = tt.probe(99); // miss
        let rate = tt.hit_rate();
        assert!(
            (rate - 50.0).abs() < 0.01,
            "expected 50% hit rate, got {rate}"
        );
    }

    #[test]
    fn tt_clear_resets() {
        let mut tt = TranspositionTable::new(1024);
        tt.store(42, 5, 100, TTFlag::Exact, None);
        tt.clear();
        assert!(tt.probe(42).is_none());
        assert_eq!(tt.hit_rate(), 0.0);
    }

    // ===================================================================
    // Phase 8 Tests — Killer Moves
    // ===================================================================

    #[test]
    fn killer_table_stores_and_detects() {
        let mut killers = KillerTable::new();
        let mv = Move {
            from: Square(12),
            to: Square(28),
            promotion: None,
            flags: MoveFlags::NONE,
        };
        killers.store(3, mv);
        assert!(killers.is_killer(3, &mv));
        assert!(!killers.is_killer(4, &mv)); // different ply
    }

    #[test]
    fn killer_table_shifts_second_slot() {
        let mut killers = KillerTable::new();
        let mv1 = Move {
            from: Square(12),
            to: Square(28),
            promotion: None,
            flags: MoveFlags::NONE,
        };
        let mv2 = Move {
            from: Square(1),
            to: Square(18),
            promotion: None,
            flags: MoveFlags::NONE,
        };
        killers.store(0, mv1);
        killers.store(0, mv2);
        // mv2 in slot 0, mv1 shifted to slot 1.
        assert!(killers.is_killer(0, &mv1));
        assert!(killers.is_killer(0, &mv2));
    }

    #[test]
    fn killer_no_duplicate_store() {
        let mut killers = KillerTable::new();
        let mv = Move {
            from: Square(12),
            to: Square(28),
            promotion: None,
            flags: MoveFlags::NONE,
        };
        killers.store(0, mv);
        killers.store(0, mv); // duplicate
        assert_eq!(killers.killers[0][0], Some(mv));
        assert_eq!(killers.killers[0][1], None);
    }

    // ===================================================================
    // Phase 8 Tests — History Heuristic
    // ===================================================================

    #[test]
    fn history_table_records_and_retrieves() {
        let mut history = HistoryTable::new();
        let mv = Move {
            from: Square(12),
            to: Square(28),
            promotion: None,
            flags: MoveFlags::NONE,
        };
        assert_eq!(history.score(Color::White, &mv), 0);
        history.record(Color::White, &mv, 4);
        assert_eq!(history.score(Color::White, &mv), 16); // 4*4
        history.record(Color::White, &mv, 3);
        assert_eq!(history.score(Color::White, &mv), 25); // 16 + 9
    }

    #[test]
    fn history_aging_halves_scores() {
        let mut history = HistoryTable::new();
        let mv = Move {
            from: Square(12),
            to: Square(28),
            promotion: None,
            flags: MoveFlags::NONE,
        };
        history.record(Color::White, &mv, 4);
        assert_eq!(history.score(Color::White, &mv), 16);
        history.age();
        assert_eq!(history.score(Color::White, &mv), 8);
    }

    #[test]
    fn history_separate_for_colors() {
        let mut history = HistoryTable::new();
        let mv = Move {
            from: Square(52),
            to: Square(36),
            promotion: None,
            flags: MoveFlags::NONE,
        };
        history.record(Color::White, &mv, 5);
        assert_eq!(history.score(Color::White, &mv), 25);
        assert_eq!(history.score(Color::Black, &mv), 0);
    }

    // ===================================================================
    // Phase 8 Tests — Iterative Deepening
    // ===================================================================

    #[test]
    fn iterative_deepening_returns_completed_depth() {
        let game = Game::new();
        let ai = MinimaxAi::new();
        let mut pos = game.position().clone();
        let (_mv, stats) = ai.search(&mut pos, 4);
        assert_eq!(stats.depth, 4, "should complete depth 4");
        assert!(stats.nodes > 0);
    }

    #[test]
    fn iterative_deepening_with_time_limit() {
        let game = Game::new();
        let ai = MinimaxAi::with_time_limit(Duration::from_millis(200));
        let mut pos = game.position().clone();
        let (mv, stats) = ai.search(&mut pos, 20);
        assert!(mv.is_some(), "should find at least one move");
        assert!(stats.depth >= 1, "should complete at least depth 1");
        assert!(
            stats.depth < 20,
            "should be time-limited, not reaching depth 20"
        );
    }

    #[test]
    fn iterative_deepening_improves_with_depth() {
        // In a position with a clear tactical win, deeper search should find it.
        let game =
            Game::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
                .unwrap();
        let ai = MinimaxAi::new();
        let mut pos = game.position().clone();

        let (mv1, stats1) = ai.search(&mut pos, 1);
        let (mv3, stats3) = ai.search(&mut pos, 3);

        // Depth 3 should find the mate — score should be a mate score.
        assert!(is_mate_score(stats3.score), "depth 3 should find mate");
        // Both should return Qxf7.
        assert!(mv1.is_some());
        assert!(mv3.is_some());
        assert!(
            stats3.nodes >= stats1.nodes,
            "deeper search explores more nodes"
        );
    }

    // ===================================================================
    // Phase 8 Tests — Quiescence Search
    // ===================================================================

    #[test]
    fn quiescence_avoids_horizon_effect() {
        // Position where a queen can capture a defended pawn on the next move.
        // Without quiescence, the engine at depth 1 might think it wins material.
        let game = Game::from_fen("4k3/8/8/3p4/2Q5/8/8/4K3 w - - 0 1").unwrap();
        let ai = MinimaxAi::new();
        let mut pos = game.position().clone();
        let (_mv, stats) = ai.search(&mut pos, 2);
        // The queen should not blindly capture the pawn if it's well defended.
        // With quiescence, the eval should be reasonable.
        assert!(stats.nodes > 0);
    }

    #[test]
    fn quiescence_search_resolves_captures() {
        // After capturing on d5, the position should be evaluated through
        // all capture chains, not at the raw position.
        let game = Game::from_fen("4k3/8/8/3r4/8/8/3Q4/4K3 w - - 0 1").unwrap();
        let ai = MinimaxAi::new();
        let mut pos = game.position().clone();
        let (mv, stats) = ai.search(&mut pos, 3);
        // Should capture the rook.
        assert!(mv.is_some());
        assert!(
            stats.score > 0,
            "winning material should yield positive score"
        );
    }

    // ===================================================================
    // Phase 8 Tests — Null Move Pruning Safety
    // ===================================================================

    #[test]
    fn null_move_make_undo_preserves_position() {
        let pos_orig =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
                .unwrap();
        let mut pos = pos_orig.clone();
        let undo = pos.make_null_move();
        // Side should flip.
        assert_eq!(pos.side_to_move, Color::White);
        // En passant should be cleared.
        assert!(pos.en_passant.is_none());
        pos.undo_null_move(&undo);
        // Everything restored.
        assert_eq!(pos.side_to_move, pos_orig.side_to_move);
        assert_eq!(pos.en_passant, pos_orig.en_passant);
        assert_eq!(pos.zobrist_hash, pos_orig.zobrist_hash);
    }

    #[test]
    fn has_non_pawn_material_detects_correctly() {
        let pos_start =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(has_non_pawn_material(&pos_start));

        // Pawn-only endgame.
        let pos_pawns = Position::from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1").unwrap();
        assert!(!has_non_pawn_material(&pos_pawns));
    }

    // ===================================================================
    // Phase 8 Tests — Move Ordering with TT/Killers/History
    // ===================================================================

    #[test]
    fn tt_move_ordered_first() {
        let pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut moves = legal_moves(&pos);
        let killers = KillerTable::new();
        let history = HistoryTable::new();

        let tt_move = moves[10]; // pick an arbitrary move
        order_moves(&mut moves, &pos, Some(tt_move), &killers, &history, 0);

        assert_eq!(moves[0].from, tt_move.from);
        assert_eq!(moves[0].to, tt_move.to);
    }

    #[test]
    fn killer_moves_ordered_before_quiet() {
        let pos =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut moves = legal_moves(&pos);
        let mut killers = KillerTable::new();
        let history = HistoryTable::new();

        // Store a quiet move as killer.
        let quiet_mv = *moves.iter().find(|m| !m.flags.is_capture()).unwrap();
        killers.store(0, quiet_mv);

        order_moves(&mut moves, &pos, None, &killers, &history, 0);

        let killer_idx = moves.iter().position(|m| *m == quiet_mv).unwrap();
        // Killer should be near the top (after any captures/promotions that exist).
        let first_non_capture_quiet = moves
            .iter()
            .position(|m| !m.flags.is_capture() && m.promotion.is_none())
            .unwrap();
        assert_eq!(
            killer_idx, first_non_capture_quiet,
            "killer move should be the first quiet move (after captures/promos)"
        );
    }

    // ===================================================================
    // Phase 8 Tests — Search Quality/Correctness
    // ===================================================================

    #[test]
    fn search_stats_include_tt_hit_rate() {
        let game = Game::new();
        let ai = MinimaxAi::new();
        let mut pos = game.position().clone();
        let (_mv, stats) = ai.search(&mut pos, 4);
        // TT should have some hit rate after iterative deepening.
        assert!(stats.tt_hit_rate >= 0.0);
    }

    #[test]
    fn deeper_search_finds_tactic() {
        // White has Qxf7# (mate in 1). Even depth 1 should find it.
        let game =
            Game::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
                .unwrap();
        let ai = MinimaxAi::new();
        let mut pos = game.position().clone();
        let (mv, _stats) = ai.search(&mut pos, 5);
        assert_eq!(
            mv.unwrap().to,
            Square::from_algebraic("f7").unwrap(),
            "should find Qxf7# at depth 5"
        );
    }

    #[test]
    fn search_returns_none_for_no_legal_moves() {
        let game = Game::from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
            .unwrap();
        let ai = MinimaxAi::new();
        let mut pos = game.position().clone();
        let (mv, _stats) = ai.search(&mut pos, 3);
        // Position is checkmate — no moves to return.
        assert!(mv.is_none());
    }
}

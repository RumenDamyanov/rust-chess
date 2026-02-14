//! Pre-computed attack tables for fast move generation.
//!
//! All tables are initialised once (via `OnceLock`) and live for the lifetime
//! of the process.  Sliding-piece attacks use "plain magic bitboards" — a
//! standard technique that trades ~800 KB of memory for O(1) lookup per square.

use crate::engine::types::{Bitboard, Color, Square};
use std::sync::OnceLock;

// =========================================================================
// Public API
// =========================================================================

/// Get a reference to the global attack tables.
pub fn tables() -> &'static AttackTables {
    static TABLES: OnceLock<AttackTables> = OnceLock::new();
    TABLES.get_or_init(AttackTables::init)
}

/// Pre-computed attack/move tables for every piece type.
pub struct AttackTables {
    pub knight: [Bitboard; 64],
    pub king: [Bitboard; 64],
    /// `pawn_attacks[color][square]` — squares a pawn on `square` attacks.
    pub pawn_attacks: [[Bitboard; 64]; 2],
    /// Rook magic entries (one per square).
    pub rook_magics: [MagicEntry; 64],
    /// Bishop magic entries (one per square).
    pub bishop_magics: [MagicEntry; 64],
    /// Shared attack table backing store for rook magics.
    rook_table: Vec<Bitboard>,
    /// Shared attack table backing store for bishop magics.
    bishop_table: Vec<Bitboard>,
}

/// A single magic-bitboard entry for one square.
pub struct MagicEntry {
    pub mask: Bitboard,
    pub magic: u64,
    pub shift: u8,
    /// Offset into the shared attack table.
    pub offset: usize,
}

impl AttackTables {
    // -------------------------------------------------------------------
    // Leaper lookups
    // -------------------------------------------------------------------

    /// Knight attacks from a square.
    #[inline]
    pub fn knight_attacks(&self, sq: Square) -> Bitboard {
        self.knight[sq.0 as usize]
    }

    /// King attacks from a square.
    #[inline]
    pub fn king_attacks(&self, sq: Square) -> Bitboard {
        self.king[sq.0 as usize]
    }

    /// Pawn attack squares for a given colour.
    #[inline]
    pub fn pawn_attacks(&self, color: Color, sq: Square) -> Bitboard {
        self.pawn_attacks[color.index()][sq.0 as usize]
    }

    // -------------------------------------------------------------------
    // Slider lookups (magic bitboards)
    // -------------------------------------------------------------------

    /// Rook attacks from `sq` given current `occupied` bitboard.
    #[inline]
    pub fn rook_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        let entry = &self.rook_magics[sq.0 as usize];
        let idx = magic_index(entry, occupied);
        self.rook_table[entry.offset + idx]
    }

    /// Bishop attacks from `sq` given current `occupied` bitboard.
    #[inline]
    pub fn bishop_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        let entry = &self.bishop_magics[sq.0 as usize];
        let idx = magic_index(entry, occupied);
        self.bishop_table[entry.offset + idx]
    }

    /// Queen attacks = rook | bishop.
    #[inline]
    pub fn queen_attacks(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        self.rook_attacks(sq, occupied) | self.bishop_attacks(sq, occupied)
    }
}

// =========================================================================
// Magic index computation
// =========================================================================

#[inline]
fn magic_index(entry: &MagicEntry, occupied: Bitboard) -> usize {
    let blockers = occupied & entry.mask;
    let hash = blockers.0.wrapping_mul(entry.magic);
    (hash >> entry.shift) as usize
}

// =========================================================================
// Initialisation
// =========================================================================

impl AttackTables {
    fn init() -> Self {
        let knight = init_knight_attacks();
        let king = init_king_attacks();
        let pawn_attacks = init_pawn_attacks();
        let (rook_magics, rook_table) = init_rook_magics();
        let (bishop_magics, bishop_table) = init_bishop_magics();

        AttackTables {
            knight,
            king,
            pawn_attacks,
            rook_magics,
            bishop_magics,
            rook_table,
            bishop_table,
        }
    }
}

// =========================================================================
// Knight attacks
// =========================================================================

fn init_knight_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard::EMPTY; 64];
    let offsets: [(i8, i8); 8] = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ];
    for sq in 0..64u8 {
        let file = (sq & 7) as i8;
        let rank = (sq >> 3) as i8;
        let mut bb = 0u64;
        for &(dr, df) in &offsets {
            let r = rank + dr;
            let f = file + df;
            if (0..8).contains(&r) && (0..8).contains(&f) {
                bb |= 1u64 << (r * 8 + f);
            }
        }
        table[sq as usize] = Bitboard(bb);
    }
    table
}

// =========================================================================
// King attacks
// =========================================================================

fn init_king_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard::EMPTY; 64];
    let offsets: [(i8, i8); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    for sq in 0..64u8 {
        let file = (sq & 7) as i8;
        let rank = (sq >> 3) as i8;
        let mut bb = 0u64;
        for &(dr, df) in &offsets {
            let r = rank + dr;
            let f = file + df;
            if (0..8).contains(&r) && (0..8).contains(&f) {
                bb |= 1u64 << (r * 8 + f);
            }
        }
        table[sq as usize] = Bitboard(bb);
    }
    table
}

// =========================================================================
// Pawn attacks
// =========================================================================

fn init_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut table = [[Bitboard::EMPTY; 64]; 2];
    for sq in 0..64u8 {
        let file = (sq & 7) as i8;
        let rank = (sq >> 3) as i8;

        // White pawns attack NW and NE (rank + 1).
        if rank < 7 {
            let mut bb = 0u64;
            if file > 0 {
                bb |= 1u64 << ((rank + 1) * 8 + (file - 1));
            }
            if file < 7 {
                bb |= 1u64 << ((rank + 1) * 8 + (file + 1));
            }
            table[Color::White.index()][sq as usize] = Bitboard(bb);
        }

        // Black pawns attack SW and SE (rank - 1).
        if rank > 0 {
            let mut bb = 0u64;
            if file > 0 {
                bb |= 1u64 << ((rank - 1) * 8 + (file - 1));
            }
            if file < 7 {
                bb |= 1u64 << ((rank - 1) * 8 + (file + 1));
            }
            table[Color::Black.index()][sq as usize] = Bitboard(bb);
        }
    }
    table
}

// =========================================================================
// Magic bitboards — shared helpers
// =========================================================================

/// Enumerate all subsets of `mask` using the Carry-Rippler trick.
fn enumerate_subsets(mask: u64) -> Vec<u64> {
    let mut subsets = Vec::new();
    let mut subset = 0u64;
    loop {
        subsets.push(subset);
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 {
            break;
        }
    }
    subsets
}

/// Compute sliding attacks along rays from `sq`, using `blockers` as obstacles.
/// `deltas` lists the (rank_delta, file_delta) ray directions.
fn sliding_attacks(sq: u8, blockers: u64, deltas: &[(i8, i8)]) -> u64 {
    let file = (sq & 7) as i8;
    let rank = (sq >> 3) as i8;
    let mut attacks = 0u64;
    for &(dr, df) in deltas {
        let mut r = rank + dr;
        let mut f = file + df;
        while (0..8).contains(&r) && (0..8).contains(&f) {
            let bit = 1u64 << (r * 8 + f);
            attacks |= bit;
            if blockers & bit != 0 {
                break; // blocked
            }
            r += dr;
            f += df;
        }
    }
    attacks
}

/// Relevant blocker mask for a rook on `sq` (excludes edge squares on the ray).
fn rook_mask(sq: u8) -> u64 {
    let file = (sq & 7) as i8;
    let rank = (sq >> 3) as i8;
    let mut mask = 0u64;
    // Horizontal ray — exclude edges (file 0 and 7).
    for f in 1..7i8 {
        if f != file {
            mask |= 1u64 << (rank * 8 + f);
        }
    }
    // Vertical ray — exclude edges (rank 0 and 7).
    for r in 1..7i8 {
        if r != rank {
            mask |= 1u64 << (r * 8 + file);
        }
    }
    mask
}

/// Relevant blocker mask for a bishop on `sq` (excludes board edges).
fn bishop_mask(sq: u8) -> u64 {
    let file = (sq & 7) as i8;
    let rank = (sq >> 3) as i8;
    let mut mask = 0u64;
    for &(dr, df) in &[(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
        let mut r = rank + dr;
        let mut f = file + df;
        while (1..7).contains(&r) && (1..7).contains(&f) {
            mask |= 1u64 << (r * 8 + f);
            r += dr;
            f += df;
        }
    }
    mask
}

const ROOK_DELTAS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const BISHOP_DELTAS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];

// =========================================================================
// Magic table initialization — runtime magic finder
// =========================================================================

/// Simple xorshift64 PRNG for magic number search.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a sparse random number (few bits set) — much more likely to be a
/// valid magic. Standard technique: AND three random values together.
fn sparse_random(state: &mut u64) -> u64 {
    xorshift64(state) & xorshift64(state) & xorshift64(state)
}

fn init_rook_magics() -> ([MagicEntry; 64], Vec<Bitboard>) {
    find_magics(rook_mask, &ROOK_DELTAS, 0xABCD_1234_5678_EF01)
}

fn init_bishop_magics() -> ([MagicEntry; 64], Vec<Bitboard>) {
    find_magics(bishop_mask, &BISHOP_DELTAS, 0x1234_ABCD_EF01_5678)
}

/// Find magic numbers at runtime for all 64 squares.
///
/// For each square, trial-and-error with sparse random candidates until we
/// find a collision-free mapping. Typically <100 candidates per square,
/// total init takes well under 100ms.
fn find_magics(
    mask_fn: fn(u8) -> u64,
    deltas: &[(i8, i8)],
    seed: u64,
) -> ([MagicEntry; 64], Vec<Bitboard>) {
    let mut rng = seed;
    let mut all_tables: Vec<Bitboard> = Vec::new();
    let mut entries: [MagicEntry; 64] = std::array::from_fn(|_| MagicEntry {
        mask: Bitboard::EMPTY,
        magic: 0,
        shift: 0,
        offset: 0,
    });

    for sq in 0..64u8 {
        let mask = mask_fn(sq);
        let bits = mask.count_ones() as u8;
        let shift = 64 - bits;
        let table_size = 1usize << bits;

        // Pre-compute all blocker subsets and their attack sets.
        let subsets = enumerate_subsets(mask);
        let attacks: Vec<u64> = subsets
            .iter()
            .map(|&b| sliding_attacks(sq, b, deltas))
            .collect();

        // Search for a magic that maps every subset to a unique index
        // (or to the same attack set — "constructive collision" is OK).
        let magic = 'search: loop {
            let candidate = sparse_random(&mut rng);

            // Quick reject: want at least 6 bits in the upper byte of
            // candidate * mask to get good hash distribution.
            if (candidate.wrapping_mul(mask) & 0xFF00_0000_0000_0000).count_ones() < 6 {
                continue;
            }

            let mut table = vec![u64::MAX; table_size]; // sentinel
            let mut ok = true;

            for (i, &blockers) in subsets.iter().enumerate() {
                let idx = (blockers.wrapping_mul(candidate) >> shift) as usize;
                if table[idx] == u64::MAX {
                    table[idx] = attacks[i];
                } else if table[idx] != attacks[i] {
                    ok = false;
                    break;
                }
            }

            if ok {
                break 'search candidate;
            }
        };

        let offset = all_tables.len();
        entries[sq as usize] = MagicEntry {
            mask: Bitboard(mask),
            magic,
            shift,
            offset,
        };

        // Build the final table for this square.
        let mut table = vec![Bitboard::EMPTY; table_size];
        for (i, &blockers) in subsets.iter().enumerate() {
            let idx = (blockers.wrapping_mul(magic) >> shift) as usize;
            table[idx] = Bitboard(attacks[i]);
        }
        all_tables.extend_from_slice(&table);
    }

    (entries, all_tables)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(name: &str) -> Square {
        Square::from_algebraic(name).unwrap()
    }

    // -------------------------------------------------------------------
    // Knight
    // -------------------------------------------------------------------

    #[test]
    fn knight_center_attacks() {
        let t = tables();
        let attacks = t.knight_attacks(sq("e4"));
        // A knight on e4 attacks: d2, f2, c3, g3, c5, g5, d6, f6 = 8 squares.
        assert_eq!(attacks.pop_count(), 8);
        for name in ["d2", "f2", "c3", "g3", "c5", "g5", "d6", "f6"] {
            assert!(
                attacks.is_set(sq(name)),
                "knight on e4 should attack {name}"
            );
        }
    }

    #[test]
    fn knight_corner_attacks() {
        let t = tables();
        let attacks = t.knight_attacks(sq("a1"));
        assert_eq!(attacks.pop_count(), 2);
        assert!(attacks.is_set(sq("b3")));
        assert!(attacks.is_set(sq("c2")));
    }

    #[test]
    fn knight_edge_attacks() {
        let t = tables();
        let attacks = t.knight_attacks(sq("a4"));
        assert_eq!(attacks.pop_count(), 4); // b2, c3, c5, b6
    }

    // -------------------------------------------------------------------
    // King
    // -------------------------------------------------------------------

    #[test]
    fn king_center_attacks() {
        let t = tables();
        let attacks = t.king_attacks(sq("e4"));
        assert_eq!(attacks.pop_count(), 8);
    }

    #[test]
    fn king_corner_attacks() {
        let t = tables();
        let attacks = t.king_attacks(sq("a1"));
        assert_eq!(attacks.pop_count(), 3); // a2, b1, b2
    }

    // -------------------------------------------------------------------
    // Pawn attacks
    // -------------------------------------------------------------------

    #[test]
    fn white_pawn_attacks() {
        let t = tables();
        let atk = t.pawn_attacks(Color::White, sq("e4"));
        assert_eq!(atk.pop_count(), 2);
        assert!(atk.is_set(sq("d5")));
        assert!(atk.is_set(sq("f5")));
    }

    #[test]
    fn black_pawn_attacks() {
        let t = tables();
        let atk = t.pawn_attacks(Color::Black, sq("e4"));
        assert_eq!(atk.pop_count(), 2);
        assert!(atk.is_set(sq("d3")));
        assert!(atk.is_set(sq("f3")));
    }

    #[test]
    fn pawn_attacks_a_file() {
        let t = tables();
        // White pawn on a2 attacks only b3.
        let atk = t.pawn_attacks(Color::White, sq("a2"));
        assert_eq!(atk.pop_count(), 1);
        assert!(atk.is_set(sq("b3")));
    }

    #[test]
    fn pawn_attacks_h_file() {
        let t = tables();
        let atk = t.pawn_attacks(Color::White, sq("h2"));
        assert_eq!(atk.pop_count(), 1);
        assert!(atk.is_set(sq("g3")));
    }

    // -------------------------------------------------------------------
    // Rook attacks (magic bitboards)
    // -------------------------------------------------------------------

    #[test]
    fn rook_attacks_empty_board() {
        let t = tables();
        let attacks = t.rook_attacks(sq("e4"), Bitboard::EMPTY);
        // On an empty board from e4: 4+3+4+3 = 14 squares.
        assert_eq!(attacks.pop_count(), 14);
    }

    #[test]
    fn rook_attacks_blocked() {
        let t = tables();
        // Place a blocker on e6.
        let occ = Bitboard::from_square(sq("e6"));
        let attacks = t.rook_attacks(sq("e4"), occ);
        // Upward: e5, e6 (blocked). Cannot reach e7, e8.
        assert!(attacks.is_set(sq("e5")));
        assert!(attacks.is_set(sq("e6"))); // can capture the blocker
        assert!(!attacks.is_set(sq("e7")));
    }

    #[test]
    fn rook_attacks_corner() {
        let t = tables();
        let attacks = t.rook_attacks(sq("a1"), Bitboard::EMPTY);
        assert_eq!(attacks.pop_count(), 14);
    }

    // -------------------------------------------------------------------
    // Bishop attacks (magic bitboards)
    // -------------------------------------------------------------------

    #[test]
    fn bishop_attacks_empty_board() {
        let t = tables();
        let attacks = t.bishop_attacks(sq("e4"), Bitboard::EMPTY);
        // e4 bishop: diagonals reach 13 squares.
        assert_eq!(attacks.pop_count(), 13);
    }

    #[test]
    fn bishop_attacks_blocked() {
        let t = tables();
        let occ = Bitboard::from_square(sq("c6"));
        let attacks = t.bishop_attacks(sq("e4"), occ);
        // NW diagonal: d5, c6 (blocked). Cannot reach b7, a8.
        assert!(attacks.is_set(sq("d5")));
        assert!(attacks.is_set(sq("c6")));
        assert!(!attacks.is_set(sq("b7")));
    }

    #[test]
    fn bishop_attacks_corner() {
        let t = tables();
        let attacks = t.bishop_attacks(sq("a1"), Bitboard::EMPTY);
        // a1 bishop: only NE diagonal — 7 squares.
        assert_eq!(attacks.pop_count(), 7);
    }

    // -------------------------------------------------------------------
    // Queen attacks
    // -------------------------------------------------------------------

    #[test]
    fn queen_attacks_empty_board() {
        let t = tables();
        let attacks = t.queen_attacks(sq("e4"), Bitboard::EMPTY);
        // Rook(14) + Bishop(13) = 27.
        assert_eq!(attacks.pop_count(), 27);
    }

    // -------------------------------------------------------------------
    // Sanity: all 64 squares have valid tables
    // -------------------------------------------------------------------

    #[test]
    fn all_knight_tables_populated() {
        let t = tables();
        for sq in 0..64u8 {
            // Every square should have at least 2 knight attacks.
            assert!(
                t.knight[sq as usize].pop_count() >= 2,
                "knight table empty for sq {sq}"
            );
        }
    }

    #[test]
    fn all_king_tables_populated() {
        let t = tables();
        for sq in 0..64u8 {
            assert!(
                t.king[sq as usize].pop_count() >= 3,
                "king table empty for sq {sq}"
            );
        }
    }

    #[test]
    fn rook_magic_no_collisions() {
        // If magic init succeeded without panic, collisions are impossible.
        // This test exists to exercise the init path.
        let t = tables();
        let _ = t.rook_attacks(sq("a1"), Bitboard::EMPTY);
    }

    #[test]
    fn bishop_magic_no_collisions() {
        let t = tables();
        let _ = t.bishop_attacks(sq("a1"), Bitboard::EMPTY);
    }
}

//! Zobrist hashing for incremental position identification.
//!
//! Each aspect of a position (piece on square, side to move, castling rights,
//! en passant file) gets a random 64-bit key. The position hash is the XOR of
//! all applicable keys. This allows O(1) incremental updates on make/undo.

use crate::engine::types::{Color, PieceType, Square};

// ---------------------------------------------------------------------------
// Table dimensions
// ---------------------------------------------------------------------------

/// 16 possible castling-rights bitmasks (0..15).
const CASTLING_KEYS: usize = 16;
/// 8 en-passant files (a..h). We only hash the file, not the full square.
const EP_KEYS: usize = 8;
/// Total number of random keys.
#[cfg(test)]
const TOTAL_KEYS: usize = 2 * 6 * 64 + 1 + CASTLING_KEYS + EP_KEYS;

// ---------------------------------------------------------------------------
// ZobristKeys — immutable singleton
// ---------------------------------------------------------------------------

/// Pre-computed Zobrist random keys (generated once at startup via `OnceLock`).
pub struct ZobristKeys {
    /// piece\[color\]\[piece_type\]\[square\] — random key for a piece on a square.
    pub piece: [[[u64; 64]; 6]; 2],
    /// XOR this when it is Black's turn to move.
    pub side_to_move: u64,
    /// castling\[rights_as_u8\] — one key per possible castling bitmask (0..15).
    pub castling: [u64; CASTLING_KEYS],
    /// en_passant\[file\] — one key per possible en-passant file.
    pub en_passant: [u64; EP_KEYS],
}

/// Static singleton holding the Zobrist keys (initialised once).
static ZOBRIST: std::sync::OnceLock<ZobristKeys> = std::sync::OnceLock::new();

/// Get a reference to the global Zobrist keys.
pub fn keys() -> &'static ZobristKeys {
    ZOBRIST.get_or_init(ZobristKeys::init)
}

impl ZobristKeys {
    /// Generate all keys using a deterministic PRNG seeded with a fixed value.
    /// Using a fixed seed ensures reproducible hashes across runs.
    fn init() -> Self {
        let mut rng = Xorshift64::new(0x3243_F6A8_885A_308D); // π digits

        let mut piece = [[[0u64; 64]; 6]; 2];
        for color in &mut piece {
            for pt in color {
                for sq in pt {
                    *sq = rng.next_u64();
                }
            }
        }

        let side_to_move = rng.next_u64();

        let mut castling = [0u64; CASTLING_KEYS];
        for key in &mut castling {
            *key = rng.next_u64();
        }

        let mut en_passant = [0u64; EP_KEYS];
        for key in &mut en_passant {
            *key = rng.next_u64();
        }

        ZobristKeys {
            piece,
            side_to_move,
            castling,
            en_passant,
        }
    }

    // -----------------------------------------------------------------------
    // Convenience accessors
    // -----------------------------------------------------------------------

    /// Key for a specific piece on a specific square.
    #[inline]
    pub fn piece_key(&self, color: Color, piece: PieceType, sq: Square) -> u64 {
        self.piece[color.index()][piece.index()][sq.0 as usize]
    }

    /// Key for a specific en-passant file (0-7).
    #[inline]
    pub fn ep_key(&self, file: u8) -> u64 {
        self.en_passant[file as usize]
    }

    /// Key for a specific castling-rights bitmask.
    #[inline]
    pub fn castling_key(&self, rights: u8) -> u64 {
        self.castling[rights as usize]
    }
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64)
// ---------------------------------------------------------------------------

/// Minimal xorshift64 PRNG — deterministic, fast, good distribution.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure state is never zero (xorshift zero → always zero).
        Xorshift64 {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_initialised() {
        let k = keys();
        // Side-to-move key must be nonzero.
        assert_ne!(k.side_to_move, 0);
    }

    #[test]
    fn keys_are_deterministic() {
        let k1 = keys();
        let k2 = keys();
        // Same pointer (OnceLock singleton).
        assert!(std::ptr::eq(k1, k2));
        // Same values (deterministic seed).
        assert_eq!(
            k1.piece_key(Color::White, PieceType::King, Square(4)),
            k2.piece_key(Color::White, PieceType::King, Square(4)),
        );
    }

    #[test]
    fn piece_keys_unique() {
        let k = keys();
        // Spot-check: no two piece keys should collide.
        let a = k.piece_key(Color::White, PieceType::Pawn, Square(0));
        let b = k.piece_key(Color::White, PieceType::Pawn, Square(1));
        let c = k.piece_key(Color::Black, PieceType::Pawn, Square(0));
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn castling_keys_unique() {
        let k = keys();
        // All 16 castling keys should be distinct.
        let mut set = std::collections::HashSet::new();
        for i in 0..16u8 {
            assert!(
                set.insert(k.castling_key(i)),
                "duplicate castling key for {i}"
            );
        }
    }

    #[test]
    fn ep_keys_unique() {
        let k = keys();
        let mut set = std::collections::HashSet::new();
        for f in 0..8u8 {
            assert!(set.insert(k.ep_key(f)), "duplicate EP key for file {f}");
        }
    }

    #[test]
    fn total_key_count() {
        // Sanity: we expect the constant to be correct.
        assert_eq!(TOTAL_KEYS, 768 + 1 + 16 + 8);
    }

    #[test]
    fn xorshift_never_zero() {
        let mut rng = Xorshift64::new(42);
        for _ in 0..10_000 {
            let v = rng.next_u64();
            assert_ne!(v, 0, "xorshift produced zero");
        }
    }

    #[test]
    fn xorshift_distribution_basic() {
        // Very rough check: < 10% of generated values should share the same
        // upper nibble. This catches catastrophic RNG failures.
        let mut rng = Xorshift64::new(123456);
        let mut buckets = [0u32; 16];
        let n = 10_000u32;
        for _ in 0..n {
            let v = rng.next_u64();
            buckets[(v >> 60) as usize] += 1;
        }
        for (i, &count) in buckets.iter().enumerate() {
            assert!(
                count < n / 5,
                "bucket {i} has {count}/{n} — distribution looks biased",
            );
        }
    }
}

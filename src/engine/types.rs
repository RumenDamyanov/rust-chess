use std::fmt;

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

/// The two sides in a chess game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    /// Index for array lookups: White=0, Black=1.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl std::ops::Not for Color {
    type Output = Self;
    fn not(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::White => write!(f, "white"),
            Color::Black => write!(f, "black"),
        }
    }
}

// ---------------------------------------------------------------------------
// PieceType
// ---------------------------------------------------------------------------

/// The six piece kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    /// All piece types in order.
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    /// Number of piece types.
    pub const COUNT: usize = 6;

    /// Index for array lookups: Pawn=0 .. King=5.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Material value in centipawns.
    pub fn value(self) -> i32 {
        match self {
            PieceType::Pawn => 100,
            PieceType::Knight => 320,
            PieceType::Bishop => 330,
            PieceType::Rook => 500,
            PieceType::Queen => 900,
            PieceType::King => 0, // not used numerically
        }
    }

    /// Single uppercase letter for white, lowercase for black.
    pub fn to_char(self, color: Color) -> char {
        let c = match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        };
        match color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    /// Parse a piece character (case-insensitive for type; use separate color).
    pub fn from_char(c: char) -> Option<(Color, PieceType)> {
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let piece = match c.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _ => return None,
        };
        Some((color, piece))
    }
}

impl fmt::Display for PieceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PieceType::Pawn => write!(f, "pawn"),
            PieceType::Knight => write!(f, "knight"),
            PieceType::Bishop => write!(f, "bishop"),
            PieceType::Rook => write!(f, "rook"),
            PieceType::Queen => write!(f, "queen"),
            PieceType::King => write!(f, "king"),
        }
    }
}

// ---------------------------------------------------------------------------
// Square
// ---------------------------------------------------------------------------

/// A square on the chess board (0..63, LERF: a1=0, h8=63).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Square(pub u8);

impl Square {
    pub const NUM: usize = 64;

    #[inline]
    pub fn new(index: u8) -> Self {
        debug_assert!(index < 64, "Square index out of range: {index}");
        Square(index)
    }

    #[inline]
    pub fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline]
    pub fn rank(self) -> u8 {
        self.0 >> 3
    }

    #[inline]
    pub fn from_file_rank(file: u8, rank: u8) -> Self {
        debug_assert!(file < 8 && rank < 8);
        Square(rank * 8 + file)
    }

    /// Parse algebraic notation like "e4".
    pub fn from_algebraic(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some(Square::from_file_rank(file, rank))
        } else {
            None
        }
    }

    /// Convert to algebraic notation like "e4".
    pub fn to_algebraic(self) -> String {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        format!("{file}{rank}")
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_algebraic())
    }
}

// ---------------------------------------------------------------------------
// Bitboard
// ---------------------------------------------------------------------------

/// A 64-bit bitboard — one bit per square.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(!0u64);

    #[inline]
    pub fn from_square(sq: Square) -> Self {
        Bitboard(1u64 << sq.0)
    }

    #[inline]
    pub fn is_set(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.0) != 0
    }

    #[inline]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u64 << sq.0;
    }

    #[inline]
    pub fn clear(&mut self, sq: Square) {
        self.0 &= !(1u64 << sq.0);
    }

    #[inline]
    pub fn pop_count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_not_empty(self) -> bool {
        self.0 != 0
    }

    /// Least significant bit index (first set square).
    #[inline]
    pub fn lsb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square(self.0.trailing_zeros() as u8))
        }
    }

    /// Pop the least significant bit, returning the square.
    #[inline]
    pub fn pop_lsb(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            let sq = Square(self.0.trailing_zeros() as u8);
            self.0 &= self.0 - 1; // clear LSB
            Some(sq)
        }
    }

    /// Iterate over all set bit positions as `Square`s.
    #[inline]
    pub fn iter(self) -> BitboardIter {
        BitboardIter(self)
    }
}

/// Iterator over set bits in a `Bitboard`.
pub struct BitboardIter(Bitboard);

impl Iterator for BitboardIter {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        self.0.pop_lsb()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.0.pop_count() as usize;
        (count, Some(count))
    }
}

impl ExactSizeIterator for BitboardIter {}

impl std::ops::BitAnd for Bitboard {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Bitboard(self.0 & rhs.0)
    }
}

impl std::ops::BitOr for Bitboard {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Bitboard(self.0 | rhs.0)
    }
}

impl std::ops::BitXor for Bitboard {
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl std::ops::Not for Bitboard {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Bitboard(!self.0)
    }
}

impl std::ops::BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Bitboard(0x{:016x})", self.0)?;
        for rank in (0..8).rev() {
            write!(f, "  {} ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::from_file_rank(file, rank);
                write!(f, "{}", if self.is_set(sq) { '1' } else { '.' })?;
                if file < 7 {
                    write!(f, " ")?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "    a b c d e f g h")
    }
}

// ---------------------------------------------------------------------------
// MoveFlags
// ---------------------------------------------------------------------------

/// Flags for special move types packed in a single byte.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoveFlags(pub u8);

impl MoveFlags {
    pub const NONE: MoveFlags = MoveFlags(0);
    pub const CAPTURE: MoveFlags = MoveFlags(1);
    pub const EN_PASSANT: MoveFlags = MoveFlags(2);
    pub const CASTLING: MoveFlags = MoveFlags(4);
    pub const DOUBLE_PUSH: MoveFlags = MoveFlags(8);

    #[inline]
    pub fn is_capture(self) -> bool {
        self.0 & Self::CAPTURE.0 != 0
    }

    #[inline]
    pub fn is_en_passant(self) -> bool {
        self.0 & Self::EN_PASSANT.0 != 0
    }

    #[inline]
    pub fn is_castling(self) -> bool {
        self.0 & Self::CASTLING.0 != 0
    }

    #[inline]
    pub fn is_double_push(self) -> bool {
        self.0 & Self::DOUBLE_PUSH.0 != 0
    }
}

impl std::ops::BitOr for MoveFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        MoveFlags(self.0 | rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

/// A chess move: from-square, to-square, optional promotion, and flags.
/// Kept at ≤ 8 bytes so it can be passed by value efficiently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceType>,
    pub flags: MoveFlags,
}

impl Move {
    pub fn new(from: Square, to: Square) -> Self {
        Move {
            from,
            to,
            promotion: None,
            flags: MoveFlags::NONE,
        }
    }

    pub fn with_flags(from: Square, to: Square, flags: MoveFlags) -> Self {
        Move {
            from,
            to,
            promotion: None,
            flags,
        }
    }

    pub fn with_promotion(
        from: Square,
        to: Square,
        promotion: PieceType,
        flags: MoveFlags,
    ) -> Self {
        Move {
            from,
            to,
            promotion: Some(promotion),
            flags,
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from, self.to)?;
        if let Some(promo) = self.promotion {
            write!(f, "={}", promo.to_char(Color::White).to_ascii_lowercase())?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CastlingRights
// ---------------------------------------------------------------------------

/// Castling availability bitfield: bits 0-3 = WK, WQ, BK, BQ.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const NONE: CastlingRights = CastlingRights(0);
    pub const WHITE_KINGSIDE: u8 = 1;
    pub const WHITE_QUEENSIDE: u8 = 2;
    pub const BLACK_KINGSIDE: u8 = 4;
    pub const BLACK_QUEENSIDE: u8 = 8;
    pub const ALL: CastlingRights = CastlingRights(0b1111);

    #[inline]
    pub fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    #[inline]
    pub fn remove(&mut self, flag: u8) {
        self.0 &= !flag;
    }

    #[inline]
    pub fn can_castle_kingside(self, color: Color) -> bool {
        match color {
            Color::White => self.has(Self::WHITE_KINGSIDE),
            Color::Black => self.has(Self::BLACK_KINGSIDE),
        }
    }

    #[inline]
    pub fn can_castle_queenside(self, color: Color) -> bool {
        match color {
            Color::White => self.has(Self::WHITE_QUEENSIDE),
            Color::Black => self.has(Self::BLACK_QUEENSIDE),
        }
    }

    /// Parse FEN castling string (e.g. "KQkq", "-", "Kq").
    pub fn from_fen(s: &str) -> Option<Self> {
        if s == "-" {
            return Some(CastlingRights::NONE);
        }
        let mut rights = 0u8;
        for c in s.chars() {
            match c {
                'K' => rights |= Self::WHITE_KINGSIDE,
                'Q' => rights |= Self::WHITE_QUEENSIDE,
                'k' => rights |= Self::BLACK_KINGSIDE,
                'q' => rights |= Self::BLACK_QUEENSIDE,
                _ => return None,
            }
        }
        Some(CastlingRights(rights))
    }

    /// Convert to FEN castling string.
    pub fn to_fen(self) -> String {
        if self.0 == 0 {
            return "-".to_string();
        }
        let mut s = String::with_capacity(4);
        if self.has(Self::WHITE_KINGSIDE) {
            s.push('K');
        }
        if self.has(Self::WHITE_QUEENSIDE) {
            s.push('Q');
        }
        if self.has(Self::BLACK_KINGSIDE) {
            s.push('k');
        }
        if self.has(Self::BLACK_QUEENSIDE) {
            s.push('q');
        }
        s
    }
}

impl fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_fen())
    }
}

// ---------------------------------------------------------------------------
// GameStatus & Difficulty
// ---------------------------------------------------------------------------

/// Current status of a game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameStatus {
    Active,
    Check,
    Checkmate,
    Stalemate,
    Draw(DrawReason),
}

impl GameStatus {
    pub fn as_str(&self) -> &str {
        match self {
            GameStatus::Active => "active",
            GameStatus::Check => "check",
            GameStatus::Checkmate => "checkmate",
            GameStatus::Stalemate => "stalemate",
            GameStatus::Draw(reason) => reason.as_str(),
        }
    }

    pub fn is_game_over(&self) -> bool {
        matches!(
            self,
            GameStatus::Checkmate | GameStatus::Stalemate | GameStatus::Draw(_)
        )
    }
}

impl fmt::Display for GameStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Reason for a draw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawReason {
    FiftyMoveRule,
    ThreefoldRepetition,
    InsufficientMaterial,
}

impl DrawReason {
    pub fn as_str(&self) -> &str {
        match self {
            DrawReason::FiftyMoveRule => "fifty_move_rule",
            DrawReason::ThreefoldRepetition => "threefold_repetition",
            DrawReason::InsufficientMaterial => "insufficient_material",
        }
    }
}

/// AI difficulty levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Difficulty {
    Harmless,
    Easy,
    Medium,
    Hard,
    Expert,
    Godlike,
}

impl Difficulty {
    /// Parse from string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "harmless" => Some(Difficulty::Harmless),
            "easy" => Some(Difficulty::Easy),
            "medium" => Some(Difficulty::Medium),
            "hard" => Some(Difficulty::Hard),
            "expert" => Some(Difficulty::Expert),
            "godlike" => Some(Difficulty::Godlike),
            _ => None,
        }
    }

    /// Search depth for minimax.
    pub fn depth(self) -> u32 {
        match self {
            Difficulty::Harmless => 0, // random
            Difficulty::Easy => 1,
            Difficulty::Medium => 3,
            Difficulty::Hard => 5,
            Difficulty::Expert => 6,
            Difficulty::Godlike => 8,
        }
    }
}

impl fmt::Display for Difficulty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Difficulty::Harmless => write!(f, "harmless"),
            Difficulty::Easy => write!(f, "easy"),
            Difficulty::Medium => write!(f, "medium"),
            Difficulty::Hard => write!(f, "hard"),
            Difficulty::Expert => write!(f, "expert"),
            Difficulty::Godlike => write!(f, "godlike"),
        }
    }
}

// ---------------------------------------------------------------------------
// ChessError
// ---------------------------------------------------------------------------

/// Domain errors for the chess engine.
#[derive(Debug, thiserror::Error)]
pub enum ChessError {
    #[error("invalid move: {from} -> {to}: {reason}")]
    InvalidMove {
        from: String,
        to: String,
        reason: String,
    },

    #[error("invalid FEN string: {0}")]
    InvalidFen(String),

    #[error("invalid square notation: {0}")]
    InvalidSquare(String),

    #[error("game is already over: {0}")]
    GameOver(String),

    #[error("invalid promotion piece: {0}")]
    InvalidPromotion(String),

    #[error("no moves to undo")]
    NothingToUndo,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_toggle() {
        assert_eq!(!Color::White, Color::Black);
        assert_eq!(!Color::Black, Color::White);
    }

    #[test]
    fn color_display() {
        assert_eq!(Color::White.to_string(), "white");
        assert_eq!(Color::Black.to_string(), "black");
    }

    #[test]
    fn piece_type_values() {
        assert_eq!(PieceType::Pawn.value(), 100);
        assert_eq!(PieceType::Knight.value(), 320);
        assert_eq!(PieceType::Bishop.value(), 330);
        assert_eq!(PieceType::Rook.value(), 500);
        assert_eq!(PieceType::Queen.value(), 900);
        assert_eq!(PieceType::King.value(), 0);
    }

    #[test]
    fn piece_type_char_round_trip() {
        for pt in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            let wc = pt.to_char(Color::White);
            let bc = pt.to_char(Color::Black);
            assert!(wc.is_ascii_uppercase());
            assert!(bc.is_ascii_lowercase());
            assert_eq!(PieceType::from_char(wc), Some((Color::White, pt)));
            assert_eq!(PieceType::from_char(bc), Some((Color::Black, pt)));
        }
    }

    #[test]
    fn piece_type_from_char_invalid() {
        assert_eq!(PieceType::from_char('x'), None);
        assert_eq!(PieceType::from_char('1'), None);
    }

    #[test]
    fn square_from_algebraic() {
        assert_eq!(Square::from_algebraic("a1"), Some(Square(0)));
        assert_eq!(Square::from_algebraic("h1"), Some(Square(7)));
        assert_eq!(Square::from_algebraic("a8"), Some(Square(56)));
        assert_eq!(Square::from_algebraic("h8"), Some(Square(63)));
        assert_eq!(Square::from_algebraic("e4"), Some(Square(28)));
    }

    #[test]
    fn square_to_algebraic() {
        assert_eq!(Square(0).to_algebraic(), "a1");
        assert_eq!(Square(7).to_algebraic(), "h1");
        assert_eq!(Square(56).to_algebraic(), "a8");
        assert_eq!(Square(63).to_algebraic(), "h8");
        assert_eq!(Square(28).to_algebraic(), "e4");
    }

    #[test]
    fn square_algebraic_round_trip() {
        for i in 0..64 {
            let sq = Square(i);
            let alg = sq.to_algebraic();
            assert_eq!(Square::from_algebraic(&alg), Some(sq));
        }
    }

    #[test]
    fn square_file_rank() {
        let e4 = Square::from_algebraic("e4").unwrap();
        assert_eq!(e4.file(), 4); // e = file 4
        assert_eq!(e4.rank(), 3); // 4th rank = index 3
    }

    #[test]
    fn square_from_algebraic_invalid() {
        assert_eq!(Square::from_algebraic(""), None);
        assert_eq!(Square::from_algebraic("a"), None);
        assert_eq!(Square::from_algebraic("a9"), None);
        assert_eq!(Square::from_algebraic("i1"), None);
        assert_eq!(Square::from_algebraic("abc"), None);
    }

    #[test]
    fn bitboard_basic_ops() {
        let mut bb = Bitboard::EMPTY;
        assert!(bb.is_empty());
        assert_eq!(bb.pop_count(), 0);

        let e4 = Square::from_algebraic("e4").unwrap();
        bb.set(e4);
        assert!(bb.is_not_empty());
        assert!(bb.is_set(e4));
        assert_eq!(bb.pop_count(), 1);

        bb.clear(e4);
        assert!(bb.is_empty());
    }

    #[test]
    fn bitboard_lsb_pop() {
        let mut bb = Bitboard::from_square(Square(0)) | Bitboard::from_square(Square(5));
        assert_eq!(bb.pop_count(), 2);

        assert_eq!(bb.lsb(), Some(Square(0)));
        assert_eq!(bb.pop_lsb(), Some(Square(0)));
        assert_eq!(bb.pop_count(), 1);

        assert_eq!(bb.pop_lsb(), Some(Square(5)));
        assert_eq!(bb.pop_lsb(), None);
        assert!(bb.is_empty());
    }

    #[test]
    fn bitboard_bitwise_ops() {
        let a = Bitboard(0xFF);
        let b = Bitboard(0x0F);
        assert_eq!((a & b).0, 0x0F);
        assert_eq!((a | b).0, 0xFF);
        assert_eq!((a ^ b).0, 0xF0);
        assert_eq!((!Bitboard::EMPTY).0, !0u64);
    }

    #[test]
    fn move_flags() {
        let flags = MoveFlags::CAPTURE | MoveFlags::EN_PASSANT;
        assert!(flags.is_capture());
        assert!(flags.is_en_passant());
        assert!(!flags.is_castling());
        assert!(!flags.is_double_push());
    }

    #[test]
    fn move_display() {
        let m = Move::new(
            Square::from_algebraic("e2").unwrap(),
            Square::from_algebraic("e4").unwrap(),
        );
        assert_eq!(m.to_string(), "e2e4");

        let promo = Move::with_promotion(
            Square::from_algebraic("e7").unwrap(),
            Square::from_algebraic("e8").unwrap(),
            PieceType::Queen,
            MoveFlags::NONE,
        );
        assert_eq!(promo.to_string(), "e7e8=q");
    }

    #[test]
    fn castling_rights_fen_round_trip() {
        let cases = ["-", "K", "Kq", "KQkq", "kq", "Q"];
        for s in cases {
            let cr = CastlingRights::from_fen(s).unwrap();
            assert_eq!(cr.to_fen(), s);
        }
    }

    #[test]
    fn castling_rights_flags() {
        let all = CastlingRights::ALL;
        assert!(all.can_castle_kingside(Color::White));
        assert!(all.can_castle_queenside(Color::White));
        assert!(all.can_castle_kingside(Color::Black));
        assert!(all.can_castle_queenside(Color::Black));

        let mut cr = CastlingRights::ALL;
        cr.remove(CastlingRights::WHITE_KINGSIDE);
        assert!(!cr.can_castle_kingside(Color::White));
        assert!(cr.can_castle_queenside(Color::White));
    }

    #[test]
    fn castling_rights_from_fen_invalid() {
        assert_eq!(CastlingRights::from_fen("X"), None);
        assert_eq!(CastlingRights::from_fen("KZ"), None);
    }

    #[test]
    fn game_status_strings() {
        assert_eq!(GameStatus::Active.as_str(), "active");
        assert_eq!(GameStatus::Check.as_str(), "check");
        assert_eq!(GameStatus::Checkmate.as_str(), "checkmate");
        assert_eq!(GameStatus::Stalemate.as_str(), "stalemate");
        assert_eq!(
            GameStatus::Draw(DrawReason::FiftyMoveRule).as_str(),
            "fifty_move_rule"
        );
        assert_eq!(
            GameStatus::Draw(DrawReason::ThreefoldRepetition).as_str(),
            "threefold_repetition"
        );
        assert_eq!(
            GameStatus::Draw(DrawReason::InsufficientMaterial).as_str(),
            "insufficient_material"
        );
    }

    #[test]
    fn game_status_is_game_over() {
        assert!(!GameStatus::Active.is_game_over());
        assert!(!GameStatus::Check.is_game_over());
        assert!(GameStatus::Checkmate.is_game_over());
        assert!(GameStatus::Stalemate.is_game_over());
        assert!(GameStatus::Draw(DrawReason::FiftyMoveRule).is_game_over());
    }

    #[test]
    fn difficulty_depth_mapping() {
        assert_eq!(Difficulty::Harmless.depth(), 0);
        assert_eq!(Difficulty::Easy.depth(), 1);
        assert_eq!(Difficulty::Medium.depth(), 3);
        assert_eq!(Difficulty::Hard.depth(), 5);
        assert_eq!(Difficulty::Expert.depth(), 6);
        assert_eq!(Difficulty::Godlike.depth(), 8);
    }

    #[test]
    fn difficulty_from_str() {
        assert_eq!(
            Difficulty::from_str_loose("medium"),
            Some(Difficulty::Medium)
        );
        assert_eq!(
            Difficulty::from_str_loose("GODLIKE"),
            Some(Difficulty::Godlike)
        );
        assert_eq!(Difficulty::from_str_loose("invalid"), None);
    }

    #[test]
    fn color_index() {
        assert_eq!(Color::White.index(), 0);
        assert_eq!(Color::Black.index(), 1);
    }

    #[test]
    fn piece_type_index() {
        assert_eq!(PieceType::Pawn.index(), 0);
        assert_eq!(PieceType::Knight.index(), 1);
        assert_eq!(PieceType::Bishop.index(), 2);
        assert_eq!(PieceType::Rook.index(), 3);
        assert_eq!(PieceType::Queen.index(), 4);
        assert_eq!(PieceType::King.index(), 5);
    }

    #[test]
    fn piece_type_all_constant() {
        assert_eq!(PieceType::ALL.len(), PieceType::COUNT);
        for (i, &pt) in PieceType::ALL.iter().enumerate() {
            assert_eq!(pt.index(), i);
        }
    }

    #[test]
    fn bitboard_iter() {
        let bb = Bitboard::from_square(Square(0))
            | Bitboard::from_square(Square(10))
            | Bitboard::from_square(Square(63));
        let squares: Vec<Square> = bb.iter().collect();
        assert_eq!(squares, vec![Square(0), Square(10), Square(63)]);
    }

    #[test]
    fn bitboard_iter_empty() {
        let bb = Bitboard::EMPTY;
        assert_eq!(bb.iter().count(), 0);
    }

    #[test]
    fn bitboard_iter_exact_size() {
        let bb = Bitboard::from_square(Square(1)) | Bitboard::from_square(Square(2));
        let iter = bb.iter();
        assert_eq!(iter.len(), 2);
    }
}

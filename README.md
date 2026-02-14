# rust-chess

[![CI](https://github.com/RumenDamyanov/rust-chess/actions/workflows/ci.yml/badge.svg)](https://github.com/RumenDamyanov/rust-chess/actions/workflows/ci.yml)
[![CodeQL](https://github.com/RumenDamyanov/rust-chess/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/RumenDamyanov/rust-chess/actions/workflows/github-code-scanning/codeql)
[![codecov](https://codecov.io/gh/RumenDamyanov/rust-chess/graph/badge.svg)](https://codecov.io/gh/RumenDamyanov/rust-chess)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-025e8c?logo=dependabot)](https://github.com/RumenDamyanov/rust-chess/security/dependabot)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/RumenDamyanov/rust-chess/blob/master/LICENSE.md)

> 📖 **Documentation**: [📚 Complete Wiki](https://github.com/RumenDamyanov/rust-chess/wiki) · [🚀 Quick Start](#quick-start) · [📋 API Reference](#api-endpoints) · [🏗️ Architecture](#architecture)

**rust-chess** is a high-performance chess engine and API server written in Rust. It provides a complete chess implementation with magic-bitboard move generation, AI opponent with alpha-beta pruning, and a RESTful API — all compiled to a single static binary under 3 MB.

**What makes rust-chess special:**

⚡ **Blazing Performance**: Magic bitboards, zero-copy serialization, sub-microsecond move validation — compiled to a single static binary with no runtime dependencies.

🦀 **Rust Safety Guarantees**: Memory safety without garbage collection, fearless concurrency with Tokio, type-safe API with comprehensive error handling.

🤖 **AI Engine**: Negamax with alpha-beta pruning, MVV-LVA move ordering, piece-square tables, 6 difficulty levels from random to depth-8 search.

🐳 **Tiny Docker Image**: Multi-stage build producing a `scratch`-based image at **~3 MB** with built-in health checks.

Part of the **chess platform family** alongside [go-chess](https://github.com/RumenDamyanov/go-chess) and [npm-chess](https://github.com/RumenDamyanov/npm-chess), sharing a unified API design.

## ✨ Key Features

### Core Chess Engine

- **Magic Bitboard Representation**: 64-bit bitboards with runtime-computed magic numbers for sliding piece attacks
- **Complete Rule Implementation**: All chess rules including castling, en passant, pawn promotion (all 4 piece types)
- **Legal Move Generation**: Pseudo-legal generation with legality filtering, pin detection, check evasion
- **Zobrist Hashing**: Incremental position hashing for threefold repetition detection
- **FEN & PGN Support**: Full Forsyth-Edwards Notation import/export and Portable Game Notation export
- **SAN Notation**: Standard Algebraic Notation generation and parsing with full disambiguation
- **Draw Detection**: 50-move rule, threefold repetition, insufficient material (KvK, KBvK, KNvK, KBvKB same color)
- **Game State Management**: Undo/redo with full position restoration, move history tracking

### 🤖 AI Engine

- **Negamax Search**: With alpha-beta pruning for exponential search reduction
- **MVV-LVA Move Ordering**: Most Valuable Victim – Least Valuable Attacker for better pruning
- **Piece-Square Tables**: Positional evaluation beyond material counting
- **Bishop Pair Bonus**: 30 centipawn bonus for the bishop pair advantage
- **Time Budget Control**: Check every 4096 nodes to respect time limits
- **6 Difficulty Levels**: Harmless (random), Easy (depth 1), Medium (depth 3), Hard (depth 5), Expert (depth 6), Godlike (depth 8)

### 🚀 REST API

- **14 Endpoints**: Complete game lifecycle — create, move, undo, AI, analysis, hints, PGN, FEN
- **JSON API**: Consistent request/response format with structured error handling
- **Board Representation**: 8×8 array (rank 8 → rank 1), uppercase = White, lowercase = Black, null = empty
- **CORS Support**: Pre-configured for cross-origin requests (any origin/method/header)
- **Structured Logging**: `tracing`-based request/response logging with configurable levels

### 🛠️ Technical Excellence

- **236 Tests**: 219 unit tests + 17 perft integration tests, ~89% code coverage
- **Zero Warnings**: Clean `cargo clippy -D warnings` and `cargo fmt --check`
- **CI/CD Pipeline**: GitHub Actions with fmt → clippy → test → perft → coverage → audit → Docker
- **2.93 MB Docker Image**: `scratch`-based with built-in `--health-check` flag (no curl needed)
- **Zero Runtime Dependencies**: Static binary with musl libc

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) 1.82+ (stable, edition 2024)

### Build & Run

```bash
# Clone the repository
git clone https://github.com/RumenDamyanov/rust-chess.git
cd rust-chess

# Build and run
cargo run

# Or using Make
make run
```

The server starts on `http://localhost:8082` by default.

### Verify

```bash
curl http://localhost:8082/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "language": "rust",
  "engine": "rust-chess",
  "uptime": 5
}
```

### Play a Quick Game

```bash
# Create a game
GAME=$(curl -s -X POST http://localhost:8082/api/games \
  -H "Content-Type: application/json" \
  -d '{"whitePlayer":"Alice","blackPlayer":"Bob"}')
ID=$(echo $GAME | jq -r '.id')

# Make a move (e2 to e4)
curl -s -X POST "http://localhost:8082/api/games/$ID/moves" \
  -H "Content-Type: application/json" \
  -d '{"from":"e2","to":"e4"}' | jq '.lastMove'

# Ask the AI to respond
curl -s -X POST "http://localhost:8082/api/games/$ID/ai-move" \
  -H "Content-Type: application/json" \
  -d '{"difficulty":"medium"}' | jq '{move: .move, thinkingTime: .thinkingTime}'

# Get analysis
curl -s "http://localhost:8082/api/games/$ID/analysis?depth=5" | jq '{eval: .evaluation, bestMove: .bestMove}'

# Export PGN
curl -s "http://localhost:8082/api/games/$ID/pgn"
```

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8082` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `CHESS_AI_TIMEOUT` | `5000` | AI move timeout in milliseconds |
| `CHESS_AI_DEFAULT_DIFFICULTY` | `medium` | Default AI difficulty |

```bash
PORT=9000 CHESS_AI_DEFAULT_DIFFICULTY=hard cargo run
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server health check with uptime and version |
| `POST` | `/api/games` | Create a new game |
| `GET` | `/api/games` | List all games (pagination, status filter) |
| `GET` | `/api/games/{id}` | Get game state (board, moves, status) |
| `DELETE` | `/api/games/{id}` | Delete a game |
| `POST` | `/api/games/{id}/moves` | Make a move (from/to or SAN) |
| `GET` | `/api/games/{id}/moves` | Get move history |
| `POST` | `/api/games/{id}/undo` | Undo last move |
| `POST` | `/api/games/{id}/ai-move` | AI makes a move |
| `POST` | `/api/games/{id}/ai-hint` | Get AI hint (doesn't modify game) |
| `GET` | `/api/games/{id}/legal-moves` | Get legal moves (optional `?from=e2` filter) |
| `POST` | `/api/games/{id}/fen` | Load position from FEN |
| `GET` | `/api/games/{id}/pgn` | Export game as PGN (text/plain) |
| `GET` | `/api/games/{id}/analysis` | Position analysis (eval, best move, depth) |

### Error Response Format

All errors return structured JSON:

```json
{
  "error": {
    "code": "INVALID_MOVE",
    "message": "Move e2e5 is not legal in the current position"
  }
}
```

Error codes: `GAME_NOT_FOUND`, `INVALID_MOVE`, `INVALID_FEN`, `GAME_OVER`, `NOTHING_TO_UNDO`, `INVALID_SQUARE`, `INTERNAL_ERROR`.

## Development

### Common Commands

```bash
make build          # Debug build
make release        # Optimized release build
make test           # Run all tests
make test-verbose   # Run tests with output
make lint           # Run clippy linter
make fmt            # Format code
make check          # fmt + lint + test (pre-commit)
make coverage       # Generate coverage report (needs cargo-tarpaulin)
make audit          # Security audit (needs cargo-audit)
make clean          # Clean build artifacts
```

### Docker

```bash
make docker         # Build Docker image (~3 MB)
make docker-run     # Run in container
make docker-size    # Check image size
```

```bash
# Or manually
docker build -t rust-chess .
docker run -p 8082:8082 rust-chess

# With custom port
docker run -e PORT=9000 -p 9000:9000 rust-chess
```

The `scratch`-based image includes a built-in health check via `--health-check` flag (no curl/wget needed).

## Architecture

```
src/
├── main.rs              # Entry point: CLI flags, tracing, server startup
├── lib.rs               # Crate root: module declarations
├── config.rs            # Environment variable configuration
├── engine/
│   ├── mod.rs           # Module re-exports
│   ├── types.rs         # Core types: Color, PieceType, Square, Bitboard, Move, etc.
│   ├── zobrist.rs       # Zobrist hashing (deterministic PRNG, OnceLock singleton)
│   ├── board.rs         # Position: bitboard arrays, make/undo move, FEN, attack detection
│   ├── attacks.rs       # Magic bitboard attack tables (runtime-computed magics)
│   ├── movegen.rs       # Legal move generation (pseudo-legal + legality filter)
│   ├── game.rs          # Game controller: history, undo, status detection, draw rules
│   ├── san.rs           # SAN generation & parsing with disambiguation
│   └── pgn.rs           # PGN export with Seven Tag Roster
├── ai/
│   ├── mod.rs           # Module re-exports
│   ├── evaluation.rs    # Material + PST + bishop pair evaluation
│   └── engine.rs        # AiEngine trait, RandomAi, MinimaxAi (negamax + alpha-beta)
└── api/
    ├── mod.rs           # Module declarations
    ├── router.rs        # Axum router with all routes and middleware
    ├── handlers.rs      # 14 request handler functions
    ├── models.rs        # Request/response JSON models + conversion helpers
    ├── errors.rs        # Structured API error handling with HTTP status mapping
    └── state.rs         # Shared application state (Arc<RwLock<HashMap<String, Game>>>)
```

### Design Principles

- **Magic Bitboards**: Runtime-computed magic numbers for O(1) sliding piece attack lookups
- **LERF Mapping**: Little-Endian Rank-File (a1=0, h8=63) for cache-friendly bitboard operations
- **Incremental Zobrist**: Position hash updated via XOR on every make/undo for O(1) repetition detection
- **Type-safe moves**: Moves validated at the type level with `MoveFlags` (capture/castle/ep/promo)
- **Shared API design**: Endpoint paths and JSON schemas align with go-chess and npm-chess
- **Error propagation**: `thiserror` derives with automatic HTTP status mapping

### Performance Notes

- **Move generation**: Magic bitboards provide O(1) attack lookups for bishops, rooks, and queens
- **AI search**: Alpha-beta with MVV-LVA ordering typically prunes ~90% of nodes at depth 5
- **Memory**: Each `Position` is ~200 bytes (12 bitboards + metadata), fully stack-allocated
- **Binary size**: Release build with LTO + strip produces a ~2.5 MB static binary

## Project Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Project Scaffolding | ✅ Complete |
| 2 | Core Types & Board Representation | ✅ Complete |
| 3 | Move Generation (Magic Bitboards) | ✅ Complete |
| 4 | Game Logic (SAN, PGN, Draw Detection) | ✅ Complete |
| 5 | AI Engine (Minimax + Alpha-Beta) | ✅ Complete |
| 6 | REST API (14 Endpoints) | ✅ Complete |
| 7 | Docker, CI & Polish | ✅ Complete |

## License

This project is licensed under the MIT License — see the [LICENSE.md](LICENSE.md) file for details.

## Related Projects

- [go-chess](https://github.com/RumenDamyanov/go-chess) — Go chess engine & API (port 8080)
- [npm-chess](https://github.com/RumenDamyanov/npm-chess) — TypeScript chess library (port 8081)
- [js-chess](https://github.com/RumenDamyanov/js-chess) — JavaScript chess frontend
- [react-chess](https://github.com/RumenDamyanov/react-chess) — React chess UI

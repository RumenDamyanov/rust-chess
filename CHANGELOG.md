# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-14

### Added

#### Core Engine
- Magic bitboard-based board representation (LERF mapping: a1=0, h8=63)
- Runtime magic number computation for sliding piece attacks (bishop, rook, queen)
- Zobrist hashing with incremental XOR updates for position identification
- Complete legal move generation with pseudo-legal + legality filter
- Full chess rules: castling (KS/QS), en passant, pawn promotion (N/B/R/Q)
- Check, checkmate, and stalemate detection
- Draw detection: 50-move rule, threefold repetition, insufficient material
- FEN import/export (Forsyth-Edwards Notation)
- SAN generation and parsing with full disambiguation
- PGN export with Seven Tag Roster and SetUp/FEN tags
- Game controller with undo/redo, move history, and position tracking

#### AI Engine
- `AiEngine` trait with pluggable implementations
- `RandomAi` — random legal move selection (Harmless difficulty)
- `MinimaxAi` — negamax search with alpha-beta pruning
- MVV-LVA (Most Valuable Victim – Least Valuable Attacker) move ordering
- Piece-square tables for positional evaluation (pawn, knight, bishop, rook, queen, king)
- Bishop pair bonus (30 centipawns)
- Time budget control (check every 4096 nodes)
- 6 difficulty levels: Harmless (random), Easy (depth 1), Medium (depth 3), Hard (depth 5), Expert (depth 6), Godlike (depth 8)

#### REST API
- 14 HTTP endpoints via Axum 0.8 on port 8082
- `GET /health` — server health with uptime, version, and engine info
- `POST /api/games` — create game with player names
- `GET /api/games` — list games with pagination and status filter
- `GET /api/games/{id}` — full game state (board array, FEN, moves, captures)
- `DELETE /api/games/{id}` — delete game
- `POST /api/games/{id}/moves` — make move (from/to coordinates)
- `GET /api/games/{id}/moves` — move history
- `POST /api/games/{id}/undo` — undo last move
- `POST /api/games/{id}/ai-move` — AI plays a move at specified difficulty
- `POST /api/games/{id}/ai-hint` — get AI hint without modifying game
- `GET /api/games/{id}/legal-moves` — legal moves with optional square filter
- `POST /api/games/{id}/fen` — load position from FEN string
- `GET /api/games/{id}/pgn` — export game as PGN (text/plain)
- `GET /api/games/{id}/analysis` — position analysis with evaluation and best move
- Structured JSON error responses with error codes
- CORS support (permissive: any origin/method/header)
- `tower-http` tracing middleware for request/response logging

#### Infrastructure
- Multi-stage Dockerfile (`rust:alpine` → `scratch`) producing ~3 MB image
- Built-in `--health-check` CLI flag for Docker HEALTHCHECK (no curl needed)
- GitHub Actions CI pipeline: fmt → clippy → test → perft → coverage → audit → Docker
- Makefile with 14 targets (build, run, test, lint, fmt, check, coverage, audit, docker, etc.)
- 236 tests (219 unit + 17 perft integration), ~89% code coverage
- Zero compiler warnings (`cargo clippy -D warnings`)
- Consistent code formatting (`cargo fmt --check`)

[0.1.0]: https://github.com/RumenDamyanov/rust-chess/releases/tag/v0.1.0
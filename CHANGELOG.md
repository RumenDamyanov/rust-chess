# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase 10: WebSocket Real-Time Events
- WebSocket connection manager (`WsManager`) with per-game client tracking and broadcast
- `GET /ws/games/{id}` — WebSocket upgrade endpoint for real-time game event streaming
- Typed message system with 8 server→client event types:
  - `subscribed` — sent on connect with full game state snapshot
  - `move_made` — broadcast after any player move with SAN, from/to, player, FEN
  - `ai_thinking` — broadcast when AI computation starts (includes difficulty)
  - `ai_move_complete` — broadcast when AI finishes with move details and thinking time
  - `game_state` — broadcast after undo/FEN load with updated position
  - `game_over` — broadcast on checkmate, stalemate, or draw
  - `error` — sent for invalid game connections
  - `pong` — response to client ping commands
- Client→server command protocol: `subscribe`, `unsubscribe`, `ping`
- Automatic stale client cleanup when send fails
- REST handler integration: all mutation endpoints (`make_move`, `undo_move`, `ai_move`, `load_fen`) broadcast events to WebSocket subscribers
- Non-blocking broadcast via `tokio::spawn` — broadcasts don't delay REST responses
- Multi-client support: multiple WS clients per game, game-isolated broadcasts
- 35 new tests (341 total: 312 unit + 17 perft + 12 WS integration, zero warnings)

#### Phase 9: LLM Chat
- Multi-provider LLM abstraction with trait-based architecture (`LlmProvider`)
- 5 provider implementations: OpenAI (gpt-4o-mini), Anthropic (claude-3-haiku), Gemini (gemini-1.5-flash), xAI (grok-1.5), DeepSeek (deepseek-chat)
- OpenAI-compatible shared provider for OpenAI, xAI, and DeepSeek APIs
- `POST /api/games/{id}/chat` — game-scoped chat with AI (context-aware)
- `POST /api/games/{id}/react` — AI reaction to a chess move
- `POST /api/chat` — general chess chat (no game context)
- `GET /api/chat/status` — check LLM availability and configured provider
- Conversation tracking per game with message history
- Context-aware prompts: FEN position, move count, game phase, legal moves, check status
- System prompt: friendly chess coach persona with chess terminology
- Response cleaning: bracket artifact removal, prefix stripping, sentence-boundary truncation (280 char max)
- Per-request provider/API key override support
- Auto-detect provider from configured API keys (priority: OpenAI → Anthropic → Gemini → xAI → DeepSeek)
- `LlmConfig` with environment variable configuration for all providers
- Follow-up suggestion generation based on game phase
- 46 new tests (306 total, all passing, zero warnings)

#### Phase 8: Enhanced AI
- Iterative deepening with time-limited search
- Transposition table (1M entries, Zobrist-keyed, depth-preferred replacement)
- Quiescence search with standing pat and delta pruning
- Killer move heuristic (2 slots per ply)
- History heuristic table with aging between iterations
- Null move pruning (R=2, skipped in check/pawn-only endgames)
- Late move reduction for quiet moves at depth ≥ 3
- Enhanced move ordering: TT move → captures (MVV-LVA) → promotions → killers → history
- `NullMoveUndo` and `make_null_move`/`undo_null_move` on Position
- `legal_captures()` generator for quiescence search

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
# Contributing to rust-chess

Thank you for your interest in contributing to rust-chess! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** and clone it locally
2. **Install Rust**: [rustup.rs](https://rustup.rs/) (stable, 1.82+)
3. **Create a feature branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes** following our coding standards
5. **Test your changes**: `cargo test`
6. **Submit a pull request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rust-chess.git
cd rust-chess

# Build the project
cargo build

# Run tests
cargo test

# Run the server
cargo run

# Build release binary
cargo build --release
```

## Development Workflow

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run a specific test module
cargo test engine::board

# Run perft integration tests
cargo test --test perft

# Run tests in release mode
cargo test --release
```

### Linting and Formatting

```bash
# Format code
cargo fmt --all

# Check formatting (CI uses this)
cargo fmt --all -- --check

# Run clippy linter (CI uses -D warnings)
cargo clippy --all-targets --all-features -- -D warnings
```

### Coverage

```bash
# Generate coverage report (needs cargo-tarpaulin)
cargo tarpaulin --out html --output-dir coverage/
```

### Security Audit

```bash
# Audit dependencies (needs cargo-audit)
cargo audit
```

## Pull Request Process

1. **Update documentation** for any user-facing changes
2. **Add tests** for new features or bug fixes
3. **Ensure all tests pass** (`cargo test`)
4. **Ensure code is properly formatted** (`cargo fmt --all -- --check`)
5. **Ensure clippy passes** (`cargo clippy --all-targets --all-features -- -D warnings`)
6. **Update CHANGELOG.md** with a description of your changes
7. **Request review** from maintainers

### PR Requirements

- ✅ All tests must pass (debug and release)
- ✅ Code coverage must not decrease (maintain >85%)
- ✅ Clippy must pass with `-D warnings`
- ✅ Code must be formatted with `cargo fmt`
- ✅ Documentation must be updated
- ✅ Commit messages must follow our guidelines

## Coding Standards

### Rust

- Follow **idiomatic Rust** patterns
- Use `#[must_use]` on functions where ignoring the return value is likely a bug
- Prefer **enums** over boolean flags for clarity
- Use `thiserror` for error types; implement `Display` and `Error`
- Document public APIs with `///` doc comments
- Avoid `unsafe` unless absolutely necessary (and document why)

### File Structure

```
src/
├── engine/       # Core chess engine (types, board, movegen, game, san, pgn)
├── ai/           # AI engine (evaluation, search)
├── api/          # REST API (router, handlers, models, errors, state)
├── config.rs     # Configuration
├── lib.rs        # Crate root
└── main.rs       # Binary entry point
```

### Naming Conventions

- **Files**: `snake_case.rs`
- **Structs/Enums**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Modules**: `snake_case`

### Code Style

- Use **rustfmt** for formatting (default configuration)
- Follow **clippy** recommendations
- Maximum line length: handled by rustfmt
- Use **explicit types** where inference is ambiguous
- Prefer **iterators** over manual loops where idiomatic

## Testing Guidelines

### Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_does_something_specific() {
        // Arrange
        let input = ...;

        // Act
        let result = ...;

        // Assert
        assert_eq!(result, expected);
    }
}
```

### Coverage Requirements

- **Overall coverage**: >85%
- **Engine module**: >90% coverage required
- **AI module**: >80% coverage
- **API module**: >80% coverage

### Test Types

1. **Unit Tests**: Test individual functions/methods in isolation (`#[cfg(test)]` modules)
2. **Integration Tests**: Test module interactions (`tests/` directory)
3. **Perft Tests**: Chess move generation correctness validation
4. **API Tests**: Test REST API endpoints with mock state

### What to Test

- ✅ All public APIs
- ✅ Edge cases and error conditions
- ✅ Chess rules compliance
- ✅ Move validation and generation
- ✅ Game state transitions
- ✅ AI move generation
- ✅ FEN/PGN parsing and generation
- ✅ API request/response shapes

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks
- **ci**: CI/CD changes

### Examples

```
feat(engine): add support for Chess960

fix(ai): correct minimax evaluation for endgame

docs(readme): update installation instructions

test(movegen): add perft tests for en passant edge cases
```

## Bug Reports

When reporting bugs, please include:

1. **Description** of the issue
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment** (Rust version, OS, etc.)
6. **Code sample** or test case (if applicable)

## Feature Requests

When requesting features, please include:

1. **Use case** description
2. **Proposed solution**
3. **Alternatives considered**
4. **Additional context**

## Questions?

If you have questions:

- Open an issue with the `question` label
- Contact: contact@rumenx.com
- Check existing documentation in `/wiki` and `/.ai`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to rust-chess! 🚀♟️

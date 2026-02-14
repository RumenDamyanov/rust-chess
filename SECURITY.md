# Security Policy

## Reporting a Vulnerability

The rust-chess team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:

**security@rumenx.com**

### What to Include

Please include the following information in your report:

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Potential impact** of the vulnerability
- **Suggested fix** (if you have one)
- **Your contact information** for follow-up questions

### Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix timeline**: Depends on severity (see below)

### Severity Levels

- **Critical**: Fix within 24-48 hours
- **High**: Fix within 7 days
- **Medium**: Fix within 30 days
- **Low**: Fix in next regular release

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Best Practices

When using rust-chess:

1. **Keep dependencies updated**: Run `cargo audit` regularly
2. **Validate user input**: Always validate moves and game states
3. **Rate limiting**: Implement rate limiting for API endpoints
4. **Authentication**: Use proper authentication for API access
5. **Environment variables**: Store sensitive data in environment variables
6. **HTTPS only**: Use HTTPS for API communication in production

## Known Security Considerations

### AI Move Generation

- AI move generation can be CPU-intensive at higher depths
- Implement timeouts to prevent DoS attacks (built-in via `CHESS_AI_TIMEOUT`)
- Consider rate limiting AI requests

### Game State Validation

- Always validate game states before processing
- Sanitize FEN/PGN input from untrusted sources
- Validate move legality server-side (enforced by the engine)

### API Security

- Implement authentication for sensitive endpoints
- Use rate limiting to prevent abuse
- Validate all input parameters
- Sanitize error messages (structured errors, no stack traces)

### Docker Security

- The `scratch`-based image has zero attack surface (no shell, no OS)
- No unnecessary packages or utilities included
- Static binary with musl — no dynamic loading vulnerabilities

## Dependencies

We regularly monitor and update dependencies to address security vulnerabilities:

- **Automated**: Dependabot alerts and updates
- **Manual**: Monthly dependency review
- **Audit**: Run `cargo audit` in CI/CD pipeline

## Security Updates

Security updates will be:

- Released as patch versions (e.g., 0.1.1)
- Documented in CHANGELOG.md
- Announced via GitHub releases

## Acknowledgments

We will acknowledge security researchers who responsibly disclose vulnerabilities:

- Listed in CHANGELOG.md
- Credit in release notes
- Public acknowledgment (with permission)

## Contact

For security concerns:

- **Email**: security@rumenx.com
- **General contact**: contact@rumenx.com
- **GitHub**: https://github.com/RumenDamyanov/rust-chess/issues

---

Thank you for helping keep rust-chess secure! 🔒

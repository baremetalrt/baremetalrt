# Security Policy

## Supported Versions

| Version        | Supported          |
| -------------- | ------------------ |
| 0.4.x (beta)   | :white_check_mark: |
| < 0.4.0        | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in BareMetalRT, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@baremetalrt.ai**

### What to include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact
- Any suggested fix (if you have one)

### What to expect

- **Acknowledgment** within 48 hours
- **Status update** within 7 days
- We will work with you to understand and resolve the issue before any public disclosure

### Scope

The following are in scope for security reports:

- The BareMetalRT daemon and coordinator
- The web application at baremetalrt.ai
- The TCP transport layer
- Authentication and authorization mechanisms
- The Windows installer

### Out of Scope

- The upstream TensorRT-LLM submodule (report to NVIDIA directly)
- Third-party dependencies (report to the respective maintainers)
- Social engineering attacks

## Security Best Practices for Users

- Keep your BareMetalRT installation up to date
- Do not expose the daemon port to the public internet without proper firewall rules
- Use the authentication system — do not run in open/unauthenticated mode in production

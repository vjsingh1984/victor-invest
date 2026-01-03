# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please email security concerns to: **security@example.com**

Include the following information:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt within 48 hours
2. **Assessment**: We will assess the vulnerability within 7 days
3. **Resolution**: Critical vulnerabilities will be addressed within 30 days
4. **Disclosure**: We will coordinate disclosure timing with you

## Security Best Practices

When using InvestiGator, follow these security guidelines:

### Configuration

1. **Environment Variables**: Store all credentials in environment variables, never in code
   ```bash
   export DATABASE_URL="postgresql://user:password@host/db"
   export FRED_API_KEY="your-api-key"
   ```

2. **Configuration Files**: Never commit `.env` files or files with credentials
   - Use `.env.example` as a template
   - Add sensitive files to `.gitignore`

3. **Database Access**: Use least-privilege database accounts
   - Create read-only accounts for analysis
   - Separate accounts for different environments

### API Keys

- Rotate API keys periodically
- Use separate keys for development and production
- Monitor API key usage for anomalies

### Network Security

- Use TLS/SSL for database connections
- Restrict database access to known IP addresses
- Use VPN or private networks for sensitive data

### Data Handling

- Sanitize user inputs before database queries
- Use parameterized queries (SQLAlchemy handles this)
- Encrypt sensitive data at rest

## Known Limitations

### Financial Disclaimer

InvestiGator is for informational purposes only. It is NOT:
- Financial advice
- A recommendation to buy or sell securities
- A substitute for professional financial guidance

### Data Accuracy

- Financial data may be delayed or incomplete
- SEC filings are processed automatically and may contain parsing errors
- Technical indicators are calculated based on available data

### No Warranty

This software is provided "as is" without warranty of any kind. Users assume all risks associated with its use.

## Security Features

### Built-in Protections

- **SQL Injection Prevention**: All database queries use SQLAlchemy ORM with parameterized queries
- **Input Validation**: User inputs are validated before processing
- **Logging**: Security-relevant events are logged (without sensitive data)

### Audit Logging

Enable audit logging for security monitoring:
```yaml
logging:
  level: INFO
  audit:
    enabled: true
    file: logs/audit.log
```

## Dependency Security

We regularly update dependencies to address security vulnerabilities:

```bash
# Check for known vulnerabilities
pip-audit

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Responsible Disclosure

We appreciate security researchers who:
- Give us reasonable time to address issues before public disclosure
- Make a good-faith effort to avoid privacy violations
- Do not exploit vulnerabilities beyond what's necessary to demonstrate them

## Contact

For security concerns: **security@example.com**

For general questions: Use GitHub Issues or Discussions

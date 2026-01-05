# Credential Management Guide

This guide covers the comprehensive credential management system for InvestiGator, which provides secure handling of database credentials, API keys, and other secrets.

## Overview

The credential management system consists of five modules:

| Module | Purpose |
|--------|---------|
| `credentials.py` | Core credential resolution (database, env vars) |
| `node_credentials.py` | Node-scoped credentials, audit logging, misuse detection |
| `mcp_credentials.py` | MCP server credential handling |
| `credential_rotation.py` | Credential rotation scheduling |
| `credential_sanitizer.py` | Leakage detection and redaction |

## Quick Start

### 1. Set Up Environment

Create `~/.investigator/env`:

```bash
# SEC Database
export SEC_DB_HOST="dataserver1.singh.local"
export SEC_DB_USER="investigator"
export SEC_DB_PASSWORD="your_password"
export SEC_DB_NAME="sec_database"
export SEC_DB_PORT="5432"

# Stock Database
export STOCK_DB_HOST="dataserver1.singh.local"
export STOCK_DB_USER="stockuser"
export STOCK_DB_PASSWORD="your_password"
export STOCK_DB_NAME="stock"
export STOCK_DB_PORT="5432"

# API Keys (optional)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

Source before running:
```bash
source ~/.investigator/env
```

### 2. Use Database Credentials

```python
from investigator.infrastructure.credentials import get_database_credentials

# Get SEC database credentials
creds = get_database_credentials("sec")
print(creds.host, creds.port, creds.database)

# Connection string for SQLAlchemy
conn_str = creds.connection_string
```

### 3. Node Credential Context (In Workflows)

```python
from investigator.infrastructure.node_credentials import NodeCredentialContext

# In a workflow handler
cred_ctx = NodeCredentialContext.from_node(node, context)
db = cred_ctx.get_database("sec")
api_key = cred_ctx.get_api_key("anthropic")
```

## Features

### Unified Credential Resolution

Credentials are resolved through multiple layers:

1. **Victor Framework CredentialManager** (if available)
2. **Environment Variables**
3. **Configuration Files**

```python
from investigator.infrastructure.credentials import (
    get_database_credentials,
    validate_database_connection,
)

# Resolution happens automatically
creds = get_database_credentials("sec")

# Validate connection works
if validate_database_connection("sec"):
    print("Connection successful!")
```

### Workflow YAML Credential Requirements

Define credential requirements in workflow YAML:

```yaml
- id: fetch_sec_data
  type: compute
  handler: fetch_sec_data
  credentials_required:
    - type: database
      name: sec
      required: true
    - type: database
      name: stock
      required: false  # Optional
  constraints:
    timeout: 60
```

### Audit Logging

All credential access is automatically logged:

```python
from investigator.infrastructure.node_credentials import CredentialAuditLogger

# Get recent entries
entries = CredentialAuditLogger.get_entries()
for entry in entries:
    print(f"{entry.timestamp}: {entry.node_id} accessed {entry.credential_name}")

# Get statistics
stats = CredentialAuditLogger.get_statistics()
print(f"Total accesses: {stats['total_accesses']}")
print(f"Success rate: {stats['success_rate']*100:.0f}%")
```

### Misuse Pattern Detection

Detect suspicious credential access patterns:

```python
violations = CredentialAuditLogger.detect_misuse_patterns()
for v in violations:
    print(f"[{v['severity']}] {v['pattern']}: {v['message']}")
```

Patterns detected:
- **Excessive frequency**: >100 accesses in 1 minute
- **High failure rate**: >50% failures for a credential
- **Off-hours access**: Access outside 6am-10pm
- **New accessor**: Node accessing credential for first time

### MCP Server Credentials

Handle credentials for MCP server connections:

```python
from investigator.infrastructure.mcp_credentials import (
    MCPCredentialResolver,
    MCPServerCredentials,
    inject_mcp_credentials,
)

# Define server requirements
server_creds = MCPServerCredentials(
    server_name="investigator",
    required_credentials=["sec_db", "stock_db"],
    optional_credentials=["anthropic"],
    env_mappings={
        "sec_db": "SEC_DB",
        "stock_db": "STOCK_DB",
        "anthropic": "ANTHROPIC_API_KEY",
    },
)

# Inject into server config
resolver = MCPCredentialResolver()
config = inject_mcp_credentials(server_config, server_creds, resolver)
```

### Credential Rotation

Schedule and track credential rotations:

```python
from investigator.infrastructure.credential_rotation import (
    RotationScheduler,
    RotationPolicy,
    get_default_scheduler,
)

# Get scheduler with default policies
scheduler = get_default_scheduler()

# Check pending rotations
pending = scheduler.get_pending_rotations()
for entry in pending:
    print(f"{entry.credential_name}: {entry.days_until_rotation} days until rotation")

# Record a rotation
scheduler.record_rotation("database:sec", rotated_by="admin")

# Add notification callback
def notify(entry):
    print(f"ALERT: {entry.credential_name} needs rotation!")

scheduler.add_notification_callback(notify)
scheduler.check_and_notify()
```

### Credential Leakage Detection

Prevent credential exposure in outputs:

```python
from investigator.infrastructure.credential_sanitizer import (
    scan_for_credentials,
    redact_credentials,
    CredentialSanitizer,
)

# Scan text
findings = scan_for_credentials("api_key = 'sk-test123...'")
if findings:
    print(f"Found {len(findings)} potential credentials!")

# Redact credentials
safe_text = redact_credentials(sensitive_text)

# Scan complex data structures
sanitizer = CredentialSanitizer()
result = sanitizer.scan(workflow_output)
if result.has_credentials:
    print(f"WARNING: {result.summary}")
    safe_output = result.redacted_output
```

## Security Best Practices

### 1. Never Commit Secrets

The `.gitignore` already includes:
- `.env`
- `.env.local`
- `.victor/`
- `config.local.json`

### 2. Use Environment Files

Keep secrets in `~/.investigator/env` (outside repo):
```bash
# Good: File outside repository
~/.investigator/env

# Bad: File in repository
.env  # Even if gitignored, risky
```

### 3. Validate Before Use

```python
from investigator.infrastructure.credentials import validate_database_connection

if not validate_database_connection("sec"):
    raise RuntimeError("Database not accessible!")
```

### 4. Use Node Credential Context

Always use `NodeCredentialContext` in handlers:
```python
# Good: Uses audit logging and validation
cred_ctx = NodeCredentialContext.from_node(node, context)
db = cred_ctx.get_database("sec")

# Avoid: Direct environment access
password = os.environ.get("SEC_DB_PASSWORD")  # No audit trail
```

### 5. Scan Outputs

Before logging or returning data:
```python
sanitizer = CredentialSanitizer(strict_mode=True)
try:
    result = sanitizer.scan(output_data)
    return result.redacted_output
except CredentialLeakageError:
    logger.error("Credential leak prevented!")
    raise
```

### 6. Set Up Rotation

```python
from investigator.infrastructure.credential_rotation import get_default_scheduler
from pathlib import Path

# Persist rotation state
scheduler = get_default_scheduler(
    storage_path=Path("~/.investigator/rotation_state.json").expanduser()
)
```

## API Reference

### credentials.py

| Function | Description |
|----------|-------------|
| `get_database_credentials(alias)` | Get database credentials by alias |
| `validate_database_connection(alias)` | Test database connection |

### node_credentials.py

| Class | Description |
|-------|-------------|
| `NodeCredentialContext` | Scoped credential container for nodes |
| `CredentialAuditLogger` | Audit logging for credential access |
| `CredentialValidator` | Pre-flight credential validation |

### mcp_credentials.py

| Class | Description |
|-------|-------------|
| `MCPCredentialResolver` | Resolve credentials for MCP servers |
| `MCPClientAuth` | Authentication configuration |
| `MCPServerCredentials` | Server credential requirements |

### credential_rotation.py

| Class | Description |
|-------|-------------|
| `RotationScheduler` | Manage rotation schedules |
| `RotationPolicy` | Define rotation rules |

### credential_sanitizer.py

| Function | Description |
|----------|-------------|
| `scan_for_credentials(text)` | Scan text for credentials |
| `redact_credentials(text)` | Redact credentials from text |
| `CredentialSanitizer` | Full-featured scanner |

## Troubleshooting

### "Database credentials not found"

1. Ensure `~/.investigator/env` exists
2. Run `source ~/.investigator/env`
3. Check variable names match expected format

### "Connection failed"

1. Verify host is reachable: `ping dataserver1.singh.local`
2. Check port is open: `nc -zv dataserver1.singh.local 5432`
3. Verify credentials are correct

### "High failure rate detected"

1. Check credential values haven't expired
2. Verify database user has correct permissions
3. Review audit log for details:
   ```python
   entries = CredentialAuditLogger.get_entries(credential_name="sec")
   ```

## Testing

Run credential tests:
```bash
source ~/.investigator/env
PYTHONPATH=./src:. pytest tests/test_credentials.py -v
```

## Files

| File | Location |
|------|----------|
| Core credentials | `src/investigator/infrastructure/credentials.py` |
| Node context | `src/investigator/infrastructure/node_credentials.py` |
| MCP credentials | `src/investigator/infrastructure/mcp_credentials.py` |
| Rotation | `src/investigator/infrastructure/credential_rotation.py` |
| Sanitizer | `src/investigator/infrastructure/credential_sanitizer.py` |
| Tests | `tests/test_credentials.py` |

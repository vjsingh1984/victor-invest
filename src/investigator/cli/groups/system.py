"""
System management commands for InvestiGator CLI
"""

import platform
import subprocess
import sys
from pathlib import Path

import click


@click.group()
@click.pass_context
def system(ctx):
    """System management and diagnostics

    Check system health, run tests, and manage configuration.

    Examples:
        investigator system status
        investigator system test
        investigator system setup
        investigator system info
    """
    pass


@system.command("status")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed status")
@click.pass_context
def status(ctx, verbose):
    """Check system status and health

    Verifies all components are running and accessible.
    """
    import asyncio

    click.echo("InvestiGator System Status")
    click.echo("=" * 60)

    checks = []

    # Check Python
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python", True, py_version))

    # Check Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            checks.append(("Ollama", True, f"{len(models)} models"))
            if verbose:
                for m in models[:5]:
                    click.echo(f"    - {m['name']}")
        else:
            checks.append(("Ollama", False, "Not responding"))
    except Exception as e:
        checks.append(("Ollama", False, str(e)[:30]))

    # Check Database
    try:
        from investigator.infrastructure.database.db import get_database_engine
        engine = get_database_engine()
        with engine.connect() as conn:
            pass
        checks.append(("Database", True, "Connected"))
    except Exception as e:
        checks.append(("Database", False, str(e)[:30]))

    # Check Cache
    try:
        from investigator.infrastructure.cache import get_cache_manager
        cache = get_cache_manager()
        checks.append(("Cache", True, "Initialized"))
    except Exception as e:
        checks.append(("Cache", False, str(e)[:30]))

    # Check Redis (optional)
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_timeout=2)
        r.ping()
        checks.append(("Redis", True, "Connected"))
    except Exception:
        checks.append(("Redis", False, "Not available"))

    # Display results
    click.echo("\nComponent Status:")
    click.echo("-" * 40)

    all_ok = True
    for name, ok, detail in checks:
        status = click.style("OK", fg="green") if ok else click.style("FAIL", fg="red")
        click.echo(f"  {name:15s}: [{status}] {detail}")
        if not ok:
            all_ok = False

    click.echo("\n" + "=" * 60)
    if all_ok:
        click.echo(click.style("All systems operational", fg="green"))
    else:
        click.echo(click.style("Some components unavailable", fg="yellow"))


@system.command("test")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--pattern", "-k", help="Test pattern to match")
@click.option("--coverage", is_flag=True, help="Run with coverage")
@click.pass_context
def test(ctx, verbose, pattern, coverage):
    """Run system health tests

    Runs the test suite to verify system functionality.

    Examples:
        investigator system test
        investigator system test --pattern test_cache
        investigator system test --coverage
    """
    click.echo("Running tests...")
    click.echo("=" * 60)

    cmd = ["python3", "-m", "pytest", "tests/"]

    if verbose:
        cmd.append("-v")
    if pattern:
        cmd.extend(["-k", pattern])
    if coverage:
        cmd.extend(["--cov=investigator", "--cov-report=term-missing"])

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


@system.command("setup")
@click.option("--skip-deps", is_flag=True, help="Skip dependency installation")
@click.option("--skip-db", is_flag=True, help="Skip database setup")
@click.pass_context
def setup(ctx, skip_deps, skip_db):
    """Setup system dependencies and configuration

    Initializes directories, installs dependencies, and configures the system.
    """
    click.echo("Setting up InvestiGator...")
    click.echo("=" * 60)

    steps = []

    # 1. Check Python
    click.echo("\n1. Checking Python version...")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 11):
        click.echo(f"   Python {py_version}")
        steps.append(("Python", True))
    else:
        click.echo(f"   Python {py_version} (requires 3.11+)", err=True)
        steps.append(("Python", False))

    # 2. Install dependencies
    if not skip_deps:
        click.echo("\n2. Installing dependencies...")
        result = subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            capture_output=True
        )
        if result.returncode == 0:
            click.echo("   Dependencies installed")
            steps.append(("Dependencies", True))
        else:
            click.echo("   Failed to install dependencies", err=True)
            steps.append(("Dependencies", False))
    else:
        click.echo("\n2. Skipping dependencies...")

    # 3. Create directories
    click.echo("\n3. Creating directories...")
    dirs = [
        "data/sec_cache",
        "data/llm_cache",
        "data/technical_cache",
        "data/vector_db",
        "logs",
        "reports",
        "results",
        "models",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    click.echo(f"   Created {len(dirs)} directories")
    steps.append(("Directories", True))

    # 4. Database setup
    if not skip_db:
        click.echo("\n4. Setting up database...")
        try:
            from investigator.infrastructure.database.db import get_engine
            from sqlalchemy import text

            engine = get_engine()

            schema_file = Path("schema/consolidated_schema.sql")
            if schema_file.exists():
                with open(schema_file) as f:
                    schema_sql = f.read()

                with engine.connect() as conn:
                    for statement in schema_sql.split(";"):
                        if statement.strip():
                            try:
                                conn.execute(text(statement))
                            except Exception:
                                pass  # Table may already exist
                    conn.commit()

                click.echo("   Database schema initialized")
                steps.append(("Database", True))
            else:
                click.echo("   Schema file not found")
                steps.append(("Database", False))
        except Exception as e:
            click.echo(f"   Database setup failed: {e}", err=True)
            steps.append(("Database", False))
    else:
        click.echo("\n4. Skipping database setup...")

    # 5. Verify Ollama
    click.echo("\n5. Checking Ollama...")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            click.echo("   Ollama is running")
            steps.append(("Ollama", True))
        else:
            click.echo("   Ollama not responding")
            steps.append(("Ollama", False))
    except Exception:
        click.echo("   Ollama not available (optional)")
        steps.append(("Ollama", False))

    # Summary
    click.echo("\n" + "=" * 60)
    click.echo("Setup Summary:")
    for name, ok in steps:
        status = click.style("OK", fg="green") if ok else click.style("FAIL", fg="red")
        click.echo(f"  [{status}] {name}")

    required = ["Python", "Directories"]
    if all(ok for name, ok in steps if name in required):
        click.echo(click.style("\nSetup complete!", fg="green"))
    else:
        click.echo(click.style("\nSetup incomplete - check errors above", fg="yellow"))
        sys.exit(1)


@system.command("info")
@click.pass_context
def info(ctx):
    """Show system information

    Displays detailed system and resource information.
    """
    import psutil

    click.echo("InvestiGator System Information")
    click.echo("=" * 60)

    # System
    click.echo("\nSystem:")
    click.echo(f"  Platform:     {platform.system()} {platform.release()}")
    click.echo(f"  Architecture: {platform.machine()}")
    click.echo(f"  Python:       {platform.python_version()}")
    click.echo(f"  Node:         {platform.node()}")

    # Resources
    click.echo("\nResources:")
    click.echo(f"  CPU Cores:    {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    click.echo(f"  CPU Usage:    {psutil.cpu_percent()}%")

    memory = psutil.virtual_memory()
    click.echo(f"  Memory:       {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent}%)")

    disk = psutil.disk_usage(".")
    click.echo(f"  Disk:         {disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB ({disk.percent}%)")

    # Package versions
    click.echo("\nKey Packages:")
    packages = ["click", "pandas", "sqlalchemy", "pydantic", "fastapi"]
    for pkg in packages:
        try:
            import importlib.metadata
            version = importlib.metadata.version(pkg)
            click.echo(f"  {pkg:15s}: {version}")
        except Exception:
            click.echo(f"  {pkg:15s}: not installed")

    # Investigator info
    click.echo("\nInvestiGator:")
    try:
        import importlib.metadata
        version = importlib.metadata.version("investigator")
        click.echo(f"  Version:      {version}")
    except Exception:
        click.echo(f"  Version:      development")

    click.echo(f"  Working Dir:  {Path.cwd()}")
    click.echo(f"  Config:       config.yaml")


@system.command("config")
@click.option("--edit", is_flag=True, help="Open config in editor")
@click.option("--validate", is_flag=True, help="Validate config file")
@click.pass_context
def config(ctx, edit, validate):
    """View or edit configuration

    Displays current configuration or opens it for editing.
    """
    import yaml

    config_path = Path("config.yaml")

    if edit:
        import os
        editor = os.environ.get("EDITOR", "vim")
        subprocess.run([editor, str(config_path)])
        return

    if not config_path.exists():
        click.echo("Config file not found. Run 'investigator system setup' first.")
        return

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    if validate:
        click.echo("Validating configuration...")
        errors = []

        # Check required sections
        required = ["ollama", "cache", "orchestrator"]
        for section in required:
            if section not in config_data:
                errors.append(f"Missing section: {section}")

        if errors:
            click.echo(click.style("Validation failed:", fg="red"))
            for e in errors:
                click.echo(f"  - {e}")
            sys.exit(1)
        else:
            click.echo(click.style("Configuration is valid", fg="green"))
        return

    # Display config
    click.echo("Current Configuration")
    click.echo("=" * 60)
    click.echo(yaml.dump(config_data, default_flow_style=False))


@system.command("logs")
@click.option("--lines", "-n", default=50, type=int, help="Number of lines to show")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Filter by level")
@click.pass_context
def logs(ctx, lines, follow, level):
    """View system logs

    Displays recent log entries from the application.
    """
    log_dir = Path("logs")

    if not log_dir.exists():
        click.echo("No logs directory found")
        return

    # Find latest log file
    log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
        click.echo("No log files found")
        return

    latest = log_files[0]
    click.echo(f"Showing: {latest}")
    click.echo("=" * 60)

    if follow:
        cmd = ["tail", "-f", str(latest)]
        if level:
            cmd = f"tail -f {latest} | grep {level}"
            subprocess.run(cmd, shell=True)
        else:
            subprocess.run(cmd)
    else:
        with open(latest) as f:
            all_lines = f.readlines()
            if level:
                all_lines = [l for l in all_lines if level in l]
            for line in all_lines[-lines:]:
                click.echo(line.rstrip())


@system.command("metrics")
@click.option("--days", "-d", default=7, type=int, help="Days of history")
@click.pass_context
def metrics(ctx, days):
    """View system metrics

    Displays performance metrics and statistics.
    """
    import json
    import glob
    from datetime import datetime, timedelta

    click.echo(f"System Metrics (last {days} days)")
    click.echo("=" * 60)

    cutoff = datetime.now() - timedelta(days=days)
    metrics_files = glob.glob("metrics/metrics_*.json")

    all_metrics = []
    for filepath in sorted(metrics_files):
        try:
            with open(filepath) as f:
                data = json.load(f)
                ts = datetime.fromisoformat(data.get("timestamp", ""))
                if ts >= cutoff:
                    all_metrics.append(data)
        except Exception:
            continue

    if not all_metrics:
        click.echo("No metrics data available")
        return

    latest = all_metrics[-1]

    if "system_metrics" in latest:
        sm = latest["system_metrics"]
        click.echo("\nSystem Metrics:")
        click.echo(f"  Total Analyses:    {sm.get('total_analyses', 0)}")
        total = sm.get('total_analyses', 1) or 1
        success = sm.get('successful_analyses', 0)
        click.echo(f"  Success Rate:      {(success / total * 100):.1f}%")

        hits = sm.get('cache_hits', 0)
        misses = sm.get('cache_misses', 0)
        total_cache = hits + misses
        if total_cache > 0:
            click.echo(f"  Cache Hit Rate:    {(hits / total_cache * 100):.1f}%")

    if "agent_metrics" in latest:
        click.echo("\nAgent Performance:")
        for agent, metrics in latest["agent_metrics"].items():
            execs = metrics.get('executions', 0)
            avg_dur = metrics.get('average_duration', 0)
            fails = metrics.get('failures', 0)
            success_rate = ((execs - fails) / execs * 100) if execs > 0 else 0
            click.echo(f"  {agent}:")
            click.echo(f"    Executions: {execs}, Avg: {avg_dur:.1f}s, Success: {success_rate:.0f}%")

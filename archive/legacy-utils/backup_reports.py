#!/usr/bin/env python3
"""
InvestiGator - Report Backup Utility
Copyright (c) 2025 Vijaykumar Singh
Licensed under the Apache License 2.0

Optional utility to backup peer group reports with timestamps
Use this if you want to keep historical versions
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import argparse


def backup_peer_group_reports(source_dir: str = "reports", backup_dir: str = "reports/backups"):
    """
    Backup current peer group reports with timestamp

    Args:
        source_dir: Source directory containing reports
        backup_dir: Backup directory to copy files to
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(backup_dir) / f"backup_{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_dir)

    # Files to backup (without timestamps)
    files_to_backup = []

    # Peer group PDF reports
    pdf_dir = source_path / "peer_group_pdfs"
    if pdf_dir.exists():
        for pdf_file in pdf_dir.glob("peer_group_*.pdf"):
            # Only backup files without timestamps (clean versions)
            if not any(char.isdigit() for char in pdf_file.stem.split("_")[-1]):
                files_to_backup.append(pdf_file)

    # Peer group JSON/MD reports
    for report_type in ["peer_group", "peer_group_comprehensive"]:
        report_dir = source_path / report_type
        if report_dir.exists():
            for ext in ["*.json", "*_summary.md"]:
                for report_file in report_dir.glob(ext):
                    # Only backup files without timestamps
                    if not any(char.isdigit() for char in report_file.stem.split("_")[-1]):
                        files_to_backup.append(report_file)

    # Charts
    chart_dir = source_path / "charts"
    if chart_dir.exists():
        for chart_file in chart_dir.glob("*.png"):
            files_to_backup.append(chart_file)

    # Copy files to backup directory
    backed_up = 0
    for file_path in files_to_backup:
        try:
            # Maintain directory structure in backup
            relative_path = file_path.relative_to(source_path)
            backup_file_path = backup_path / relative_path
            backup_file_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(file_path, backup_file_path)
            backed_up += 1
            print(f"✅ Backed up: {relative_path}")

        except Exception as e:
            print(f"❌ Failed to backup {file_path}: {e}")

    print(f"\n🎉 Backup complete!")
    print(f"📁 Backup location: {backup_path}")
    print(f"📄 Files backed up: {backed_up}")

    return str(backup_path)


def list_backups(backup_dir: str = "reports/backups"):
    """List available backups"""
    backup_path = Path(backup_dir)

    if not backup_path.exists():
        print("No backups found.")
        return

    backups = list(backup_path.glob("backup_*"))
    if not backups:
        print("No backups found.")
        return

    print("📋 Available backups:")
    for backup in sorted(backups, reverse=True):
        # Parse timestamp from folder name
        timestamp_str = backup.name.replace("backup_", "")
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Count files in backup
            file_count = sum(1 for _ in backup.rglob("*") if _.is_file())

            print(f"  📅 {formatted_time} - {file_count} files - {backup}")
        except ValueError:
            print(f"  📁 {backup}")


def restore_backup(backup_name: str, backup_dir: str = "reports/backups", target_dir: str = "reports"):
    """
    Restore a specific backup

    Args:
        backup_name: Name of backup folder (e.g., backup_20250603_185000)
        backup_dir: Directory containing backups
        target_dir: Directory to restore files to
    """
    backup_path = Path(backup_dir) / backup_name
    target_path = Path(target_dir)

    if not backup_path.exists():
        print(f"❌ Backup not found: {backup_path}")
        return False

    print(f"🔄 Restoring backup from {backup_path}...")

    restored = 0
    for file_path in backup_path.rglob("*"):
        if file_path.is_file():
            try:
                # Calculate relative path and target location
                relative_path = file_path.relative_to(backup_path)
                target_file_path = target_path / relative_path
                target_file_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(file_path, target_file_path)
                restored += 1
                print(f"✅ Restored: {relative_path}")

            except Exception as e:
                print(f"❌ Failed to restore {file_path}: {e}")

    print(f"\n🎉 Restore complete!")
    print(f"📄 Files restored: {restored}")
    return True


def main():
    """Main function for backup utility"""
    parser = argparse.ArgumentParser(description="InvestiGator Report Backup Utility")
    parser.add_argument("action", choices=["backup", "list", "restore"], help="Action to perform")
    parser.add_argument("--backup-name", help="Backup name for restore action")
    parser.add_argument("--source-dir", default="reports", help="Source directory for backup (default: reports)")
    parser.add_argument("--backup-dir", default="reports/backups", help="Backup directory (default: reports/backups)")
    parser.add_argument("--target-dir", default="reports", help="Target directory for restore (default: reports)")

    args = parser.parse_args()

    if args.action == "backup":
        backup_peer_group_reports(args.source_dir, args.backup_dir)
    elif args.action == "list":
        list_backups(args.backup_dir)
    elif args.action == "restore":
        if not args.backup_name:
            print("❌ --backup-name required for restore action")
            return
        restore_backup(args.backup_name, args.backup_dir, args.target_dir)


if __name__ == "__main__":
    main()

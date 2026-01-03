#!/usr/bin/env python3
"""
RL Policy Deployment Script

Deploy trained RL policy for use in valuation pipeline.
This script validates the policy and activates it for production use.

Supports both:
- Single Policy: ContextualBanditPolicy (legacy)
- Dual Policy: TechnicalRLPolicy + FundamentalRLPolicy (recommended)

Usage:
    python scripts/rl_deploy.py                  # Deploy dual policy (default)
    python scripts/rl_deploy.py --validate       # Validate before deploy
    python scripts/rl_deploy.py --rollback       # Rollback to previous
    python scripts/rl_deploy.py --status         # Show deployment status

Environment:
    PYTHONPATH=./src:. python scripts/rl_deploy.py
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, date
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from investigator.domain.services.rl.policy.contextual_bandit import ContextualBanditPolicy
from investigator.domain.services.rl.policy import DualRLPolicy, load_dual_policy
from investigator.domain.services.rl.feature_normalizer import FeatureNormalizer
from investigator.domain.services.rl.models import ValuationContext, GrowthStage, CompanySize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path("data/rl_models")
# Single policy (legacy)
POLICY_PATH = MODEL_DIR / "policy.pkl"
NORMALIZER_PATH = MODEL_DIR / "normalizer.pkl"
ACTIVE_POLICY_PATH = MODEL_DIR / "active_policy.pkl"
ACTIVE_NORMALIZER_PATH = MODEL_DIR / "active_normalizer.pkl"
# Dual policy (recommended)
TECHNICAL_POLICY_PATH = MODEL_DIR / "technical_policy.pkl"
FUNDAMENTAL_POLICY_PATH = MODEL_DIR / "fundamental_policy.pkl"
ACTIVE_TECHNICAL_PATH = MODEL_DIR / "active_technical_policy.pkl"
ACTIVE_FUNDAMENTAL_PATH = MODEL_DIR / "active_fundamental_policy.pkl"
# Common
BACKUP_DIR = MODEL_DIR / "backups"
DEPLOYMENT_LOG_PATH = MODEL_DIR / "deployment_log.json"


def load_policy(policy_path: Path, normalizer_path: Path) -> tuple:
    """Load legacy single policy and normalizer from files."""
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Normalizer not found: {normalizer_path}")

    normalizer = FeatureNormalizer()
    normalizer.load(str(normalizer_path))

    policy = ContextualBanditPolicy(normalizer=normalizer)
    policy.load(str(policy_path))

    return policy, normalizer


def load_dual_policy_files() -> DualRLPolicy:
    """Load dual policy from trained files."""
    if not TECHNICAL_POLICY_PATH.exists():
        raise FileNotFoundError(f"Technical policy not found: {TECHNICAL_POLICY_PATH}")
    if not FUNDAMENTAL_POLICY_PATH.exists():
        raise FileNotFoundError(f"Fundamental policy not found: {FUNDAMENTAL_POLICY_PATH}")

    policy = load_dual_policy(
        technical_path=str(TECHNICAL_POLICY_PATH),
        fundamental_path=str(FUNDAMENTAL_POLICY_PATH),
    )
    return policy


def validate_policy(policy) -> dict:
    """Validate policy with test predictions."""
    test_cases = [
        {
            "name": "Tech Growth",
            "context": ValuationContext(
                symbol="TEST",
                analysis_date=date.today(),
                sector="Technology",
                industry="Software",
                growth_stage=GrowthStage.HIGH_GROWTH,
                company_size=CompanySize.LARGE_CAP,
                profitability_score=0.8,
                pe_level=0.7,
                revenue_growth=0.3,
                fcf_margin=0.2,
            ),
        },
        {
            "name": "Dividend Value",
            "context": ValuationContext(
                symbol="TEST",
                analysis_date=date.today(),
                sector="Utilities",
                industry="Electric",
                growth_stage=GrowthStage.DIVIDEND_PAYING,
                company_size=CompanySize.LARGE_CAP,
                profitability_score=0.6,
                pe_level=0.3,
                revenue_growth=0.02,
                payout_ratio=0.6,
            ),
        },
        {
            "name": "Financial",
            "context": ValuationContext(
                symbol="TEST",
                analysis_date=date.today(),
                sector="Financials",
                industry="Banks",
                growth_stage=GrowthStage.MATURE,
                company_size=CompanySize.MEGA_CAP,
                profitability_score=0.7,
                debt_to_equity=2.0,
            ),
        },
    ]

    results = {"valid": True, "predictions": [], "errors": []}

    for case in test_cases:
        try:
            weights = policy.predict(case["context"])

            # Validate weights sum to 100
            total = sum(weights.values())
            if abs(total - 100) > 1:
                results["errors"].append(f"{case['name']}: weights sum to {total}, not 100")
                results["valid"] = False

            # Validate all weights are non-negative
            for model, weight in weights.items():
                if weight < 0:
                    results["errors"].append(f"{case['name']}: negative weight for {model}")
                    results["valid"] = False

            results["predictions"].append(
                {
                    "name": case["name"],
                    "weights": weights,
                    "valid": True,
                }
            )

        except Exception as e:
            results["errors"].append(f"{case['name']}: {str(e)}")
            results["valid"] = False

    return results


def validate_dual_policy(policy: DualRLPolicy) -> dict:
    """Validate dual policy with test predictions."""
    test_cases = [
        {
            "name": "Tech Growth",
            "context": ValuationContext(
                symbol="TEST",
                analysis_date=date.today(),
                sector="Technology",
                industry="Software",
                growth_stage=GrowthStage.HIGH_GROWTH,
                company_size=CompanySize.LARGE_CAP,
                profitability_score=0.8,
                pe_level=0.7,
                revenue_growth=0.3,
                fcf_margin=0.2,
                rsi_14=55.0,
                macd_histogram=0.02,
                adx_14=30.0,
                valuation_gap=0.15,
            ),
        },
        {
            "name": "Dividend Value",
            "context": ValuationContext(
                symbol="TEST",
                analysis_date=date.today(),
                sector="Utilities",
                industry="Electric",
                growth_stage=GrowthStage.DIVIDEND_PAYING,
                company_size=CompanySize.LARGE_CAP,
                profitability_score=0.6,
                pe_level=0.3,
                revenue_growth=0.02,
                payout_ratio=0.6,
                rsi_14=45.0,
                macd_histogram=-0.01,
                adx_14=20.0,
                valuation_gap=0.05,
            ),
        },
        {
            "name": "Financial (Short Signal)",
            "context": ValuationContext(
                symbol="TEST",
                analysis_date=date.today(),
                sector="Financials",
                industry="Banks",
                growth_stage=GrowthStage.MATURE,
                company_size=CompanySize.MEGA_CAP,
                profitability_score=0.7,
                debt_to_equity=2.0,
                rsi_14=75.0,
                macd_histogram=-0.03,
                adx_14=35.0,
                valuation_gap=-0.20,
            ),
        },
    ]

    results = {"valid": True, "predictions": [], "errors": []}

    for case in test_cases:
        try:
            # Get full prediction from dual policy
            prediction = policy.predict_full(case["context"])

            position = prediction.get("position")
            confidence = prediction.get("position_confidence", 0)
            weights = prediction.get("weights", {})
            holding = prediction.get("holding_period", "N/A")

            # Validate position is valid
            if position not in [-1, 0, 1]:
                results["errors"].append(f"{case['name']}: invalid position {position}")
                results["valid"] = False

            # Validate weights sum to ~100
            total = sum(weights.values())
            if abs(total - 100) > 1:
                results["errors"].append(f"{case['name']}: weights sum to {total}, not 100")
                results["valid"] = False

            # Validate all weights are non-negative
            for model, weight in weights.items():
                if weight < 0:
                    results["errors"].append(f"{case['name']}: negative weight for {model}")
                    results["valid"] = False

            pos_label = {1: "LONG", -1: "SHORT", 0: "SKIP"}.get(position, "?")
            results["predictions"].append({
                "name": case["name"],
                "position": pos_label,
                "confidence": f"{confidence:.1%}",
                "weights": weights,
                "holding_period": holding,
                "valid": True,
            })

        except Exception as e:
            results["errors"].append(f"{case['name']}: {str(e)}")
            results["valid"] = False

    return results


def backup_current_policy():
    """Backup current active policy (legacy single policy)."""
    if not ACTIVE_POLICY_PATH.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_policy = BACKUP_DIR / f"policy_{timestamp}.pkl"
    backup_normalizer = BACKUP_DIR / f"normalizer_{timestamp}.pkl"

    shutil.copy(ACTIVE_POLICY_PATH, backup_policy)
    if ACTIVE_NORMALIZER_PATH.exists():
        shutil.copy(ACTIVE_NORMALIZER_PATH, backup_normalizer)

    logger.info(f"Backed up current policy to {backup_policy}")
    return timestamp


def backup_current_dual_policy():
    """Backup current active dual policy."""
    if not ACTIVE_TECHNICAL_PATH.exists() and not ACTIVE_FUNDAMENTAL_PATH.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if ACTIVE_TECHNICAL_PATH.exists():
        backup_tech = BACKUP_DIR / f"technical_policy_{timestamp}.pkl"
        shutil.copy(ACTIVE_TECHNICAL_PATH, backup_tech)

    if ACTIVE_FUNDAMENTAL_PATH.exists():
        backup_fund = BACKUP_DIR / f"fundamental_policy_{timestamp}.pkl"
        shutil.copy(ACTIVE_FUNDAMENTAL_PATH, backup_fund)

    logger.info(f"Backed up current dual policy with timestamp {timestamp}")
    return timestamp


def deploy_policy(skip_validation: bool = False) -> bool:
    """Deploy trained policy to active location."""
    print("\n" + "=" * 70)
    print("DEPLOYING RL POLICY")
    print("=" * 70)

    # Load new policy
    print("\n1. Loading trained policy...")
    try:
        policy, normalizer = load_policy(POLICY_PATH, NORMALIZER_PATH)
        print(f"   Loaded: {policy.name} v{policy.version}")
        print(f"   Updates: {policy._update_count}")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        return False

    # Validate
    if not skip_validation:
        print("\n2. Validating policy...")
        validation = validate_policy(policy)
        if validation["valid"]:
            print("   Validation PASSED")
            for pred in validation["predictions"]:
                print(f"   - {pred['name']}: OK")
        else:
            print("   Validation FAILED:")
            for error in validation["errors"]:
                print(f"   - {error}")
            return False
    else:
        print("\n2. Skipping validation (--skip-validation)")

    # Backup current
    print("\n3. Backing up current policy...")
    backup_ts = backup_current_policy()
    if backup_ts:
        print(f"   Backed up to: backups/*_{backup_ts}.pkl")
    else:
        print("   No existing policy to backup")

    # Deploy
    print("\n4. Deploying new policy...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(POLICY_PATH, ACTIVE_POLICY_PATH)
    shutil.copy(NORMALIZER_PATH, ACTIVE_NORMALIZER_PATH)
    print(f"   Deployed to: {ACTIVE_POLICY_PATH}")

    # Update deployment log
    deployment_log = {
        "deployment_date": datetime.now().isoformat(),
        "policy_version": policy.version,
        "update_count": policy._update_count,
        "backup_timestamp": backup_ts,
        "validation_passed": True,
    }

    with open(DEPLOYMENT_LOG_PATH, "w") as f:
        json.dump(deployment_log, f, indent=2)

    print("\n" + "=" * 70)
    print("DEPLOYMENT COMPLETE")
    print("=" * 70)
    return True


def deploy_dual_policy(skip_validation: bool = False) -> bool:
    """Deploy trained dual policy (technical + fundamental) to active location."""
    print("\n" + "=" * 70)
    print("DEPLOYING DUAL RL POLICY (Technical + Fundamental)")
    print("=" * 70)

    # Load new dual policy
    print("\n1. Loading trained dual policy...")
    try:
        policy = load_dual_policy_files()
        tech_updates = policy.technical._update_count
        fund_updates = policy.fundamental._update_count
        print(f"   Technical Policy: {tech_updates:,} updates")
        print(f"   Fundamental Policy: {fund_updates:,} updates")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        return False

    # Validate
    if not skip_validation:
        print("\n2. Validating dual policy...")
        validation = validate_dual_policy(policy)
        if validation["valid"]:
            print("   Validation PASSED")
            for pred in validation["predictions"]:
                print(f"   - {pred['name']}: {pred['position']} ({pred['confidence']}), hold={pred['holding_period']}")
        else:
            print("   Validation FAILED:")
            for error in validation["errors"]:
                print(f"   - {error}")
            return False
    else:
        print("\n2. Skipping validation (--skip-validation)")

    # Backup current
    print("\n3. Backing up current dual policy...")
    backup_ts = backup_current_dual_policy()
    if backup_ts:
        print(f"   Backed up to: backups/*_{backup_ts}.pkl")
    else:
        print("   No existing dual policy to backup")

    # Deploy
    print("\n4. Deploying new dual policy...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(TECHNICAL_POLICY_PATH, ACTIVE_TECHNICAL_PATH)
    shutil.copy(FUNDAMENTAL_POLICY_PATH, ACTIVE_FUNDAMENTAL_PATH)
    print(f"   Technical:    {ACTIVE_TECHNICAL_PATH}")
    print(f"   Fundamental:  {ACTIVE_FUNDAMENTAL_PATH}")

    # Update deployment log
    deployment_log = {
        "deployment_date": datetime.now().isoformat(),
        "policy_type": "dual",
        "technical_updates": tech_updates,
        "fundamental_updates": fund_updates,
        "backup_timestamp": backup_ts,
        "validation_passed": not skip_validation,
    }

    with open(DEPLOYMENT_LOG_PATH, "w") as f:
        json.dump(deployment_log, f, indent=2)

    print("\n" + "=" * 70)
    print("DUAL POLICY DEPLOYMENT COMPLETE")
    print("=" * 70)
    print(f"\nThe dual policy is now active for live predictions.")
    print(f"Technical policy controls: Position signals (LONG/SHORT/SKIP)")
    print(f"Fundamental policy controls: Model weights and holding periods")
    return True


def rollback_policy() -> bool:
    """Rollback to previous policy version."""
    print("\n" + "=" * 70)
    print("ROLLING BACK RL POLICY")
    print("=" * 70)

    if not BACKUP_DIR.exists():
        print("ERROR: No backups found")
        return False

    # Find latest backup
    backups = sorted(BACKUP_DIR.glob("policy_*.pkl"), reverse=True)
    if not backups:
        print("ERROR: No backup policies found")
        return False

    latest_backup = backups[0]
    timestamp = latest_backup.stem.replace("policy_", "")
    normalizer_backup = BACKUP_DIR / f"normalizer_{timestamp}.pkl"

    print(f"\n1. Found backup: {latest_backup.name}")

    # Validate backup
    print("\n2. Validating backup...")
    try:
        policy, _ = load_policy(latest_backup, normalizer_backup)
        print(f"   Valid policy: {policy.name} v{policy.version}")
    except Exception as e:
        print(f"   ERROR: Invalid backup - {e}")
        return False

    # Restore
    print("\n3. Restoring backup...")
    shutil.copy(latest_backup, ACTIVE_POLICY_PATH)
    if normalizer_backup.exists():
        shutil.copy(normalizer_backup, ACTIVE_NORMALIZER_PATH)

    # Remove used backup
    latest_backup.unlink()
    if normalizer_backup.exists():
        normalizer_backup.unlink()

    print(f"   Restored from: {latest_backup.name}")

    print("\n" + "=" * 70)
    print("ROLLBACK COMPLETE")
    print("=" * 70)
    return True


def show_status():
    """Show deployment status."""
    print("\n" + "=" * 70)
    print("RL DEPLOYMENT STATUS")
    print("=" * 70)

    # Active Dual Policy
    print("\nActive Dual Policy:")
    if ACTIVE_TECHNICAL_PATH.exists() and ACTIVE_FUNDAMENTAL_PATH.exists():
        try:
            policy = load_dual_policy(
                technical_path=str(ACTIVE_TECHNICAL_PATH),
                fundamental_path=str(ACTIVE_FUNDAMENTAL_PATH),
            )
            print(f"  Technical Policy:    {policy.technical._update_count:,} updates")
            print(f"  Fundamental Policy:  {policy.fundamental._update_count:,} updates")
            print(f"  Ready: {policy.is_ready()}")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print("  No active dual policy deployed")

    # Trained Dual Policy
    print("\nTrained Dual Policy (pending deployment):")
    if TECHNICAL_POLICY_PATH.exists() and FUNDAMENTAL_POLICY_PATH.exists():
        try:
            policy = load_dual_policy_files()
            print(f"  Technical Policy:    {policy.technical._update_count:,} updates")
            print(f"  Fundamental Policy:  {policy.fundamental._update_count:,} updates")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print("  No trained dual policy available")

    # Legacy Active Policy
    print("\nLegacy Single Policy:")
    if ACTIVE_POLICY_PATH.exists():
        try:
            policy, _ = load_policy(ACTIVE_POLICY_PATH, ACTIVE_NORMALIZER_PATH)
            print(f"  Name: {policy.name}")
            print(f"  Version: {policy.version}")
            print(f"  Updates: {policy._update_count}")
        except Exception as e:
            print(f"  (not available: {e})")
    else:
        print("  No legacy policy deployed")

    # Backups
    print("\nBackups:")
    if BACKUP_DIR.exists():
        tech_backups = sorted(BACKUP_DIR.glob("technical_policy_*.pkl"), reverse=True)
        fund_backups = sorted(BACKUP_DIR.glob("fundamental_policy_*.pkl"), reverse=True)
        legacy_backups = sorted(BACKUP_DIR.glob("policy_*.pkl"), reverse=True)

        if tech_backups:
            print("  Dual Policy Backups:")
            for backup in tech_backups[:3]:
                ts = backup.stem.replace("technical_policy_", "")
                print(f"    - {ts}")
        if legacy_backups:
            print("  Legacy Backups:")
            for backup in legacy_backups[:3]:
                print(f"    - {backup.name}")
        if not tech_backups and not legacy_backups:
            print("  No backups")
    else:
        print("  No backup directory")

    # Deployment log
    print("\nLast Deployment:")
    if DEPLOYMENT_LOG_PATH.exists():
        with open(DEPLOYMENT_LOG_PATH) as f:
            log = json.load(f)
        print(f"  Date: {log.get('deployment_date', 'N/A')}")
        policy_type = log.get('policy_type', 'single')
        print(f"  Type: {policy_type}")
        if policy_type == 'dual':
            print(f"  Technical Updates: {log.get('technical_updates', 'N/A'):,}")
            print(f"  Fundamental Updates: {log.get('fundamental_updates', 'N/A'):,}")
        else:
            print(f"  Version: {log.get('policy_version', 'N/A')}")
        print(f"  Validation: {'PASSED' if log.get('validation_passed') else 'SKIPPED'}")
    else:
        print("  No deployment log")


def main():
    parser = argparse.ArgumentParser(description="Deploy RL policy")
    parser.add_argument("--validate", action="store_true", help="Validate and show results only")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation during deploy")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous policy")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    parser.add_argument("--legacy", action="store_true", help="Deploy legacy single policy instead of dual")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.rollback:
        success = rollback_policy()
        sys.exit(0 if success else 1)
    elif args.validate:
        print("\nValidating trained dual policy...")
        try:
            policy = load_dual_policy_files()
            validation = validate_dual_policy(policy)
            print(f"\nValidation: {'PASSED' if validation['valid'] else 'FAILED'}")
            for pred in validation["predictions"]:
                print(f"\n{pred['name']}:")
                print(f"  Position: {pred['position']} ({pred['confidence']})")
                print(f"  Holding Period: {pred['holding_period']}")
                print(f"  Weights:")
                for model, weight in pred["weights"].items():
                    print(f"    {model}: {weight:.0f}%")
            if validation["errors"]:
                print("\nErrors:")
                for error in validation["errors"]:
                    print(f"  - {error}")
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    elif args.legacy:
        # Deploy legacy single policy
        success = deploy_policy(skip_validation=args.skip_validation)
        sys.exit(0 if success else 1)
    else:
        # Default: deploy dual policy
        success = deploy_dual_policy(skip_validation=args.skip_validation)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

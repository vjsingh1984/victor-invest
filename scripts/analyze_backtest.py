#!/usr/bin/env python3
"""
Backtest Analysis Script - Slice and dice by all dimensions.

Usage:
    python scripts/analyze_backtest.py [log_file]
    python scripts/analyze_backtest.py --watch  # Poll every 10 minutes
"""

import re
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, 'src')

def analyze_backtest(log_file: str = "logs/rl_backtest_sp500_parallel.log"):
    """Analyze backtest log and print comprehensive report."""
    from investigator.domain.services.company_metadata_service import CompanyMetadataService
    metadata_service = CompanyMetadataService()

    # Parse log entries
    pattern = r"(\w+) \[(\d+)m back\]: FV=\$([0-9.]+), Price=\$([0-9.]+), Gap=([+-]?[0-9.]+)%, Signal=(\w+), Conf=(\d+)%, Reward\(L/S\)=([0-9.\-N/A]+)/([0-9.\-N/A]+)"

    signals = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                symbol, lookback, fv, price, gap, signal, conf, long_r, short_r = match.groups()
                sector = metadata_service.get_sector(symbol) or 'Unknown'

                try:
                    long_reward = float(long_r) if long_r != 'N/A' else None
                except:
                    long_reward = None
                try:
                    short_reward = float(short_r) if short_r != 'N/A' else None
                except:
                    short_reward = None

                signals.append({
                    'symbol': symbol,
                    'lookback': int(lookback),
                    'fv': float(fv),
                    'price': float(price),
                    'gap': float(gap),
                    'signal': signal,
                    'confidence': int(conf),
                    'long_reward': long_reward,
                    'short_reward': short_reward,
                    'sector': sector,
                })

    if not signals:
        print("No signals found in log file")
        return

    total_signals = len(signals)
    unique_symbols = len(set(s['symbol'] for s in signals))

    print("\n" + "=" * 70)
    print(f"BACKTEST PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Symbols processed: {unique_symbols}/616 ({unique_symbols/616*100:.1f}%)")
    print(f"Total signals: {total_signals}")
    print()

    # Signal Distribution
    print("OVERALL SIGNAL DISTRIBUTION:")
    signal_counts = defaultdict(int)
    for s in signals:
        signal_counts[s['signal']] += 1
    for sig in ['LONG', 'SHORT', 'SKIP']:
        count = signal_counts[sig]
        pct = count / total_signals * 100 if total_signals > 0 else 0
        print(f"  {sig:6}: {count:4} ({pct:5.1f}%)")
    print()

    # By Sector
    print("=" * 70)
    print("BY SECTOR")
    print("=" * 70)
    by_sector = defaultdict(lambda: {'LONG': 0, 'SHORT': 0, 'SKIP': 0, 'long_rewards': [], 'short_rewards': []})
    for s in signals:
        by_sector[s['sector']][s['signal']] += 1
        if s['long_reward'] is not None:
            by_sector[s['sector']]['long_rewards'].append(s['long_reward'])
        if s['short_reward'] is not None:
            by_sector[s['sector']]['short_rewards'].append(s['short_reward'])

    print(f"{'Sector':<25} {'LONG':>6} {'SHORT':>6} {'SKIP':>6} {'L_Avg':>8} {'S_Avg':>8}")
    print("-" * 65)
    for sector in sorted(by_sector.keys()):
        data = by_sector[sector]
        l_avg = sum(data['long_rewards'])/len(data['long_rewards']) if data['long_rewards'] else 0
        s_avg = sum(data['short_rewards'])/len(data['short_rewards']) if data['short_rewards'] else 0
        print(f"{sector:<25} {data['LONG']:>6} {data['SHORT']:>6} {data['SKIP']:>6} {l_avg:>+8.3f} {s_avg:>+8.3f}")
    print()

    # By Lookback
    print("=" * 70)
    print("BY LOOKBACK PERIOD")
    print("=" * 70)
    by_lb = defaultdict(lambda: {'LONG': 0, 'SHORT': 0, 'SKIP': 0, 'long_rewards': [], 'short_rewards': []})
    for s in signals:
        by_lb[s['lookback']][s['signal']] += 1
        if s['long_reward'] is not None:
            by_lb[s['lookback']]['long_rewards'].append(s['long_reward'])
        if s['short_reward'] is not None:
            by_lb[s['lookback']]['short_rewards'].append(s['short_reward'])

    print(f"{'Lookback':>10} {'LONG':>6} {'SHORT':>6} {'SKIP':>6} {'L_Avg':>8} {'S_Avg':>8}")
    print("-" * 55)
    for lb in sorted(by_lb.keys(), reverse=True):
        data = by_lb[lb]
        l_avg = sum(data['long_rewards'])/len(data['long_rewards']) if data['long_rewards'] else 0
        s_avg = sum(data['short_rewards'])/len(data['short_rewards']) if data['short_rewards'] else 0
        print(f"{lb:>7}m {data['LONG']:>6} {data['SHORT']:>6} {data['SKIP']:>6} {l_avg:>+8.3f} {s_avg:>+8.3f}")
    print()

    # By Confidence
    print("=" * 70)
    print("BY CONFIDENCE LEVEL")
    print("=" * 70)
    by_conf = defaultdict(lambda: {'LONG': 0, 'SHORT': 0, 'SKIP': 0})
    for s in signals:
        by_conf[s['confidence']][s['signal']] += 1

    print(f"{'Conf':>8} {'LONG':>6} {'SHORT':>6} {'SKIP':>6} {'Total':>6}")
    print("-" * 40)
    for conf in sorted(by_conf.keys()):
        data = by_conf[conf]
        total = data['LONG'] + data['SHORT'] + data['SKIP']
        print(f"{conf:>7}% {data['LONG']:>6} {data['SHORT']:>6} {data['SKIP']:>6} {total:>6}")
    print()

    # Signal Accuracy
    print("=" * 70)
    print("SIGNAL ACCURACY (following recommendation)")
    print("=" * 70)
    for sig in ['LONG', 'SHORT']:
        sig_rewards = []
        for s in signals:
            if s['signal'] == sig:
                reward = s['long_reward'] if sig == 'LONG' else s['short_reward']
                if reward is not None:
                    sig_rewards.append(reward)

        if sig_rewards:
            wins = sum(1 for r in sig_rewards if r > 0)
            avg = sum(sig_rewards) / len(sig_rewards)
            print(f"{sig}: {wins}/{len(sig_rewards)} profitable ({wins/len(sig_rewards)*100:.1f}%), avg: {avg:+.3f}")
    print()

    # Top performers
    print("=" * 70)
    print("TOP 5 BEST LONG SIGNALS")
    print("=" * 70)
    long_sigs = [(s['symbol'], s['lookback'], s['gap'], s['long_reward'], s['sector'])
                 for s in signals if s['signal'] == 'LONG' and s['long_reward'] is not None]
    long_sigs.sort(key=lambda x: -x[3])
    for sym, lb, gap, rew, sec in long_sigs[:5]:
        print(f"  {sym:6} [{lb:3}m] {sec:20} Gap={gap:+7.1f}% → Rew={rew:+.3f}")

    print()
    print("TOP 5 WORST LONG SIGNALS")
    print("-" * 70)
    for sym, lb, gap, rew, sec in long_sigs[-5:]:
        print(f"  {sym:6} [{lb:3}m] {sec:20} Gap={gap:+7.1f}% → Rew={rew:+.3f}")

    print()
    print("=" * 70)
    print("TOP 5 BEST SHORT SIGNALS")
    print("=" * 70)
    short_sigs = [(s['symbol'], s['lookback'], s['gap'], s['short_reward'], s['sector'])
                  for s in signals if s['signal'] == 'SHORT' and s['short_reward'] is not None]
    short_sigs.sort(key=lambda x: -x[3])
    for sym, lb, gap, rew, sec in short_sigs[:5]:
        print(f"  {sym:6} [{lb:3}m] {sec:20} Gap={gap:+7.1f}% → Rew={rew:+.3f}")

    print()
    print("TOP 5 WORST SHORT SIGNALS")
    print("-" * 70)
    for sym, lb, gap, rew, sec in short_sigs[-5:]:
        print(f"  {sym:6} [{lb:3}m] {sec:20} Gap={gap:+7.1f}% → Rew={rew:+.3f}")

    return {
        'symbols_processed': unique_symbols,
        'total_signals': total_signals,
        'signal_counts': dict(signal_counts),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", nargs="?", default="logs/rl_backtest_sp500_parallel.log")
    parser.add_argument("--watch", action="store_true", help="Poll every 10 minutes")
    args = parser.parse_args()

    if args.watch:
        print("Watching backtest progress (Ctrl+C to stop)...")
        while True:
            try:
                analyze_backtest(args.log_file)
                print("\n" + "=" * 70)
                print("Next update in 10 minutes...")
                print("=" * 70)
                time.sleep(600)
            except KeyboardInterrupt:
                print("\nStopped watching.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(60)
    else:
        analyze_backtest(args.log_file)

"""
compare_paper.py — Direct comparison of our results against paper Table I.

Paper values are from arXiv:2503.05848v1, Table I.
Prints a side-by-side table for success rate and arrival rate.

Usage
-----
    python experiments/compare_paper.py
    python experiments/compare_paper.py --csv results/raw_results.csv --n 40
"""

import argparse
import pathlib
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Paper ground-truth values (Table I)
# Format: {N: {(env, method): (success_rate, arrival_rate)}}
# ---------------------------------------------------------------------------
PAPER = {
    20: {
        ('free',    'mgr'):     (1.00,   1.0000),
        ('free',    'clf_cbf'): (0.80,   0.9745),
        ('free',    'orca'):    (1.00,   1.0000),

        ('circ15',  'mgr'):     (0.95,   0.9975),
        ('circ15',  'clf_cbf'): (0.40,   0.8625),
        ('circ15',  'orca'):    (0.65,   0.9505),

        ('rect15',  'mgr'):     (0.95,   0.9949),
        ('rect15',  'clf_cbf'): (0.30,   0.8225),
        ('rect15',  'orca'):    (0.75,   0.9640),

        ('swap',    'mgr'):     (1.00,   1.0000),
        ('swap',    'clf_cbf'): (1.00,   1.0000),
        ('swap',    'orca'):    (1.00,   1.0000),
    },
    40: {
        ('free',    'mgr'):     (1.00,   1.0000),
        ('free',    'clf_cbf'): (0.60,   0.9598),
        ('free',    'orca'):    (1.00,   1.0000),

        ('circ15',  'mgr'):     (0.95,   0.9981),
        ('circ15',  'clf_cbf'): (0.10,   0.7808),
        ('circ15',  'orca'):    (0.55,   0.9285),

        ('rect15',  'mgr'):     (0.90,   0.9931),
        ('rect15',  'clf_cbf'): (0.10,   0.7350),
        ('rect15',  'orca'):    (0.65,   0.9468),

        ('swap',    'mgr'):     (1.00,   1.0000),
        ('swap',    'clf_cbf'): (1.00,   1.0000),
        ('swap',    'orca'):    (1.00,   1.0000),
    },
    60: {
        ('free',    'mgr'):     (1.00,   1.0000),
        ('free',    'clf_cbf'): (0.40,   0.9355),
        ('free',    'orca'):    (1.00,   1.0000),

        ('circ15',  'mgr'):     (0.90,   0.9962),
        ('circ15',  'clf_cbf'): (0.00,   0.6942),
        ('circ15',  'orca'):    (0.40,   0.8952),

        ('rect15',  'mgr'):     (0.85,   0.9895),
        ('rect15',  'clf_cbf'): (0.00,   0.6625),
        ('rect15',  'orca'):    (0.55,   0.9278),

        ('swap',    'mgr'):     (1.00,   1.0000),
        ('swap',    'clf_cbf'): (0.95,   0.9997),
        ('swap',    'orca'):    (1.00,   1.0000),
    },
    80: {
        ('free',    'mgr'):     (1.00,   1.0000),
        ('free',    'clf_cbf'): (0.15,   0.9073),
        ('free',    'orca'):    (1.00,   1.0000),

        ('circ15',  'mgr'):     (0.85,   0.9938),
        ('circ15',  'clf_cbf'): (0.00,   0.6275),
        ('circ15',  'orca'):    (0.30,   0.8660),

        ('rect15',  'mgr'):     (0.80,   0.9862),
        ('rect15',  'clf_cbf'): (0.00,   0.5988),
        ('rect15',  'orca'):    (0.45,   0.9085),

        ('swap',    'mgr'):     (1.00,   1.0000),
        ('swap',    'clf_cbf'): (0.80,   0.9975),
        ('swap',    'orca'):    (1.00,   1.0000),
    },
}

ENV_ORDER    = ['free', 'circ15', 'rect15', 'swap']
METHOD_ORDER = ['mgr', 'clf_cbf', 'orca']
METHOD_LABEL = {'mgr': 'MGR', 'clf_cbf': 'CLF-CBF', 'orca': 'ORCA'}


def load_ours(csv_path: pathlib.Path) -> dict:
    """Return {(env, method): (success_mean, arrival_mean)} from CSV."""
    df = pd.read_csv(csv_path)
    out = {}
    for (env, method), g in df.groupby(['env', 'method']):
        out[(env, method)] = (g['success_rate'].mean(), g['arrival_rate'].mean())
    return out


def detect_n(csv_path: pathlib.Path) -> int:
    """Infer N from the CSV."""
    df = pd.read_csv(csv_path)
    return int(df['N'].iloc[0])


def delta_str(ours: float, paper: float) -> str:
    diff = ours - paper
    sign = '+' if diff >= 0 else ''
    return f"{sign}{diff*100:.1f}pp"


def check_mark(ours: float, paper: float, tol: float = 0.02) -> str:
    return '✓' if ours >= paper - tol else '✗'


def print_comparison(ours: dict, paper_n: dict, n: int) -> None:
    sep = "=" * 100

    print(sep)
    print(f"DIRECT COMPARISON: Our N={n} Results vs Paper Table I (arXiv:2503.05848v1)")
    print(sep)

    # --- Success Rate ---
    print("\n── SUCCESS RATE ──────────────────────────────────────────────────────────────")
    header = f"{'Env':<8}  {'Method':<9}  {'Paper':>7}  {'Ours':>7}  {'Δ':>8}  {'':>2}"
    print(header)
    print("-" * 50)
    for env in ENV_ORDER:
        for method in METHOD_ORDER:
            key = (env, method)
            paper_s, _ = paper_n.get(key, (None, None))
            our_s, _ = ours.get(key, (None, None))
            if paper_s is None or our_s is None:
                continue
            mark = check_mark(our_s, paper_s)
            print(f"{env:<8}  {METHOD_LABEL[method]:<9}  {paper_s*100:6.1f}%  {our_s*100:6.1f}%  {delta_str(our_s, paper_s):>8}  {mark}")
        print()

    # --- Arrival Rate ---
    print("── ARRIVAL RATE ──────────────────────────────────────────────────────────────")
    header = f"{'Env':<8}  {'Method':<9}  {'Paper':>8}  {'Ours':>8}  {'Δ':>8}  {'':>2}"
    print(header)
    print("-" * 55)
    for env in ENV_ORDER:
        for method in METHOD_ORDER:
            key = (env, method)
            _, paper_a = paper_n.get(key, (None, None))
            _, our_a = ours.get(key, (None, None))
            if paper_a is None or our_a is None:
                continue
            mark = check_mark(our_a, paper_a)
            print(f"{env:<8}  {METHOD_LABEL[method]:<9}  {paper_a*100:7.2f}%  {our_a*100:7.2f}%  {delta_str(our_a, paper_a):>8}  {mark}")
        print()

    # --- MGR Summary ---
    print(sep)
    print("MGR SUMMARY (primary method)")
    print(sep)
    print(f"{'Env':<8}  {'Paper Suc':>10}  {'Ours Suc':>10}  {'Paper Arr':>10}  {'Ours Arr':>10}  {'Status'}")
    print("-" * 70)
    for env in ENV_ORDER:
        key = (env, 'mgr')
        ps, pa = paper_n.get(key, (0, 0))
        os_, oa = ours.get(key, (0, 0))
        suc_ok = check_mark(os_, ps)
        arr_ok = check_mark(oa, pa)
        status = '✓ MATCH' if suc_ok == '✓' and arr_ok == '✓' else '✗ GAP'
        print(f"{env:<8}  {ps*100:9.1f}%  {os_*100:9.1f}%  {pa*100:9.2f}%  {oa*100:9.2f}%  {status}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results/raw_results.csv')
    parser.add_argument('--n', type=int, default=None,
                        help='Robot count (inferred from CSV if omitted)')
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run experiments/run_experiments.py first.")
        sys.exit(1)

    ours = load_ours(csv_path)
    n = args.n if args.n is not None else detect_n(csv_path)

    if n not in PAPER:
        print(f"WARNING: No paper values for N={n}. Available: {sorted(PAPER.keys())}")
        sys.exit(1)

    print_comparison(ours, PAPER[n], n)


if __name__ == '__main__':
    main()

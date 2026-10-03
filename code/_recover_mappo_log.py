# -*- coding: utf-8 -*-
"""
    Recover per-seed results from MAPPO logs if the final JSON is lost.
"""
import re
import json
import sys
import os

SEED_RE = re.compile(r'---\s*seed\s+(\d+)\s*---')
FINAL_RE = re.compile(r'final\s+gap\s*=\s*([\d.]+)%\s*\+\-\s*([\d.]+)')
BEST_RE = re.compile(r'best=\s*([\d.]+)%')
UPDATE_RE = re.compile(r'u=\s*(\d+)\s+eval_gap=')
EARLY_RE = re.compile(r'early\s+stop\s+at\s+u=(\d+)')

def parse_log(path):
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    seeds = {}
    cur = None
    for ln in lines:
        m = SEED_RE.search(ln)
        if m:
            cur = {'seed': int(m.group(1)), 'final_gap': None, 'gap_std': None,
                   'best_gap': None, 'updates_done': 0, 'complete': False}
            seeds[cur['seed']] = cur
            continue
        if cur is None:
            continue
        m = FINAL_RE.search(ln)
        if m:
            cur['final_gap'] = float(m.group(1))
            cur['gap_std'] = float(m.group(2))
            cur['complete'] = True
            continue
        m = BEST_RE.search(ln)
        if m:
            cur['best_gap'] = float(m.group(1))
        m = UPDATE_RE.search(ln)
        if m:
            cur['updates_done'] = max(cur['updates_done'], int(m.group(1)))
        m = EARLY_RE.search(ln)
        if m:
            cur['updates_done'] = int(m.group(1))
    return seeds

def aggregate(seeds):
    done = [s for s in seeds.values() if s['complete']]
    if not done:
        return {'n_complete': 0}
    gaps = [s['final_gap'] for s in done]
    mean = sum(gaps) / len(gaps)
    var = sum((g - mean) ** 2 for g in gaps) / (len(gaps) - 1)
    return {
        'n_complete': len(done),
        'final_gap_mean': round(mean, 4),
        'final_gap_std': round(var ** 0.5, 4),
        'best_gap_mean': round(sum(s['best_gap'] for s in done if s['best_gap'] is not None) / len(done), 4),
        'seeds': [s['seed'] for s in done],
    }

def main():
    if len(sys.argv) < 2:
        print('usage: python _recover_mappo_log.py <log_path> [out_json]')
        return 1
    path = sys.argv[1]
    seeds = parse_log(path)
    agg = aggregate(seeds)
    result = {'source': os.path.basename(path), 'per_seed': seeds, 'aggregate': agg}
    print('seeds found: %d, complete: %d' % (len(seeds), agg['n_complete']))
    for s in sorted(seeds.values(), key=lambda x: x['seed']):
        if s['complete']:
            print('  seed %-8d final_gap=%.2f%% best=%.2f%% updates=%d' %
                  (s['seed'], s['final_gap'], s['best_gap'] or 0.0, s['updates_done']))
        else:
            print('  seed %-8d INCOMPLETE (up to u=%d, best=%.2f%%)' %
                  (s['seed'], s['updates_done'], s['best_gap'] or 0.0))
    if agg['n_complete'] > 0:
        print('aggregate: mean final gap = %.2f%% +- %.2f (n=%d)' %
              (agg['final_gap_mean'], agg['final_gap_std'], agg['n_complete']))
    if len(sys.argv) >= 3:
        json.dump(result, open(sys.argv[2], 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print('saved: %s' % sys.argv[2])
    return 0

if __name__ == '__main__':
    sys.exit(main())

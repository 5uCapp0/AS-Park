# -*- coding: utf-8 -*-
"""
    Launch L3 experiment jobs (multi-seed).
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
PY = sys.executable
RUNNER = os.path.join(HERE, 'run_seed_job.py')

SEEDS = [20260925, 7, 42, 123, 2024]
SIGMA2_GRID = [0.25, 0.5, 1.0, 2.0]


def launch(cmd_list):
    procs = [subprocess.Popen(cmd) for cmd in cmd_list]
    for p in procs:
        p.wait()
    return len(procs)


def main():
    os.makedirs(DATA, exist_ok=True)
    sigma_jobs = []
    for s2 in SIGMA2_GRID:
        for s in SEEDS:
            out = os.path.join(DATA, 'job_sigma_%s_%d.json' % (s2, s))
            sigma_jobs.append([PY, RUNNER, 'sigma', str(s2), str(s), out])
    two_d_jobs = []
    for s in SEEDS:
        out = os.path.join(DATA, 'job_2d_%d.json' % s)
        two_d_jobs.append([PY, RUNNER, '2d', str(s), out])
    main_jobs = []
    for s in SEEDS:
        out = os.path.join(DATA, 'job_main_%d.json' % s)
        main_jobs.append([PY, RUNNER, 'main', str(s), out])

    all_jobs = sigma_jobs + two_d_jobs + main_jobs
    n1 = launch(all_jobs[:16])
    print('batch1 done: %d jobs' % n1, flush=True)
    n2 = launch(all_jobs[16:])
    print('batch2 done: %d jobs' % n2, flush=True)
    print('ALL JOBS DONE', flush=True)


if __name__ == '__main__':
    main()

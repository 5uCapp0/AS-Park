# -*- coding: utf-8 -*-
"""
    Run all L2 experiments end to end.
"""
import os, sys, subprocess, time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

MODULES = ['l2_a1', 'l2_fig4', 'l2_a2', 'l2_a3', 'l2_a4', 'l2_a5', 'l2_visualize']


def run(mod):
    t0 = time.time()
    print(f"\n########## RUN {mod} ##########", flush=True)
    r = subprocess.run([PY, "-u", os.path.join(HERE, mod + ".py")], cwd=HERE)
    print(f"########## {mod} exit={r.returncode} ({time.time()-t0:.0f}s) ##########", flush=True)
    return r.returncode


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if target == 'all':
        for m in MODULES:
            run(m)
    else:
        run(target)
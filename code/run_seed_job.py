# -*- coding: utf-8 -*-
"""
    Run a single seed job for the L3 online dispatcher.
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    mode = sys.argv[1]
    if mode == 'sigma':
        sigma2 = float(sys.argv[2])
        seed = int(sys.argv[3])
        out_json = sys.argv[4]
        import l3_sigma_sensitivity as s
        res, eps2, lam = s.run_seed_sigma(seed, 'cpu', sigma2)
        payload = {'mode': 'sigma', 'sigma2': sigma2, 'seed': seed,
                   'res': res, 'eps2': eps2, 'lam_star': lam}
    elif mode == '2d':
        seed = int(sys.argv[2])
        out_json = sys.argv[3]
        import l3_online_2d as o2
        res, eps2, lam = o2.run_seed_2d(seed, 'cpu')
        payload = {'mode': '2d', 'seed': seed, 'res': res, 'eps2': eps2, 'lam_star': lam}
    elif mode == 'main':
        seed = int(sys.argv[2])
        out_json = sys.argv[3]
        import l3_online_dispatcher as lod
        res, eps2, lam, _ = lod.run_seed(seed, 'cpu')
        payload = {'mode': 'main', 'seed': seed, 'res': res, 'eps2': eps2, 'lam_star': lam}
    else:
        raise SystemExit('bad mode: ' + str(mode))
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    print('saved', out_json, flush=True)


if __name__ == '__main__':
    main()

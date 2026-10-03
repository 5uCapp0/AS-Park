# -*- coding: utf-8 -*-
"""
    Calibrate congestion / visibility of scenario families (read-only geometry selection).
"""
import numpy as np
import l2_common as L
from common import bfs_path, closed_loop_assign, simulate_movement


def measure(W, H, make_obs, gen, n_scenes=60, p=0.4, seed=20260925):
    rng = np.random.default_rng(seed)
    obstacles = make_obs(W, H, rng)
    scenes = gen(n_scenes, W, H, obstacles, p, rng)
    confs, waits = [], []
    for sc in scenes:
        N, M = len(sc['car_pos']), len(sc['spots'])
        scores = np.zeros((N, M))
        assign, _ = closed_loop_assign(scores)
        paths = []
        for i in range(N):
            if i not in assign:
                continue
            j = assign[i]
            pth = bfs_path(W, H, obstacles, sc['car_pos'][i], sc['spots'][j])
            if pth:
                paths.append(pth)
        _, w, c = simulate_movement(paths)
        confs.append(c); waits.append(w)
    return float(np.mean(confs)), float(np.mean(waits))


def obs_open(W, H, rng):
    return L.build_open_obstacles(W, H, n_obs=6, rng=rng)


def gen_open_factory(n_cars, M):
    def gen(n, W, H, obstacles, p, rng):
        return L.gen_scenes_open(n, W, H, obstacles, n_cars, M, p, rng)
    return gen


def obs_corridor(W, H, rng):
    obs, _ = L.build_corridor_obstacles(W, H)
    return obs


def gen_corr_factory(n_cars, M):
    def gen(n, W, H, obstacles, p, rng):
        return L.gen_scenes_corridor(n, W, H, obstacles, n_cars, M, p, rng)
    return gen


if __name__ == '__main__':
    print("=== family congestion calibration (dummy equal split, mean over 60 scenes)===")
    cands_A = [
        ("A:10x10 obs6 N14 M14", 10, 10, obs_open, gen_open_factory(14, 14)),
        ("A:10x10 obs6 N12 M14", 10, 10, obs_open, gen_open_factory(12, 14)),
        ("A:8x8  obs5 N12 M12", 8, 8, obs_open, gen_open_factory(12, 12)),
        ("A:10x10 obs10 N14 M14", 10, 10, obs_open, gen_open_factory(14, 14)),
    ]
    cands_B = [
        ("B:12x8 wall N8 M10", 12, 8, obs_corridor, gen_corr_factory(8, 10)),
        ("B:12x8 wall N8 M8", 12, 8, obs_corridor, gen_corr_factory(8, 8)),
        ("B:12x8 wall N10 M12", 12, 8, obs_corridor, gen_corr_factory(10, 12)),
    ]
    for name, W, H, obsf, genf in cands_A + cands_B:
        mc, mw = measure(W, H, obsf, genf)
        flag = "OK" if mc >= 2.0 else "low"
        print(f"  {name:28s}  avg_conflicts={mc:6.2f}  avg_wait={mw:6.2f}  [{flag}]")

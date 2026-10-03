# -*- coding: utf-8 -*-
"""
    Shared utilities for the AS-Park closed-loop scheduling experiments (torch).
"""
import os
import sys
from collections import deque

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import parking_sim as ps

SEED = 20260925
FIG = os.path.join(os.path.dirname(_THIS_DIR), 'figs')
os.makedirs(FIG, exist_ok=True)

CAR_COLORS = {0: '#3498db', 1: '#2ecc71', 2: '#e67e22'}
CAR_NAMES = {0: 'normal', 1: 'EV', 2: 'large'}
SPOT_EDGE = {0: '#7f8c8d', 1: '#27ae60', 2: '#d35400'}
SPOT_NAMES = {0: 'normal', 1: 'EV', 2: 'large'}


def reset_seed(seed=SEED):
    ps.rng = np.random.default_rng(seed)
    import torch
    torch.manual_seed(seed)


# ============================================================
# ============================================================
def bfs_path(W, H, obstacles, start, goal):
    if start == goal:
        return [start]
    d = {start: 0}
    parent = {}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in obstacles and (nx, ny) not in d:
                d[(nx, ny)] = d[(x, y)] + 1
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))
                if (nx, ny) == goal:
                    path = [goal]
                    cur = goal
                    while cur != start:
                        cur = parent[cur]
                        path.append(cur)
                    return path[::-1]
    return None


# ============================================================
# ============================================================
def closed_loop_assign(scores, car_order=None):
    N, M = scores.shape
    if car_order is None:
        car_order = list(range(N))
    occupied = set()
    assign = {}
    for i in car_order:
        avail = [j for j in range(M) if j not in occupied]
        if not avail:
            break
        j = max(avail, key=lambda jj: scores[i, jj])
        assign[i] = j
        occupied.add(j)
    return assign, occupied


# ============================================================
# ============================================================
def simulate_movement(paths, max_steps=None):
    n = len(paths)
    if n == 0:
        return [], 0, 0
    step = [0] * n
    wait = [0] * n
    n_conflicts = 0
    max_len = max(len(p) for p in paths)
    cap = max_steps if max_steps is not None else max_len * 4 + 10
    for _ in range(cap):
        nxt = [None] * n
        for i in range(n):
            if step[i] + 1 < len(paths[i]):
                nxt[i] = paths[i][step[i] + 1]
            else:
                nxt[i] = paths[i][step[i]]
        cur = [paths[i][step[i]] for i in range(n)]
        move = [True] * n
        for i in range(n):
            for j in range(i + 1, n):
                if nxt[i] == nxt[j] and nxt[i] != cur[i] and nxt[j] != cur[j]:
                    move[j] = False
                    n_conflicts += 1
        for i in range(n):
            for j in range(i + 1, n):
                if nxt[i] == cur[j] and nxt[j] == cur[i] and nxt[i] != cur[i]:
                    move[j] = False
                    n_conflicts += 1
        for i in range(n):
            if move[i] and step[i] + 1 < len(paths[i]):
                step[i] += 1
            elif step[i] + 1 < len(paths[i]):
                wait[i] += 1
        if all(step[i] == len(paths[i]) - 1 for i in range(n)):
            break
    final = [paths[i][step[i]] for i in range(n)]
    return final, int(sum(wait)), int(n_conflicts)


def assert_no_collision(paths):
    n = len(paths)
    max_len = max(len(p) for p in paths)
    for t in range(max_len + 2):
        pos = [paths[i][min(t, len(paths[i]) - 1)] for i in range(n)]
        seen = {}
        for i, p in enumerate(pos):
            if p in seen:
                raise AssertionError(f"vertex conflict @t={t}: car {seen[p]} and {i} both at {p}")
            seen[p] = i
        for i in range(n):
            for j in range(i + 1, n):
                pi = paths[i][min(t, len(paths[i]) - 1)]
                pj = paths[j][min(t, len(paths[j]) - 1)]
                ni = paths[i][min(t + 1, len(paths[i]) - 1)]
                nj = paths[j][min(t + 1, len(paths[j]) - 1)]
                if pi == nj and pj == ni and pi != pj:
                    raise AssertionError(f"edge conflict @t={t}: cars {i} and {j} swap head-on")
    return True


def simulate_movement_trace(paths, max_steps=None):
    n = len(paths)
    if n == 0:
        return [], 0, 0
    step = [0] * n
    wait = [0] * n
    n_conflicts = 0
    max_len = max(len(p) for p in paths)
    cap = max_steps if max_steps is not None else max_len * 4 + 10
    trace = []
    for _ in range(cap):
        trace.append([paths[i][step[i]] for i in range(n)])
        nxt = [None] * n
        for i in range(n):
            nxt[i] = paths[i][step[i] + 1] if step[i] + 1 < len(paths[i]) else paths[i][step[i]]
        cur = [paths[i][step[i]] for i in range(n)]
        move = [True] * n
        for i in range(n):
            for j in range(i + 1, n):
                if nxt[i] == nxt[j] and nxt[i] != cur[i] and nxt[j] != cur[j]:
                    move[j] = False
                    n_conflicts += 1
        for i in range(n):
            for j in range(i + 1, n):
                if nxt[i] == cur[j] and nxt[j] == cur[i] and nxt[i] != cur[i]:
                    move[j] = False
                    n_conflicts += 1
        for i in range(n):
            if move[i] and step[i] + 1 < len(paths[i]):
                step[i] += 1
            elif step[i] + 1 < len(paths[i]):
                wait[i] += 1
        if all(step[i] == len(paths[i]) - 1 for i in range(n)):
            trace.append([paths[i][step[i]] for i in range(n)])
            break
    return trace, int(sum(wait)), int(n_conflicts)

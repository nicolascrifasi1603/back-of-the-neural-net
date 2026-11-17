"""Back of the Neural Net — More than a Hack 2027 (BAR-ATH possession challenge).

Rebuilds the submitted possession file from the match data bundle:

    python make_submission.py                      # writes back-of-the-neural-net.csv
    python make_submission.py --out other_name.csv

Pipeline (see back-of-the-neural-net.md for the reasoning behind each rule):
  1. Load tracking, rename player columns H_/A_ by team.
  2. Base layer: nearest-player claims within 1.5 m, then sticky possession (N = 10).
  3. Five dead-ball detectors mark frames (they never write labels):
     throw-ins, goals, goal kicks, corners, fouls / free kicks.
  4. Labels = base layer, with D on every marked graded frame and X on ungraded frames.

The code is the v8 path from my exploration notebooks, with plots and checks removed.
Every parameter is hand-set; see back-of-the-neural-net.md for how each was chosen.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'starter'))
import games                       # noqa: E402  (organiser helper, repo root)
import load_smart_tagging as lst   # noqa: E402  (organiser helper)

DATA = os.path.join(ROOT, 'data')
FPS = 25
N_STICKY = 10


def runs_of(mask):
    """(start, end) of each run of True in a boolean array, end inclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def longest_run(mask):
    r = runs_of(mask)
    return max((e - s + 1 for s, e in r), default=0)


# =============================================================================
# 1. Load data
# =============================================================================
def load():
    frame_df = pd.read_pickle(os.path.join(DATA, 'tactical_tracking.pkl'))
    with open(os.path.join(DATA, 'slot_team.json')) as f:
        slot_team = json.load(f)

    rename = {}
    for col in frame_df.columns:
        slot = col.rsplit('_', 1)[0]                       # '5_x' -> '5'
        if slot in slot_team:
            rename[col] = f'{slot_team[slot]}_{col}'       # '5_x' -> 'H_5_x'
    frame_df = frame_df.rename(columns=rename)
    return frame_df


# =============================================================================
# 2. Base layer: nearest player, graded frames, claims, sticky possession
# =============================================================================
def build_step_a(frame_df, LENGTH, WIDTH):
    x_cols = [c for c in frame_df.columns if c[:2] in ('H_', 'A_') and c.endswith('_x')]
    y_cols = [c[:-2] + '_y' for c in x_cols]

    players_x = frame_df[x_cols].to_numpy(dtype=float) * LENGTH
    players_y = frame_df[y_cols].to_numpy(dtype=float) * WIDTH
    ball_x = frame_df['ball_x'].to_numpy(dtype=float) * LENGTH
    ball_y = frame_df['ball_y'].to_numpy(dtype=float) * WIDTH

    dist = np.sqrt((players_x - ball_x[:, None])**2 + (players_y - ball_y[:, None])**2)
    ball_visible = ~np.isnan(ball_x)
    has_answer = ball_visible & ~np.all(np.isnan(dist), axis=1)

    n = len(frame_df)
    slot_team_of_col = np.array([c[0] for c in x_cols])
    nearest_idx = np.full(n, -1)
    nearest_idx[has_answer] = np.nanargmin(dist[has_answer], axis=1)
    nearest_dist = np.full(n, np.nan)
    nearest_dist[has_answer] = dist[has_answer, nearest_idx[has_answer]]
    nearest_team = np.full(n, 'none', dtype=object)
    nearest_team[has_answer] = slot_team_of_col[nearest_idx[has_answer]]

    step_a = pd.DataFrame({
        'frame': np.arange(n),
        'ball_visible': ball_visible,
        'nearest_team': nearest_team,
        'nearest_dist': nearest_dist,
    })

    # Graded frames and halves
    graded = np.zeros(n, dtype=bool)
    half_id = np.full(n, None, dtype=object)
    for half, first, last in games.half_video_blocks('BAR-ATH'):
        graded[first:last + 1] = True
        half_id[first:last + 1] = half
    step_a['graded'] = graded
    step_a['half'] = half_id

    # Claims: graded, ball visible, nearest player within 1.5 m
    POSS_RADIUS = 1.5
    mask = step_a['graded'] & step_a['ball_visible']
    close = step_a['nearest_dist'] <= POSS_RADIUS
    step_a['claim'] = step_a['nearest_team'].where(mask & close)

    # Ball speed (m/s) from position; NaN when this or the previous frame is missing
    ball_x_m = pd.Series(ball_x)
    ball_y_m = pd.Series(ball_y)
    step_a['ball_speed'] = (np.sqrt(ball_x_m.diff()**2 + ball_y_m.diff()**2) * FPS).to_numpy()
    return step_a


def sticky_possession(claims, n):
    """Possession label per frame for one half. Switch after n claims in a row, backdated."""
    labels = [None] * len(claims)
    current, count, run_start = None, 0, None
    for i, claim in enumerate(claims):
        if pd.isna(claim):
            pass
        elif current is None:
            current = claim
        elif claim == current:
            count = 0
        else:
            if count == 0:
                run_start = i
            count += 1
            if count >= n:
                current = claim
                labels[run_start:i] = [claim] * (i - run_start)
                count = 0
        labels[i] = current
    return labels


def apply_sticky(df, n):
    labels = pd.Series(None, index=df.index, dtype=object)
    for h in df['half'].dropna().unique():
        rows = df['half'] == h
        labels.loc[rows] = sticky_possession(df.loc[rows, 'claim'].tolist(), n)
    return labels.fillna('X')


# =============================================================================
# 3a. Throw-ins (marks step_a['throw_in_id'])
# =============================================================================
def detect_throw_ins(frame_df, step_a, LENGTH, WIDTH):
    EDGE_M, BRIDGE_S, MIN_MISS_S, MIN_LEN_S, CORNER_M = 3.0, 2.0, 3.0, 3.0, 5.0
    BRIDGE, MIN_MISS, MIN_LEN = (int(round(s * FPS)) for s in (BRIDGE_S, MIN_MISS_S, MIN_LEN_S))

    bx = frame_df['ball_x'].to_numpy(dtype=float) * LENGTH
    by = frame_df['ball_y'].to_numpy(dtype=float) * WIDTH
    tracked = ~np.isnan(by)
    missing = ~tracked
    edge = tracked & ((by <= EDGE_M) | (by >= WIDTH - EDGE_M))
    beyond = tracked & ((by < 0) | (by > WIDTH))

    rows = []
    for half, first, last in games.half_video_blocks('BAR-ATH'):
        out = (edge | missing)[first:last + 1].copy()
        for s, e in runs_of(~out):                         # bridge short INSIDE runs
            if 0 < s and e < len(out) - 1 and (e - s + 1) < BRIDGE:
                out[s:e + 1] = True
        for s, e in runs_of(out):
            s, e = s + first, e + first
            seg = slice(s, e + 1)
            out_sign = longest_run(missing[seg]) >= MIN_MISS or beyond[seg].any()
            long_enough = (e - s + 1) >= MIN_LEN
            from_line = edge[e] and e < last
            corner = from_line and (bx[e] <= CORNER_M or bx[e] >= LENGTH - CORNER_M)
            gone = np.flatnonzero(beyond[seg] | missing[seg])
            d_start = s + gone[0] if len(gone) else s
            rows.append({'d_start': d_start, 'd_end': e, 'out_sign': out_sign,
                         'long_enough': long_enough, 'from_line': bool(from_line),
                         'corner': bool(corner)})

    stretches = pd.DataFrame(rows)
    keep = (stretches['out_sign'] & stretches['long_enough']
            & stretches['from_line'] & ~stretches['corner'])
    throw_ins = stretches[keep].reset_index(drop=True)

    step_a['throw_in_id'] = -1
    for i, r in throw_ins.iterrows():
        step_a.loc[r['d_start']:r['d_end'], 'throw_in_id'] = i
    return throw_ins


# =============================================================================
# 3b. Goals (marks step_a['goal_id'])
# =============================================================================
def detect_goals(frame_df, step_a, events, LENGTH):
    G_PAIR_S, G_MIN_MISS_S, G_NEAR_GOAL_M, G_END_SEARCH_S = 30, 3, 15, 30
    G_VISIBLE_S, G_STILL_MS, G_KICK_MS, G_KICK_RUN, G_KICK_SEARCH_S = 1, 2.0, 6.0, 3, 10
    G_PAIR, G_MIN_MISS, G_END_SEARCH, G_VISIBLE, G_KICK_SEARCH = (
        int(round(s * FPS)) for s in (G_PAIR_S, G_MIN_MISS_S, G_END_SEARCH_S, G_VISIBLE_S, G_KICK_SEARCH_S))

    g_bx = frame_df['ball_x'].to_numpy(dtype=float) * LENGTH
    g_vis = ~np.isnan(g_bx)
    g_spd = step_a['ball_speed'].to_numpy()

    goal_tags_f = sorted((e['start_frame'], e['end_frame'] - 1) for e in events
                         if e['code'].upper() == 'GOALS')
    ko_tags_f = sorted((e['start_frame'], e['end_frame'] - 1) for e in events
                       if e['code'].upper() == 'SET PIECES'
                       and str(e['tags'].get('Type', '')).lower() == 'kick off (after goal)')
    g_miss_runs = runs_of(~g_vis)

    def near_goal_line(x):
        return x < G_NEAR_GOAL_M or x > LENGTH - G_NEAR_GOAL_M

    def find_goal_start(g_s, g_e):
        long_runs = [(s, e) for s, e in g_miss_runs if g_s <= s <= g_e and e - s + 1 >= G_MIN_MISS]
        near = [(s, e) for s, e in long_runs if s > 0 and near_goal_line(g_bx[s - 1])]
        if near:
            return near[0][0]
        if long_runs:
            return long_runs[0][0]
        return None

    def find_goal_end(search_from, fallback_end):
        hi = min(search_from + G_END_SEARCH, len(g_vis) - 1)
        R = None
        for s, e in runs_of(g_vis[search_from:hi + 1]):
            if e - s + 1 >= G_VISIBLE:
                R = search_from + s
                break
        if R is None:
            return fallback_end
        first_sec = g_spd[R:R + FPS]
        med = np.nanmedian(first_sec) if (~np.isnan(first_sec)).any() else np.nan
        if not med < G_STILL_MS:
            return R - 1
        fast = np.nan_to_num(g_spd[R:R + G_KICK_SEARCH + 1]) > G_KICK_MS
        for i in range(len(fast) - G_KICK_RUN + 1):
            if fast[i:i + G_KICK_RUN].all():
                return R + i - 1
        return R - 1

    rows = []
    for g_s, g_e in goal_tags_f:
        ko = next(((s, e) for s, e in ko_tags_f if g_s <= s <= g_e + G_PAIR), None)
        start = find_goal_start(g_s, g_e)
        search_from = ko[0] if ko else g_e
        end = find_goal_end(search_from, ko[1] if ko else g_e)
        ok = start is not None and start <= end
        rows.append({'d_start': start if ok else -1, 'd_end': end if ok else -1})
    goals = pd.DataFrame(rows)

    step_a['goal_id'] = -1
    for i, r in goals.iterrows():
        if r['d_start'] >= 0:
            step_a.loc[r['d_start']:r['d_end'], 'goal_id'] = i
    return goals


# =============================================================================
# 3c. Goal kicks (marks step_a['gk_id'])
# =============================================================================
def detect_goal_kicks(frame_df, step_a, LENGTH, WIDTH):
    GK_DEPTH_M = 6.0
    GK_Y_MIN, GK_Y_MAX = 23.0, 45.0
    GK_STILL_MS, GK_BRIDGE_S, GK_MIN_STILL_S = 2.0, 1.5, 2.0
    GK_KICK_MS, GK_KICK_RUN, GK_KICK_WIN_S = 6.0, 3, 5.0
    GK_LOOKBACK_S, GK_MIN_MISS_S, GK_NEAR_M, GK_CHAIN_S = 30.0, 3.0, 15.0, 2.0
    (GK_BRIDGE, GK_MIN_STILL, GK_KICK_WIN, GK_LOOKBACK, GK_MIN_MISS, GK_CHAIN) = (
        int(round(v * FPS)) for v in
        (GK_BRIDGE_S, GK_MIN_STILL_S, GK_KICK_WIN_S, GK_LOOKBACK_S, GK_MIN_MISS_S, GK_CHAIN_S))

    kx = frame_df['ball_x'].to_numpy(dtype=float) * LENGTH
    ky = frame_df['ball_y'].to_numpy(dtype=float) * WIDTH
    kvis = ~np.isnan(kx)
    kspd = step_a['ball_speed'].to_numpy(dtype=float)
    kgr = step_a['graded'].to_numpy(dtype=bool)

    # Step 1: still ball in a six-yard box
    in_box = kvis & ((kx <= GK_DEPTH_M) | (kx >= LENGTH - GK_DEPTH_M)) & (ky >= GK_Y_MIN) & (ky <= GK_Y_MAX)
    still_in_box = in_box & (kspd < GK_STILL_MS) & kgr
    stretches = []
    for s, e in runs_of(still_in_box):
        if stretches and s - stretches[-1][1] - 1 <= GK_BRIDGE:
            stretches[-1] = (stretches[-1][0], e)
        else:
            stretches.append((s, e))
    stretches = [(s, e) for s, e in stretches if still_in_box[s:e + 1].sum() >= GK_MIN_STILL]

    # Step 2: end of D = frame before the kick
    def gk_find_end(e):
        fast = np.nan_to_num(kspd[e + 1:e + 1 + GK_KICK_WIN]) > GK_KICK_MS
        for i in range(len(fast) - GK_KICK_RUN + 1):
            if fast[i:i + GK_KICK_RUN].all():
                return e + i
        return e

    # Step 3: start of D = when the ball went out over that goal line
    def gk_find_start(s, line_x):
        lo = max(s - GK_LOOKBACK, 0)
        beyond = kvis & ((kx < 0) if line_x == 0 else (kx > LENGTH))
        near = (lambda x: x < GK_NEAR_M) if line_x == 0 else (lambda x: x > LENGTH - GK_NEAR_M)
        events = [(a, b) for a, b in runs_of(beyond[lo:s])]
        events += [(a, b) for a, b in runs_of(~kvis[lo:s])
                   if b - a + 1 >= GK_MIN_MISS and lo + a > 0 and near(kx[lo + a - 1])]
        events = sorted((lo + a, lo + b) for a, b in events)
        if events:
            a, b = events[-1]                              # most recent exit
            for pa, pb in reversed(events[:-1]):           # chain pieces of the same exit
                if a - pb - 1 <= GK_CHAIN:
                    a = pa
                else:
                    break
            return a
        f = s - 1                                          # fallback: ball approaching the line
        while f >= lo and kvis[f] and abs(kx[f] - line_x) <= GK_DEPTH_M:
            f -= 1
        return f + 1

    rows = []
    for s, e in stretches:
        line_x = 0.0 if np.nanmedian(kx[s:e + 1]) < LENGTH / 2 else LENGTH
        rows.append({'d_start': gk_find_start(s, line_x), 'd_end': gk_find_end(e)})
    goal_kicks = pd.DataFrame(rows)

    step_a['gk_id'] = -1
    for i, r in goal_kicks.iterrows():
        step_a.loc[r['d_start']:r['d_end'], 'gk_id'] = i
    return goal_kicks


# =============================================================================
# 3d. Corners (marks step_a['corner_id'])
# =============================================================================
def detect_corners(frame_df, step_a, events, LENGTH):
    CK_LOOKBACK_S, CK_END_WIN_S, CK_VISIBLE_S, CK_MIN_MISS_S, CK_NEAR_M = 30.0, 15.0, 1.0, 3.0, 15.0
    CK_LOOKBACK, CK_END_WIN, CK_VISIBLE, CK_MIN_MISS = (
        int(round(v * FPS)) for v in (CK_LOOKBACK_S, CK_END_WIN_S, CK_VISIBLE_S, CK_MIN_MISS_S))

    cx = frame_df['ball_x'].to_numpy(dtype=float) * LENGTH
    cvis = ~np.isnan(cx)
    near_goal_line = lambda x: (x < CK_NEAR_M) or (x > LENGTH - CK_NEAR_M)  # noqa: E731
    beyond_goal_line = cvis & ((cx < 0) | (cx > LENGTH))

    ck_tags = sorted((ev['start_frame'], ev['end_frame'] - 1) for ev in events
                     if ev['code'].upper() == 'SET PIECES'
                     and str(ev['tags'].get('Type', '')).lower() == 'corner kick')

    def ck_find_end(t0):
        hi = min(t0 + CK_END_WIN, len(cvis) - 1)
        for a, b in runs_of(cvis[t0:hi + 1]):
            if b - a + 1 >= CK_VISIBLE:
                return t0 + a
        return None

    def ck_find_start(t0, R):
        lo = max(t0 - CK_LOOKBACK, 0)
        starts = []
        for a, b in runs_of(~cvis[lo:R]):
            a, b = lo + a, lo + b
            while a > 0 and not cvis[a - 1]:               # walk back to the real start of the gap
                a -= 1
            if b - a + 1 >= CK_MIN_MISS and a > 0 and near_goal_line(cx[a - 1]):
                starts.append(a)
        for a, b in runs_of(beyond_goal_line[lo:R]):
            starts.append(lo + a)
        return min(starts) if starts else None             # earliest exit event

    rows = []
    for t0, _ in ck_tags:
        R = ck_find_end(t0)
        start = ck_find_start(t0, R) if R is not None else None
        ok = R is not None and start is not None and start < R
        rows.append({'d_start': start if ok else -1, 'd_end': R - 1 if ok else -1})
    corners = pd.DataFrame(rows)

    step_a['corner_id'] = -1
    for i, r in corners.iterrows():
        if r['d_start'] >= 0:
            step_a.loc[r['d_start']:r['d_end'], 'corner_id'] = i
    return corners


# =============================================================================
# 3e. Fouls / free kicks (marks step_a['foul_id'])
# =============================================================================
def detect_fouls(frame_df, step_a):
    FL_SLOW_MS, FL_ROLL_S, FL_MIN_SLOW_S, FL_RISE_WIN_S = 1.6, 5.0, 5.0, 2.0
    FL_FAST_S, FL_RISE_MS, FL_KICK_MS, FL_KICK_RUN, FL_KICK_WIN_S = 3.0, 0.5, 6.0, 3, 3.0
    FL_ROLL, FL_MIN_SLOW, FL_RISE_WIN, FL_FAST, FL_KICK_WIN = (
        int(round(v * FPS)) for v in (FL_ROLL_S, FL_MIN_SLOW_S, FL_RISE_WIN_S, FL_FAST_S, FL_KICK_WIN_S))

    h_cols = [c for c in frame_df.columns if c.startswith('H_') and c.endswith('_s')]
    a_cols = [c for c in frame_df.columns if c.startswith('A_') and c.endswith('_s')]

    def roll(cols):
        return (frame_df[cols].astype(float).mean(axis=1)
                .rolling(FL_ROLL, center=True, min_periods=1).mean().to_numpy())

    fh, fa = roll(h_cols), roll(a_cols)
    fgr = step_a['graded'].to_numpy(dtype=bool)
    fspd = step_a['ball_speed'].to_numpy(dtype=float)

    slow = (fh < FL_SLOW_MS) & (fa < FL_SLOW_MS) & fgr
    slow_runs = [(s, e) for s, e in runs_of(slow) if e - s + 1 >= FL_MIN_SLOW]

    def fl_find_rise(s, e):
        base_h = np.nanmean(fh[max(s, e - FL_MIN_SLOW + 1):e + 1])
        base_a = np.nanmean(fa[max(s, e - FL_MIN_SLOW + 1):e + 1])
        for r in range(e + 1, min(e + 1 + FL_RISE_WIN, len(fh) - FL_FAST)):
            if not fgr[r:r + FL_FAST].all():
                break
            if (np.nanmean(fh[r:r + FL_FAST]) - base_h >= FL_RISE_MS and
                    np.nanmean(fa[r:r + FL_FAST]) - base_a >= FL_RISE_MS):
                return r
        return None

    def fl_find_kick(r):
        lo = max(r - FL_KICK_WIN, 0)
        hot = np.nan_to_num(fspd[lo:r + FL_KICK_WIN + 1]) > FL_KICK_MS
        kicks = [lo + i for i in range(len(hot) - FL_KICK_RUN + 1)
                 if hot[i:i + FL_KICK_RUN].all() and (i == 0 or not hot[i - 1])]
        return min(kicks, key=lambda k: abs(k - r)) if kicks else None

    other_d = ((step_a['throw_in_id'] >= 0) | (step_a['goal_id'] >= 0) |
               (step_a['gk_id'] >= 0) | (step_a['corner_id'] >= 0)).to_numpy()

    rows = []
    for s, e in slow_runs:
        r = fl_find_rise(s, e)
        if r is None:
            continue                                       # no clear restart -> live play
        k = fl_find_kick(r)
        d_end = (k if k is not None else r) - 1
        if d_end <= s or other_d[s:d_end + 1].any():       # specialist detectors win
            continue
        rows.append({'d_start': s, 'd_end': d_end})
    fouls = pd.DataFrame(rows)

    step_a['foul_id'] = -1
    for i, r in fouls.iterrows():
        step_a.loc[r['d_start']:r['d_end'], 'foul_id'] = i
    return fouls


# =============================================================================
# 4. Labels
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default=os.path.join(ROOT, 'back-of-the-neural-net.csv'))
    args = parser.parse_args()

    LENGTH, WIDTH = games.GAMES['BAR-ATH']['pitch']        # 105.3, 68.0 m
    frame_df = load()
    step_a = build_step_a(frame_df, LENGTH, WIDTH)
    events = lst.load()

    counts = {
        'throw-ins': len(detect_throw_ins(frame_df, step_a, LENGTH, WIDTH)),
        'goals': len(detect_goals(frame_df, step_a, events, LENGTH)),
        'goal kicks': len(detect_goal_kicks(frame_df, step_a, LENGTH, WIDTH)),
        'corners': len(detect_corners(frame_df, step_a, events, LENGTH)),
        'fouls': len(detect_fouls(frame_df, step_a)),
    }

    graded = step_a['graded'].to_numpy()
    is_d = ((step_a['throw_in_id'] >= 0) | (step_a['goal_id'] >= 0) | (step_a['gk_id'] >= 0) |
            (step_a['corner_id'] >= 0) | (step_a['foul_id'] >= 0)).to_numpy()

    labels = apply_sticky(step_a, N_STICKY).copy()
    labels[is_d & graded] = 'D'

    pd.DataFrame({'frame': step_a['frame'], 'label': labels}).to_csv(args.out, index=False)

    print('detector spells:', ', '.join(f'{k} {v}' for k, v in counts.items()))
    print(labels.value_counts().to_string())
    print(f'wrote {len(labels)} rows to {args.out}')


if __name__ == '__main__':
    main()

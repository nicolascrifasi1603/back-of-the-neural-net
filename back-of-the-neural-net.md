# Back of the Neural Net — technical document

**More than a Hack 2027 · BAR–ATH possession challenge**
Nicolas Crifasi · November 2026

---

## Entry status (please read first)

- **Submitted for evaluation only.** This entry is submitted to have the file scored against the private ground truth, not to compete for the Innovation Challenge or the Final Pitch.
- **Possible rule conflict.** Every label in the submission is produced by code. No frame was labelled or corrected by hand. However, during development I checked some rule outputs against the match video and hand-marked a small number of events (listed in full in [Disclosure](#disclosure)). RULES.md allows small amounts of hand-labelled data for validation, but the organisers' email of 3 Nov 2026 asked participants not to check or calibrate against video. Some of my checks fall on the wrong side of that email, so please treat this entry accordingly.

---

## 1. Approach in one paragraph

The solution is entirely rule-based: no model is trained. It works in three stages.

1. **Base layer: who has the ball.** "Sticky possession" gives every graded frame an `H` or `A`, as if the ball were always in play.
2. **Detectors: where play is stopped.** Five dead-ball detectors each look for one kind of stoppage: throw-ins, goals, goal kicks, corners, and fouls/free kicks. A detector does not change any labels. It only *marks* the frames it thinks are a stoppage, like a highlighter.
3. **Combine.** At the end, every marked frame is overwritten with `D`. Frames that are never scored (before kick-off, half-time, after the final whistle) are always `X`.

| Frame | Base layer | Marked by a detector? | Final label |
|---|---|---|---|
| 100 | H | no | `H` |
| 101 | H | yes (throw-in) | `D` |
| 102 | A | yes (throw-in) | `D` |
| 103 | A | no | `A` |

Keeping marking and labelling separate meant each detector could be switched on alone, scored on top of the base layer, and then stacked with the others. That way I could see exactly what each one added.

## 2. Assets used (and dropped)

| Asset | Used for | Status |
|---|---|---|
| Automatic tracking (players + ball, 25 fps) | Everything: nearest player to the ball, ball position and speed, team speeds | **Used** |
| Slot → team mapping (`slot_team.json`) | Splitting players into home and away | **Used** |
| Smart Tagging (set-piece tags) | Where to look for goals, kick-offs and corners | **Used** as a search window only. A tag never becomes D directly |
| Smart Tagging (phase tags) | Tested as a live-play signal for the foul rule | **Dropped.** Phase tags also cover real dead balls (97% of set-piece candidates sat inside one), so they couldn't separate the two |
| Public 5-minute ground truth | Scoring each version and checking detector boundaries | **Used** for checking. See Disclosure for one case where it influenced a design choice |
| Video | Checking a small number of rule outputs | See Disclosure |
| Public match statistics (web) | Confirming 11 corners and 3 goals, to interpret the tag counts | **Used** for interpretation only |

## 3. Possession rules implemented

**Versions at a glance.** Each version adds one rule. The submitted file, **v8**, is the v2 base layer plus the detectors from v4–v8. v3 was tried and dropped, so it is not part of the submission.

| Version | What it adds | In v8? |
|---|---|---|
| v1 | Claims only: possession = the most recent claim | Replaced by v2 |
| v2 | Sticky possession (N = 10) | Yes, base layer |
| v3 | Missing-ball window for dead balls | No, dropped (see 3.4) |
| v4 | Throw-in detector | Yes |
| v5 | Goal detector | Yes |
| v6 | Goal-kick detector | Yes |
| v7 | Corner detector | Yes |
| v8 | Foul / free-kick detector | Yes |

### 3.1 Base layer: sticky possession (v2)

1. **Claims.** On each frame where the ball is visible, the team of the nearest player claims the ball if that player is within **1.5 m**. The median nearest-player distance on graded frames is 1.9 m, with a peak at 0.5–1 m, so 1.5 m captures players on the ball without counting most loose balls.
2. **Stickiness.** Possession only switches after the other team makes **N = 10 claims in a row**. Missing-ball frames don't break a run (the ball is missing on ~41% of graded frames). A claim by the current team resets the run.
3. **Backdating.** A switch is backdated to the first claim of the run, so the boundary lands where the new team actually won the ball.
4. **Halves** are processed separately. Possession never carries over half-time.

**Why N = 10.** Raising N from 1 to 10 cuts full-match possessions steeply, from ~676 to ~320. What it removes is mostly fake switches: balls passing near an opponent, deflections, challenges that don't win the ball. Beyond 10 the curve flattens, so higher N mainly merges possessions that may be real. N = 10 is the elbow, chosen from the full-match data, not the ground truth.

### 3.2 Dead-ball detectors

| Detector | Rule in plain words | Full-match output |
|---|---|---|
| **Throw-ins** (v4) | Ball at or beyond a touchline (within 3 m) or missing, with a sign it went out (missing ≥ 3 s, or tracked beyond the line), lasting ≥ 3 s. Short inside blips (< 2 s) are joined over | 45 spells, 10.9 min |
| **Goals** (v5) | Each GOALS tag is paired with its Kick Off (After Goal) tag. D starts at the first missing-ball run (≥ 3 s) whose last position was within 15 m of a goal line. D ends when the ball reappears after the kick-off tag opens, waiting for the kick if it reappears still | 4 spells, 9.6 min |
| **Goal kicks** (v6) | A still ball (< 2 m/s for ≥ 2 s) within 6 m of a goal line and inside the six-yard-box width. D ends at the kick (> 6 m/s for 3 frames). D starts at the most recent "ball went out over that line" event in the 30 s before | 6 spells, 1.5 min |
| **Corners** (v7) | Each corner tag sets where to look. D ends when the ball reappears (visible ≥ 1 s) within 15 s of the tag. D starts at the ball going out near a goal line in the 30 s before, following a long missing run back to its real start | 10 of 12 tags, 5.3 min |
| **Fouls / free kicks** (v8) | See 3.3 | 38 spells, 14.6 min |

**Overlaps.** Throw-ins and corners overlapped on two stoppages (1,413 frames), because the throw-in rule also fires on balls going out near a corner flag. Both mark the same dead ball, so the labels are unaffected.

### 3.3 Foul / free-kick rule (v8)

The hardest stoppage: there is no tag for most fouls (only 5 free-kick tags against roughly 27 fouls in a typical match), and the ball is often visible and still during live build-up too. The rule uses the fact that **a whistle stops both teams**:

1. **Slow:** both teams' 5 s rolling mean speed is below **1.6 m/s**, requiring each team separately, not the all-player average.
2. **Long enough:** the slow run lasts at least **5 s**.
3. **Restart:** within 2 s after the slow run, *both* teams' mean speed over the next 3 s is at least **0.5 m/s higher** than over the last 5 s of the slow run. This is a relative rise, so calm restarts, such as short free kicks, count too.
4. **Restart frame:** the ball kick closest to the speed-up (within 3 s). If the ball is missing, the speed-up frame is used.
5. **D** = start of the slow run → frame before the restart.
6. **Specialist detectors win:** any foul spell touching a D from the other four detectors is dropped, because those have better boundaries.

### 3.4 Tried and dropped

- **v3: missing-ball window.** The first dead-ball rule marked `D` wherever the ball was missing on at least 75% of the next 25 s. It caught none of the 4 public dead spells, because two of them are fouls with the ball visible throughout. It also ran past restarts (e.g. ~2 s past the kick-off) and lowered spellF1 from 0.4216 to 0.4000. I replaced it with one detector per type of stoppage (v4–v8), each starting again from v2.
- **An early foul rule.** All players' average speed below 1.7 m/s for 8 s. It fired on 45% of graded frames, far more than fouls account for. Ball-based filters (ball speed, return to the restart spot, time at the spot before the kick) and phase tags could not separate fouls from calm live play, so it was replaced by the v8 rule.

## 4. Results (public 5 minutes, local scorer)

| Version | What it adds | spellF1 | macroF1 | D F1 |
|---|---|---|---|---|
| v1 | Nearest player within 1.5 m | 0.2545 | 0.5075 | 0.00 |
| v2 | Sticky possession, N = 10 | 0.4216 | 0.5183 | 0.00 |
| v3 | Missing-ball window (dropped) | 0.4000 | 0.5148 | 0.00 |
| v4–v5 | + throw-ins, + goals | 0.4216 | 0.5183 | 0.00 |
| v6 | + goal kicks | 0.4541 | 0.6563 | 0.38 |
| v7 | + corners | 0.4541 | 0.6563 | 0.38 |
| **v8 (submitted)** | **+ fouls / free kicks** | **0.4889** | **0.8228** | **0.82** |
| Reference | `baseline_smooth125` | 0.4242 | 0.5812 | 0.15 |

## 5. Limitations

1. The initial sticky possession model is quite basic and only determines possession by the distance between the ball and the closest player. It can’t tell from the tracking data if a certain possession is ‘under control’. The ball being within 1.5m radius doesn’t account for ball possession from a deflection, tackle, or aerial duel, which isn’t ‘under control’. The N claims in a row tries to mitigate this, but it is a fixed parameter that doesn’t change according to game state. It also would absorb real possessions with fewer than 10 claims into the next possession.

2. Possession is not updated while the ball is untracked (about 41% of graded frames). With no ball there are no claims, so the model keeps the team that had possession when the ball disappeared. If the other team wins the ball during the gap, the change is placed late, after the ball returns and that team makes 10 claims, or missed entirely if possession changes back first. I did not estimate the ball’s position during gaps, for example from player movement.

3. Dead ball from a foul is hard to detect using the data signals available. Play slowing down is not just due to fouls - it is often due to moving from high speed phases to slower, build up phases. The reverse is also true for when players speed up. The foul detector will also kick in too late, as the dead ball spell should start before the players slow down, when the referee has blown his whistle.

## 6. Reproduction

Repository: https://github.com/nicolascrifasi1603/back-of-the-neural-net

```bash
git clone https://github.com/nicolascrifasi1603/back-of-the-neural-net.git
cd back-of-the-neural-net
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned versions
# put the match data bundle in data/  (not included; not redistributable)
python make_submission.py                # writes back-of-the-neural-net.csv
python evaluation/validate_submission.py back-of-the-neural-net.csv   # organisers' validator
```

`make_submission.py` is the whole pipeline in one file: the v8 path from my exploration notebook, with the plots and checks removed. It has no randomness and no training, so running it on the provided data reproduces the submitted file exactly. `data/` is excluded in `.gitignore`.

## 7. Disclosure

Below is everything I used during development beyond the code and the data provided by the organisers.

**Hand-marked events.** These were used to check rules, and in the throw-in case to choose starting parameters:
- 10 throw-ins (start and end times), used to set the throw-in rule's starting values and to check it (9/10 found)
- 4 goal stoppages, to check the goal rule's boundaries
- 7 corners, to plot what a corner looks like in the data before writing the corner rule

**Video checks** (reviewing what the rule had found, not labelling):
- 9 slow-player candidates from an early foul rule (4 fouls, 5 live play), to see why candidates failed. This version was replaced by v8
- 3 goal-kick tags identified as back-passes, not goal kicks; the goal-kick rule was not changed to fit them
- 2 individual clips (goal kick 93:18.6, corner 101:27.5) to understand detector misses

**Public ground truth.** Used for scoring and to plot the four public dead spells. One design choice was influenced by it: the foul rule's restart check originally required both teams to exceed 1.6 m/s, and rejected public spell 2. I changed it to a relative rise (+0.5 m/s) on the football reasoning that short free kicks are taken calmly, but I only noticed the problem because of the public spell. The +0.5 value was not tuned to make spell 2 pass.

**Web data.** Public match statistics (11 corners, 3 goals), to interpret the tag counts. Not used in any rule.

**AI assistance.** I used Claude (Anthropic) as a coach throughout: to question my reasoning, explain concepts, and help write and debug code. The ideas, decisions and checks are mine.

No label in the submitted file was written or corrected by hand.

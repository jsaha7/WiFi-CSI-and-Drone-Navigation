# WiFi CSI torso-position classification (Widar3.0 subset)

## 1. Background

WiFi CSI (Channel State Information) gives per-subcarrier, per-antenna
measurements of how a wireless channel is shaped by its environment, rather
than a single RSSI number. Three lines of prior work frame this project:

- **SpotFi** (Kotaru et al.) estimates angle-of-arrival from multipath
  components using CSI phase across a calibrated, physically fixed antenna
  array, then separates the direct path from reflections. The angle
  estimate depends on knowing the array's geometry and orientation exactly.
- **DeepFi** (Wang et al.) is fingerprint-based: a deep network learns a
  mapping from CSI amplitude to location during an offline survey of a
  static environment, then matches new readings against that fingerprint
  online. The fingerprint is tied to the physical conditions — furniture,
  hardware placement, orientation — present when it was recorded.
- **Ma et al.'s WiFi sensing survey** frames CSI-based sensing as detection,
  recognition, or estimation problems, and identifies generalization across
  environments, hardware, and subjects as the field's open problem rather
  than a solved one.

All three assume a **static, calibrated antenna array** observing a person
or object moving through a fixed environment. This project asks a narrower
question that sits inside that assumption: from Widar3.0 CSI, can a person's
torso **position** be classified from amplitude features alone, and how much
does that classifier depend on things (orientation, gesture, receiver count)
that a moving platform couldn't hold constant?

**Where this breaks down for a moving drone** — the assignment's Part 2
question — comes down to which of SpotFi/DeepFi's fixed assumptions a drone
violates simultaneously:

| Assumption in the static case | What a drone changes |
| --- | --- |
| Antenna array orientation is fixed and known | Roll/pitch/yaw rotate the array's frame continuously and are rarely known precisely enough in-flight |
| The reflecting/moving object is the target | On a drone, the *sensor* moves and the environment is the (mostly static) target — this is closer to a different problem (moving-receiver localization) than to Widar3's moving-person setup |
| The fingerprint/geometry is recorded once, applies later | Vibration and motor operation could perturb the channel independent of position, and this dataset has no motor-on/off comparison to isolate that |
| Packet timing and antenna spacing are stable | Both degrade with vibration and airframe flex |
| The measurement is roughly 2D (floor position) | A drone adds height and attitude as label dimensions this dataset never had |

None of this dataset's results test any of the drone-specific rows above —
it's a fixed-transmitter, fixed-receiver, moving-*person* setup, and every
number below should be read with that limit in mind.

## 2. Dataset and target

A subset of Widar3.0, session `20181128`, user6:

| | Value |
| --- | --- |
| Files | 1800 |
| Gestures | 1 (Push&Pull), 2 (Sweep), 3 (Clap), 4 (Draw-O) |
| Positions | 1, 2, 3 |
| Orientations | 1–5 |
| Repetitions | 1–5 |
| Receivers | r1–r6 |

Gesture codes are specific to the `20181128` session's bundled README (not
the public dataset homepage, which numbers them differently for this date):
1 Push&Pull, 2 Sweep, 3 Clap, 4 Draw-O, 5 Draw-Zigzag, 6 Draw-N. This subset
covers the first four.

**Filename convention** (confirmed against the actual files):
`user{u}-{gesture}-{position}-{orientation}-{repetition}-r{receiver}.dat`

**Target**: torso **position** (3 classes). Orientation, repetition, and
gesture are used as held-out factors, not targets — chance accuracy for this
3-class problem is 0.33, which is exactly what the dummy baseline gets.

**Confirmed CSI format** (checked empirically, not assumed from
documentation): Intel 5300 NIC, 3 rx antennas, 1 tx antenna, 30 subcarriers,
~1000–1450 packets per file, 100% nonzero/finite amplitude at this antenna
configuration. Parsed with
[`csiread.Intel`](https://github.com/citysu/csiread).

**Limits of this subset**: one user, one room, one recording session, and
only 3 of the dataset's 5 torso positions. No cross-user, cross-room, or
cross-session claim is supported by this data. The label is the human's
position — the WiFi transmitter and receivers are stationary throughout.

## 3. Method

- **Feature per file**: mean and std of amplitude, per subcarrier per
  antenna (30 x 3 x 2 = 180 features for one receiver; concatenated to 1080
  features when all 6 receivers are used together).
- **Models**: `DummyClassifier` (most-frequent baseline), logistic
  regression, random forest. See `code/experiment.py`.
- **Primary split**: hold out repetition 5, train on repetitions 1–4.
- **Stress test 1 — orientation**: leave-one-orientation-out, 5 folds.
- **Stress test 2 — gesture**: leave-one-gesture-out — train the position
  classifier on 3 of the 4 gestures and test on the 4th, entirely unseen
  gesture.
- **Within-gesture baseline**: repetition-holdout accuracy computed
  separately per gesture, as a same-gesture comparison point for the
  gesture-holdout numbers.
- Every run writes `metrics.json` (full manifest: feature version, receiver,
  seed, trial counts, dataset note), `split.json`, `predictions.csv`,
  `confusion_matrix.png`, `amplitude_examples.png`.

## 4. Results

**Receiver r1 only** (`results/`):

| Model | Held-out condition | Test trials | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- |
| dummy | repetition 5 | 60 | 0.333 | 0.167 |
| logistic regression | repetition 5 | 60 | 1.000 | 1.000 |
| random forest | repetition 5 | 60 | 0.900 | 0.901 |
| random forest | orientation 1 held out | 60 | 0.950 | 0.950 |
| random forest | orientation 2 held out | 60 | 0.967 | 0.967 |
| random forest | orientation 3 held out | 60 | 0.900 | 0.899 |
| random forest | orientation 4 held out | 60 | 0.850 | 0.843 |
| random forest | orientation 5 held out | 60 | 0.850 | 0.850 |
| random forest | **gesture 1 held out** | 75 | 0.920 | 0.919 |
| random forest | **gesture 2 held out** | 75 | 0.853 | 0.853 |
| random forest | **gesture 3 held out** | 75 | 0.973 | 0.973 |
| random forest | **gesture 4 held out** | 75 | 0.760 | 0.757 |
| random forest | within-gesture 1, repetition 5 held out | 15 | 1.000 | 1.000 |
| random forest | within-gesture 2, repetition 5 held out | 15 | 1.000 | 1.000 |
| random forest | within-gesture 3, repetition 5 held out | 15 | 0.867 | 0.866 |
| random forest | within-gesture 4, repetition 5 held out | 15 | 1.000 | 1.000 |

**All 6 receivers concatenated** (`results_allreceivers/`):

| Model | Held-out condition | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| random forest | repetition 5 | 1.000 | 1.000 |
| random forest | orientation 1–5 held out (5 folds) | 0.883–1.000 | 0.883–1.000 |
| random forest | gesture 1–4 held out (4 folds) | 0.973–1.000 | 0.973–1.000 |

See `results/confusion_matrix.png`, `results/amplitude_examples.png`, and
the equivalents under `results_allreceivers/` for the corresponding plots.

## 5. Interpretation

- **Repetition-holdout accuracy is consistently high** in both receiver
  configurations — the CSI amplitude features clearly separate positions
  when orientation and gesture are shared between train and test. This is
  the easiest split and, on its own, overstates how well this would work in
  practice.
- **Orientation-holdout accuracy stays fairly high with a single receiver
  (0.85–0.97)**. With only 3 position classes (chance = 0.33), there's more
  room for a classifier to land on the right answer even when orientation
  shifts the features — this result should not be read as "orientation
  doesn't matter," just that a 3-class problem is more forgiving of
  orientation-driven feature shift than a larger label set would be.
- **Gesture-holdout accuracy stayed high (0.76–0.97) with a single
  receiver**, meaning the position signal in r1's amplitude features is not
  purely an artifact of one specific gesture's motion — it transfers
  reasonably well to gestures never seen in training. Gesture 4 (Draw-O)
  held out the worst (0.76), consistent with it being a more complex, less
  repetitive motion than Push&Pull or Clap — a hypothesis, not something
  confirmed by this data.
- **All-receiver accuracy is near-perfect everywhere, including under
  orientation- and gesture-holdout.** This is flagged, not celebrated: with
  one user, one room, and 15–25 test trials per fold, spatial diversity
  across 6 receivers genuinely could resolve most of the ambiguity a single
  receiver has — but a clean ~1.00 across nearly every stress test is also
  exactly the pattern you'd see from subtle leakage (e.g., receiver-specific
  calibration artifacts correlating with position because of how the
  session was physically recorded) or from the task simply being too easy
  at this scale. Distinguishing those two explanations would need a
  shuffled-label control and/or a second session or user, neither of which
  is available here.

## 6. What this does and doesn't show

This shows that simple amplitude statistics carry enough information to
classify torso position within a single static session, that gesture and
orientation are meaningful — if moderate, at 3 classes — stress tests for a
single receiver, and that multi-receiver fusion's near-perfect scores need
scrutiny rather than automatic trust. It does not show cross-user, cross-
room, or cross-session generalization, and — per the drone-assumptions table
in Section 1 — it does not show anything about a moving sensor platform:
every measurement here was taken with a stationary transmitter and stationary
receivers observing a moving person.

## 7. How to reproduce

```bash
pip install -r requirements.txt
cd code
python experiment.py --data-dir ../data --receiver 1   --out-dir ../results
python experiment.py --data-dir ../data --receiver all --out-dir ../results_allreceivers
```

`data/` is not committed to this repo (see `.gitignore`) — place the 1800
`.dat` files (gestures 1–4, positions 1–3, session 20181128, user6) there
before running. The full Widar3.0 `20181128` session is available from the
[official Dataport download folder](https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/);
for any new session, re-confirm the antenna configuration with the nrx/ntx
sweep documented in `code/parse_widar.py` before trusting the defaults used
here.

Every run's exact configuration (feature version, receiver, seed, file
counts, gestures/positions present) is recorded in that run's
`metrics.json` — results in this README were read directly from those
files, not retyped from memory.

"""
WiFi CSI -> torso position classification (Widar3.0, session 20181128,
gestures 1-4, positions 1-3 subset).

Usage:
    python experiment.py --data-dir ../data --receiver 1 --out-dir ../results
    python experiment.py --data-dir ../data --receiver all --out-dir ../results

What this does (matches assignment Part 2):
  1. Inventories .dat files and parses labels from filenames.
  2. Extracts simple per-file amplitude features (mean/std per subcarrier per antenna).
  3. Trains a dummy baseline, logistic regression, and random forest to predict
     torso POSITION, holding out one repetition as the test set.
  4. Runs a harder stress test: leave-one-orientation-out, to see whether the
     classifier generalizes across human orientation or just memorizes it.
  5. Saves metrics, predictions, split membership, and plots to --out-dir.

Honesty notes baked into the code, not just the README:
  - Every skipped/unparseable file is counted and reported, not silently dropped.
  - The exact feature version, receiver(s) used, and split are written to
    manifest.json so a result can't be quietly reproduced differently later.
  - Nothing here claims to do drone localization -- see README Part 7.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from parse_widar import inventory, load_csi_amplitude

FEATURE_VERSION = "v1-meanstd-persubcarrier"


def extract_features_for_file(path, nrxnum=3, ntxnum=1):
    """One row of features per .dat file: per-subcarrier, per-antenna mean and
    std of amplitude across all packets. 30 subcarriers x 3 antennas x 2 stats
    = 180 features. Simple on purpose -- the assignment doesn't ask for more."""
    amp = load_csi_amplitude(path, nrxnum=nrxnum, ntxnum=ntxnum)  # (packets, 30, 3, 1)
    amp = amp[..., 0]  # drop the singleton tx axis -> (packets, 30, 3)
    mean = amp.mean(axis=0).ravel()  # (30*3,)
    std = amp.std(axis=0).ravel()
    return np.concatenate([mean, std])


def build_feature_table(trials, receiver_choice, cache_path, nrxnum, ntxnum):
    """Extract (or load cached) features for the requested receiver(s).
    receiver_choice: int (single receiver) or 'all' (concat across r1..r6,
    only for trial_ids that have all 6 receivers present)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[features] loading cached features from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        return pd.DataFrame(cached["rows"].tolist(), columns=cached["cols"])

    if receiver_choice == "all":
        by_trial = {}
        for t in trials:
            by_trial.setdefault(t.trial_id, {})[t.receiver] = t
        usable = {k: v for k, v in by_trial.items() if set(v) == {1, 2, 3, 4, 5, 6}}
        dropped = len(by_trial) - len(usable)
        if dropped:
            print(f"[features] dropping {dropped} trial(s) missing one or more receivers")
        rows = []
        for trial_id, recv_map in usable.items():
            feats = []
            for r in range(1, 7):
                feats.append(extract_features_for_file(recv_map[r].path, nrxnum, ntxnum))
            row_feat = np.concatenate(feats)
            t0 = recv_map[1]
            rows.append(dict(
                position=t0.position, orientation=t0.orientation,
                repetition=t0.repetition, gesture=t0.gesture,
                user=t0.user, receiver="all", features=row_feat,
            ))
    else:
        r = int(receiver_choice)
        subset = [t for t in trials if t.receiver == r]
        rows = []
        skipped = 0
        for t in subset:
            try:
                feat = extract_features_for_file(t.path, nrxnum, ntxnum)
            except Exception as e:
                skipped += 1
                print(f"[features] failed to parse {t.path.name}: {e}")
                continue
            rows.append(dict(
                position=t.position, orientation=t.orientation,
                repetition=t.repetition, gesture=t.gesture,
                user=t.user, receiver=r, features=feat,
            ))
        if skipped:
            print(f"[features] skipped {skipped} unparseable file(s) out of {len(subset)}")

    df = pd.DataFrame(rows)
    np.savez(cache_path,
             rows=np.array([{k: v for k, v in r.items()} for r in rows], dtype=object),
             cols=list(df.columns))
    return df


def run_repetition_holdout(df, test_repetition, seed, out_dir):
    X = np.stack(df["features"].values)
    y = df["position"].values
    is_test = df["repetition"].values == test_repetition

    scaler = StandardScaler().fit(X[~is_test])
    Xtr, Xte = scaler.transform(X[~is_test]), scaler.transform(X[is_test])
    ytr, yte = y[~is_test], y[is_test]

    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed),
    }

    results = {}
    preds_records = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        results[name] = {"accuracy": acc, "macro_f1": f1, "n_test": int(is_test.sum())}
        print(f"[repetition-holdout] {name}: acc={acc:.3f} macro_f1={f1:.3f} n_test={is_test.sum()}")
        if name == "random_forest":
            best_pred = pred
            cm = confusion_matrix(yte, pred, labels=sorted(np.unique(y)))

    for i, idx in enumerate(np.where(is_test)[0]):
        preds_records.append(dict(
            row_index=int(idx), true_position=int(yte[i]),
            pred_random_forest=int(best_pred[i]),
            orientation=int(df.iloc[idx]["orientation"]),
            repetition=int(df.iloc[idx]["repetition"]),
        ))
    pd.DataFrame(preds_records).to_csv(Path(out_dir) / "predictions.csv", index=False)

    labels = sorted(np.unique(y))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted position"); ax.set_ylabel("True position")
    ax.set_title(f"Random forest confusion matrix\n(held-out repetition {test_repetition})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return results, {"test_repetition": test_repetition,
                      "n_train": int((~is_test).sum()), "n_test": int(is_test.sum())}


def run_orientation_holdout(df, seed):
    X_all = np.stack(df["features"].values)
    y_all = df["position"].values
    orientations = sorted(df["orientation"].unique())
    fold_results = []
    for held_out in orientations:
        is_test = df["orientation"].values == held_out
        scaler = StandardScaler().fit(X_all[~is_test])
        Xtr, Xte = scaler.transform(X_all[~is_test]), scaler.transform(X_all[is_test])
        ytr, yte = y_all[~is_test], y_all[is_test]
        model = RandomForestClassifier(n_estimators=300, random_state=seed)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        fold_results.append({"held_out_orientation": int(held_out),
                              "accuracy": acc, "macro_f1": f1, "n_test": int(is_test.sum())})
        print(f"[orientation-holdout] held out orientation {held_out}: "
              f"acc={acc:.3f} macro_f1={f1:.3f}")
    return fold_results


def run_gesture_holdout(df, seed):
    """Leave-one-gesture-out: train position classifier on 3 of the 4 gestures,
    test on the held-out gesture. Only meaningful if df spans >1 gesture."""
    gestures = sorted(df["gesture"].unique())
    if len(gestures) < 2:
        print("[gesture-holdout] skipped: only one gesture present in this data")
        return []
    X_all = np.stack(df["features"].values)
    y_all = df["position"].values
    fold_results = []
    for held_out in gestures:
        is_test = df["gesture"].values == held_out
        scaler = StandardScaler().fit(X_all[~is_test])
        Xtr, Xte = scaler.transform(X_all[~is_test]), scaler.transform(X_all[is_test])
        ytr, yte = y_all[~is_test], y_all[is_test]
        model = RandomForestClassifier(n_estimators=300, random_state=seed)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        fold_results.append({"held_out_gesture": int(held_out),
                              "accuracy": acc, "macro_f1": f1, "n_test": int(is_test.sum())})
        print(f"[gesture-holdout] held out gesture {held_out}: "
              f"acc={acc:.3f} macro_f1={f1:.3f}")
    return fold_results


def run_within_gesture_baseline(df, test_repetition, seed):
    """For comparison with gesture-holdout: repetition-holdout accuracy computed
    separately within each gesture (same task, same-gesture train/test)."""
    results = {}
    for g in sorted(df["gesture"].unique()):
        sub = df[df["gesture"] == g]
        X = np.stack(sub["features"].values)
        y = sub["position"].values
        is_test = sub["repetition"].values == test_repetition
        if is_test.sum() == 0 or (~is_test).sum() == 0:
            continue
        scaler = StandardScaler().fit(X[~is_test])
        Xtr, Xte = scaler.transform(X[~is_test]), scaler.transform(X[is_test])
        ytr, yte = y[~is_test], y[is_test]
        model = RandomForestClassifier(n_estimators=300, random_state=seed)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        results[int(g)] = {"accuracy": acc, "macro_f1": f1, "n_test": int(is_test.sum())}
        print(f"[within-gesture baseline] gesture {g}: acc={acc:.3f} macro_f1={f1:.3f}")
    return results


def plot_amplitude_examples(df, out_dir, n_examples=5):
    """Sanity-check plot: raw feature vector (per subcarrier-antenna mean amplitude)
    for a few trials of different positions, same orientation/repetition where possible."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sample = df.groupby("position").first().reset_index()
    for _, row in sample.iterrows():
        mean_part = row["features"][:90]  # first half of feature vector = per-(subcarrier,antenna) mean
        ax.plot(mean_part, label=f"position {int(row['position'])}", alpha=0.8)
    ax.set_xlabel("subcarrier x antenna index"); ax.set_ylabel("mean amplitude")
    ax.set_title("Mean CSI amplitude by torso position (one example trial each)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "amplitude_examples.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--out-dir", default="../results")
    ap.add_argument("--receiver", default="1", help="'1'..'6' or 'all'")
    ap.add_argument("--nrxnum", type=int, default=3)
    ap.add_argument("--ntxnum", type=int, default=1)
    ap.add_argument("--test-repetition", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    trials = inventory(args.data_dir)
    print(f"[inventory] {len(trials)} labeled files found under {args.data_dir}")

    cache_path = out_dir / f"features_recv{args.receiver}_{FEATURE_VERSION}.npz"
    df = build_feature_table(trials, args.receiver, cache_path, args.nrxnum, args.ntxnum)
    print(f"[features] built table with {len(df)} rows, "
          f"{len(df['features'].iloc[0])} features/row")

    plot_amplitude_examples(df, out_dir)

    rep_results, split_info = run_repetition_holdout(df, args.test_repetition, args.seed, out_dir)
    orient_results = run_orientation_holdout(df, args.seed)
    gesture_results = run_gesture_holdout(df, args.seed)
    within_gesture_results = run_within_gesture_baseline(df, args.test_repetition, args.seed)

    manifest = {
        "feature_version": FEATURE_VERSION,
        "receiver": args.receiver,
        "nrxnum": args.nrxnum, "ntxnum": args.ntxnum,
        "seed": args.seed,
        "n_trials_total": len(trials),
        "n_rows_used": len(df),
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": f"Dataset is a user-uploaded subset from {args.data_dir} "
                f"({len(trials)} files) -- not the full Widar3.0 20181128 archive. "
                f"Gestures present: {sorted(set(t.gesture for t in trials))}, "
                f"positions present: {sorted(set(t.position for t in trials))}.",
        "split": split_info,
        "repetition_holdout_results": rep_results,
        "orientation_holdout_results": orient_results,
        "gesture_holdout_results": gesture_results,
        "within_gesture_repetition_baseline": within_gesture_results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nDone in {manifest['elapsed_seconds']}s. Results written to {out_dir}/")


if __name__ == "__main__":
    main()

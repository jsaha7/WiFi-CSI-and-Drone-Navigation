"""
Filename and CSI parsing helpers for the Widar3.0 20181128 session.

Filename convention (from the bundled dataset README, NOT the public homepage,
which disagrees on the first numeric field for this session):

    user6-1-2-3-4-r5.dat
       |  | | | |   |
       |  | | | |   receiver id (r1..r6)
       |  | | | repetition (1..5)
       |  | | human face orientation (1..5)
       |  | torso position (1..5)   <-- our prediction target
       |  gesture (1..6)
       user id

Gesture codes are session-specific (documented for 20181128 only):
  1 Push&Pull, 2 Sweep, 3 Clap, 4 Draw-O, 5 Draw-Zigzag, 6 Draw-N
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FNAME_RE = re.compile(
    r"user(?P<user>\d+)-(?P<gesture>\d+)-(?P<position>\d+)-"
    r"(?P<orientation>\d+)-(?P<repetition>\d+)-r(?P<receiver>\d+)\.dat$"
)


@dataclass
class TrialFile:
    path: Path
    user: int
    gesture: int
    position: int
    orientation: int
    repetition: int
    receiver: int

    @property
    def trial_id(self):
        # identifies a gesture-position-orientation-repetition combo,
        # independent of which receiver recorded it
        return (self.user, self.gesture, self.position,
                self.orientation, self.repetition)


def parse_filename(path: Path) -> TrialFile | None:
    m = FNAME_RE.search(path.name)
    if not m:
        return None
    g = {k: int(v) for k, v in m.groupdict().items()}
    return TrialFile(path=path, **g)


def inventory(data_dir: str) -> list[TrialFile]:
    """Walk data_dir, parse every .dat filename, skip anything that doesn't match."""
    files = sorted(Path(data_dir).rglob("*.dat"))
    trials, skipped = [], []
    for f in files:
        t = parse_filename(f)
        if t is None:
            skipped.append(f)
        else:
            trials.append(t)
    if skipped:
        print(f"[inventory] skipped {len(skipped)} files that didn't match the naming pattern")
    return trials


def load_csi_amplitude(path: Path, nrxnum=3, ntxnum=1, pl_size=0):
    """
    Read one Intel 5300 .dat file with csiread and return amplitude array
    of shape (n_packets, n_subcarriers, n_rx, n_tx).

    Confirmed empirically on this dataset subset (2026-09-09) by sweeping
    nrxnum/ntxnum and checking for 100% nonzero, finite amplitude: this
    session is nrxnum=3, ntxnum=1, 30 subcarriers, ~1000-1450 packets/file.
    Other Widar3 sessions may differ -- re-check with the same sweep before
    reusing this on a different date/session.
    """
    import csiread

    csi = csiread.Intel(str(path), nrxnum=nrxnum, ntxnum=ntxnum,
                         pl_size=pl_size, if_report=False)
    csi.read()
    amp = np.abs(csi.csi)  # (n_packets, n_subcarriers, n_rx, n_tx)
    return amp

#!/usr/bin/env python
"""Find Neocortex episodes with a 2 s depolarizing step and persistent spikes.

Edit the parameters below, then:
  conda activate pysynapse
  python scripts/find_persistent_activity.py

Writes database/persistent_activity.csv for SynapseQt File > Load Database.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = ROOT / "database"

# =============================================================================
# Parameters (edit these)
# =============================================================================
TRACES_ROOT = None  # None = settings.yaml startpath
PREFIX = "Neocortex"
DRUG = 1
STEP_MS = 2000.0  # depolarizing step duration
STEP_TOL_MS = 250.0  # allowed error on STEP_MS (e.g. 1750–2250 ms). Not the Rin pulse.
PERSIST_MS = 12000.0  # recording and last spike after depolarizing step
MIN_DEPOL_AMP = 20.0  # pA (or mV if VC waveform)
MIN_HYPER_AMP = 10.0  # abs pA for Rin pulse
REQUIRE_HYPERPOLARIZING = False  # True = Rin pulse required, not just preferred
MAX_STIM_PULSES = 2  # Rin (optional) + 2 s depol; skip PulseC / extra steps
MSH = -10.0  # min spike height [mV], Event Detection default
MSD = 1.0  # min spike distance [ms]
SPIKE_THRESH = 0.0
MIN_SPIKES = 8
N_BINS = 4
MIN_OCCUPIED_BINS = 3  # spike occupancy across the persist window
MAX_RATE_HZ = 80.0  # reject noise (too many peaks)
WAVE_MIN_DUR_MS = 80.0
WAVE_AMP_EPS = 8.0
HEADER_ONLY = False  # True = skip spike detection
MAX_FILES = 0  # 0 = no limit
N_WORKERS = 8  # 1 = sequential; keep modest on a network volume
CHUNKSIZE = 32  # files per worker batch
OUT_CSV = DATABASE_DIR / "persistent_activity.csv"
# =============================================================================

# Columns SynapseQt.loadDatabase reads (see rename_dict in SynapseQt.py)
SYNAPSE_DB_FIELDS = [
    "Show",
    "Cell",
    "Episode",
    "SweepWindow",
    "Drug",
    "DrugTime",
    "WCTime",
    "StimDescription",
]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from util.ImportData import NeuroData, separate_cell_episode  # noqa: E402
from util.MATLAB import getconsecutiveindex  # noqa: E402
from util.spk_util import spk_count, spk_window  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # noqa: N801
        def __init__(self, *args, **kwargs):
            pass

        def update(self, n=1):
            pass

        def close(self):
            pass

        def set_postfix_str(self, *args, **kwargs):
            pass

DAT_RE = re.compile(r"\.S\d+\.E\d+\.dat$", re.IGNORECASE)


def traces_root_from_settings():
    try:
        from app.config import settings
        key = sys.platform[:3]
        return Path(settings.startpath.get(key) or settings.startpath.get("dar"))
    except Exception:
        return Path("/Volumes/Assets/Edward/Data/Traces")


def parse_dac_pulses(protocol):
    """Pulses/steps from the LabWorld DAC header (same fields as generateDACdesc)."""
    pulses = []
    dac_data = getattr(protocol, "dacData", None) or []
    for ch, data in enumerate(dac_data):
        if data is None or len(data) < 24 or not data[0]:
            continue
        if data[1] and data[2]:
            start, end, amp = float(data[6]), float(data[7]), float(data[8])
            if end > start:
                pulses.append(
                    {"source": "step", "channel": ch, "start": start, "end": end, "amp": amp}
                )
        if data[14]:
            for name, si, ei, ai in (
                ("pulseA", 15, 16, 17),
                ("pulseB", 18, 19, 20),
                ("pulseC", 21, 22, 23),
            ):
                start, end, amp = float(data[si]), float(data[ei]), float(data[ai])
                if end > start:
                    pulses.append(
                        {
                            "source": name,
                            "channel": ch,
                            "start": start,
                            "end": end,
                            "amp": amp,
                        }
                    )
    return pulses


def plateaus_from_trace(Ss, ts, min_dur_ms=80.0, amp_eps=8.0):
    """Plateaus in the stimulus waveform, amplitude relative to early holding."""
    Ss = np.asarray(Ss, dtype=float)
    n_hold = max(1, int(50.0 / max(ts, 1e-6)))
    holding = float(np.median(Ss[:n_hold]))
    d = Ss - holding
    idx = np.flatnonzero(np.abs(d) >= amp_eps)
    if idx.size == 0:
        return [], holding
    min_n = max(1, int(min_dur_ms / max(ts, 1e-6)))
    blocks = getconsecutiveindex(idx, N=min_n)
    out = []
    for a, b in np.atleast_2d(blocks):
        i0, i1 = int(idx[int(a)]), int(idx[int(b)])
        out.append(
            {
                "source": "waveform",
                "channel": None,
                "start": i0 * ts,
                "end": (i1 + 1) * ts,
                "amp": float(np.median(d[i0 : i1 + 1])),
            }
        )
    return out, holding


def duration(p):
    return float(p["end"] - p["start"])


def pick_depolarizing_step(pulses, step_ms, tol_ms, min_amp):
    """Best ~2 s positive current pulse. Amp may be 0 in the header."""
    hits = []
    for p in pulses:
        dur = duration(p)
        if abs(dur - step_ms) > tol_ms:
            continue
        if p["amp"] < 0:
            continue
        if p["amp"] == 0 or p["amp"] >= min_amp:
            hits.append(p)
    if not hits:
        return None
    hits.sort(key=lambda p: (p["amp"] == 0, abs(duration(p) - step_ms), -p["amp"]))
    return hits[0]


def unique_pulses(pulses, amp_eps=1.0):
    """Non-zero pulses, de-duplicated by timing (DAC header + waveform can overlap)."""
    out = []
    seen = set()
    for p in pulses:
        if duration(p) <= 0 or abs(p["amp"]) < amp_eps:
            continue
        key = (round(p["start"] / 10.0), round(p["end"] / 10.0))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def is_depol_pulse(p, depol):
    return abs(p["start"] - depol["start"]) < 50 and abs(p["end"] - depol["end"]) < 50


def protocol_stim_ok(pulses, depol):
    """Allow at most Rin + depol. Reject a negative step after the depolarizing pulse (PulseC)."""
    real = unique_pulses(pulses)
    if len(real) > MAX_STIM_PULSES:
        return "too_many_stims"
    for p in real:
        if is_depol_pulse(p, depol):
            continue
        source = str(p.get("source", "")).lower()
        follows = p["start"] >= depol["end"] - 1
        if follows and p["amp"] < 0:
            return "neg_after_depol"
        if source == "pulsec" and p["amp"] < 0:
            return "neg_pulsec"
    return None


def pick_hyperpolarizing(pulses, depol, min_abs_amp):
    """Negative pulse of any duration, preferably before the depolarizing step."""
    cands = []
    for p in pulses:
        if p is depol:
            continue
        if p["amp"] > -min_abs_amp:
            continue
        if duration(p) <= 0:
            continue
        before = 0 if p["end"] <= depol["start"] + 1 else 1
        cands.append((before, -p["start"], p))
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def first_stim_trace(zdata):
    if zdata.Stimulus:
        ch = sorted(zdata.Stimulus)[0]
        return zdata.Stimulus[ch], ch, "Stimulus"
    if zdata.Current:
        ch = sorted(zdata.Current)[0]
        return zdata.Current[ch], ch, "Current"
    return None, None, None


def persistence_ok(spike_times, step_end, persist_ms, min_spikes, n_bins, min_occupied, max_rate_hz):
    post = np.asarray(spike_times, dtype=float)
    post = post[post >= step_end]
    if post.size < min_spikes:
        return False, post
    last = float(post[-1])
    if last < step_end + persist_ms:
        return False, post
    span_s = max((last - step_end) / 1000.0, persist_ms / 1000.0)
    if post.size / span_s > max_rate_hz:
        return False, post
    edges = np.linspace(step_end, step_end + persist_ms, n_bins + 1)
    occupied = sum(np.any((post >= edges[i]) & (post < edges[i + 1])) for i in range(n_bins))
    if occupied < min_occupied:
        return False, post
    return True, post


def make_hit(path, proto, depol, hyper, stim_source, extra=None):
    hit = {
        "path": path,
        "drug": int(proto.drug),
        "drug_name": proto.drugName,
        "duration_ms": float(proto.sweepWindow),
        "stim_desc": proto.stimDesc,
        "wc_time": float(proto.WCtime),
        "drug_time": float(proto.drugTime),
        "depol": depol,
        "hyper": hyper,
        "stim_source": stim_source,
        "voltage_channel": "",
        "n_spikes_post": "",
        "first_spike_ms": "",
        "last_spike_ms": "",
        "persist_span_ms": "",
    }
    if extra:
        hit.update(extra)
    return hit


def iter_dat_files(root, prefix):
    prefix = prefix.lower()
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().startswith(prefix):
                continue
            if not DAT_RE.search(name):
                continue
            yield os.path.join(dirpath, name)


def analyze_file(path):
    try:
        header = NeuroData(path, old=True, infoOnly=True)
    except Exception as exc:
        return None, f"header:{exc}"

    if int(header.Protocol.drug) != DRUG:
        return None, "drug"

    sweep_ms = float(header.Protocol.sweepWindow)
    pulses = parse_dac_pulses(header.Protocol)
    depol = pick_depolarizing_step(pulses, STEP_MS, STEP_TOL_MS, MIN_DEPOL_AMP)
    need_waveform = depol is None or depol["amp"] == 0

    if depol is not None and not need_waveform:
        if sweep_ms - depol["end"] < PERSIST_MS:
            return None, "too_short_after_step"
    elif sweep_ms < STEP_MS + PERSIST_MS:
        return None, "too_short_sweep"

    if HEADER_ONLY and depol is not None and depol["amp"] > 0:
        reason = protocol_stim_ok(pulses, depol)
        if reason:
            return None, reason
        hyper = pick_hyperpolarizing(pulses, depol, MIN_HYPER_AMP)
        if hyper is not None and hyper["start"] >= depol["end"] - 1:
            return None, "neg_after_depol"
        if REQUIRE_HYPERPOLARIZING and hyper is None:
            return None, "no_hyper"
        return make_hit(path, header.Protocol, depol, hyper, "DAC"), None

    try:
        z = NeuroData(path, old=True, infoOnly=False)
    except Exception as exc:
        return None, f"load:{exc}"

    ts = float(z.Protocol.msPerPoint)
    stim, stim_ch, stim_kind = first_stim_trace(z)
    if stim is not None:
        wave, _holding = plateaus_from_trace(
            stim, ts, min_dur_ms=WAVE_MIN_DUR_MS, amp_eps=WAVE_AMP_EPS
        )
        merged = list(pulses)
        if wave:
            merged.extend(wave)
            depol = pick_depolarizing_step(merged, STEP_MS, STEP_TOL_MS, MIN_DEPOL_AMP)
        hyper = pick_hyperpolarizing(pulses, depol, MIN_HYPER_AMP) if depol else None
        if hyper is None and depol is not None:
            hyper = pick_hyperpolarizing(wave, depol, max(MIN_HYPER_AMP, 20.0))
        stim_source = f"{stim_kind}{stim_ch}" if stim_ch else stim_kind
    else:
        hyper = pick_hyperpolarizing(pulses, depol, MIN_HYPER_AMP) if depol else None
        stim_source = "DAC"

    if depol is None:
        return None, "no_2s_depol"
    if depol["amp"] < MIN_DEPOL_AMP:
        return None, "weak_depol"
    if sweep_ms - depol["end"] < PERSIST_MS:
        return None, "too_short_after_step"
    stim_pulses = merged if stim is not None else pulses
    reason = protocol_stim_ok(stim_pulses, depol)
    if reason:
        return None, reason
    if hyper is not None and hyper["start"] >= depol["end"] - 1:
        hyper = None
        return None, "neg_after_depol"
    if REQUIRE_HYPERPOLARIZING and hyper is None:
        return None, "no_hyper"

    if HEADER_ONLY:
        return make_hit(path, z.Protocol, depol, hyper, stim_source), None

    if not z.Voltage:
        return None, "no_voltage"

    best = None
    for vch, Vs in z.Voltage.items():
        window = [depol["end"], None]
        vs_post = spk_window(Vs, ts, window, t0=0)
        _n, spike_time, _h = spk_count(
            vs_post, ts, msh=MSH, msd=MSD, threshold=SPIKE_THRESH
        )
        spike_time = np.asarray(spike_time, dtype=float) + depol["end"]
        ok, post = persistence_ok(
            spike_time,
            depol["end"],
            PERSIST_MS,
            MIN_SPIKES,
            N_BINS,
            MIN_OCCUPIED_BINS,
            MAX_RATE_HZ,
        )
        if not ok:
            continue
        rec = {
            "voltage_channel": vch,
            "n_spikes_post": int(post.size),
            "first_spike_ms": float(post[0]),
            "last_spike_ms": float(post[-1]),
            "persist_span_ms": float(post[-1] - depol["end"]),
        }
        if best is None or rec["n_spikes_post"] > best["n_spikes_post"]:
            best = rec

    if best is None:
        return None, "no_persistent_spikes"

    return make_hit(path, z.Protocol, depol, hyper, stim_source, extra=best), None


def cell_episode(path):
    name = os.path.basename(path)
    try:
        cell, epi = separate_cell_episode(name)
        return cell, epi
    except Exception:
        return os.path.splitext(name)[0], ""


def row_from_hit(hit):
    cell, epi = cell_episode(hit["path"])
    depol, hyper = hit["depol"], hit["hyper"]
    return {
        "Show": 1,
        "Cell": cell,
        "Episode": epi,
        "SweepWindow": int(round(hit["duration_ms"])),
        "Drug": hit["drug_name"] or "",
        "DrugTime": hit["drug_time"],
        "WCTime": hit["wc_time"],
        "StimDescription": hit["stim_desc"] or "",
        "path": hit["path"],
        "drug_level": hit["drug"],
        "has_hyperpolarizing": bool(hyper),
        "hyper_start_ms": "" if not hyper else f"{hyper['start']:.1f}",
        "hyper_end_ms": "" if not hyper else f"{hyper['end']:.1f}",
        "hyper_amp": "" if not hyper else f"{hyper['amp']:.1f}",
        "depol_start_ms": f"{depol['start']:.1f}",
        "depol_end_ms": f"{depol['end']:.1f}",
        "depol_amp": f"{depol['amp']:.1f}",
        "depol_source": depol.get("source", ""),
        "after_step_ms": f"{hit['duration_ms'] - depol['end']:.1f}",
        "stim_source": hit["stim_source"],
        "voltage_channel": hit["voltage_channel"],
        "n_spikes_post": hit["n_spikes_post"],
        "first_spike_ms": hit["first_spike_ms"],
        "last_spike_ms": hit["last_spike_ms"],
        "persist_span_ms": hit["persist_span_ms"],
    }


def main():
    root = Path(TRACES_ROOT) if TRACES_ROOT else traces_root_from_settings()
    if not root.is_dir():
        sys.exit(f"Traces root does not exist: {root}")

    print(f"Listing {PREFIX!r} .dat files under {root} ...")
    files = list(iter_dat_files(root, PREFIX))
    if MAX_FILES:
        files = files[:MAX_FILES]
    print(f"Scanning {len(files)} files  drug={DRUG}  workers={N_WORKERS}")

    hits = []
    n_seen = 0
    skip_counts = {}
    pbar = tqdm(
        total=len(files),
        unit="file",
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} scanned {postfix} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    def consume(hit, reason):
        nonlocal n_seen
        n_seen += 1
        if hit is None:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
        else:
            hits.append(hit)
        pbar.update(1)
        if hasattr(pbar, "set_postfix_str"):
            pbar.set_postfix_str(f"{len(hits)} hits", refresh=False)

    try:
        if N_WORKERS <= 1:
            for path in files:
                consume(*analyze_file(path))
        else:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                for hit, reason in pool.map(analyze_file, files, chunksize=CHUNKSIZE):
                    consume(hit, reason)
    finally:
        pbar.close()

    rows = [row_from_hit(h) for h in hits]
    rows.sort(
        key=lambda r: (
            [int(c) if c.isdigit() else c.lower() for c in re.split(r"([0-9]+)", str(r["Cell"]))],
            [int(c) if c.isdigit() else c.lower() for c in re.split(r"([0-9]+)", str(r["Episode"]))],
        )
    )
    out = Path(OUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = SYNAPSE_DB_FIELDS + [k for k in rows[0].keys() if k not in SYNAPSE_DB_FIELDS]
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        out.write_text("")

    cells = sorted({r["Cell"] for r in rows})
    print(f"Files scanned: {n_seen}")
    print(f"Hits: {len(rows)} episodes  /  {len(cells)} cells")
    print(f"Wrote {out.resolve()}")
    if skip_counts:
        top = sorted(skip_counts.items(), key=lambda kv: -kv[1])[:12]
        print("Skip reasons:", ", ".join(f"{k}={v}" for k, v in top))
    for r in rows[:15]:
        hyper = "hyper" if r["has_hyperpolarizing"] else "no-Rin"
        print(
            f"  {r['Cell']} {r['Episode']}  depol {r['depol_amp']}pA "
            f"{r['depol_start_ms']}-{r['depol_end_ms']}  {hyper}  "
            f"spikes={r['n_spikes_post']} span={r['persist_span_ms']}"
        )


if __name__ == "__main__":
    main()

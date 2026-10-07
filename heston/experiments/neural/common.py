"""Settings, cached data sets and paths shared by the PINN and DeepONet ladder scripts."""

import copy
import json
import sys
from time import perf_counter

import torch

from experiments.records import ROOT
from src.heston.model import load_domain
from src.neural.coordinates import sample_domain
from src.neural.reference import reference_values
from src.neural.training import labelled_data

CONFIG = ROOT / "experiments/config/neural"
STUDY = CONFIG / "study.json"
LADDERS = {"pinn": CONFIG / "pinn_ladder.json", "deeponet": CONFIG / "deeponet_ladder.json"}
SETTINGS_FILES = (STUDY, LADDERS["pinn"], LADDERS["deeponet"], ROOT / "experiments/config/domain.json")
RESULTS = ROOT / "results/neural/pinn_deeponet_v1"
LADDER = RESULTS / "ladder"
CACHE = ROOT / "artifacts/neural"      # rebuildable data sets and resume files

torch.set_default_dtype(torch.float64)


def keep_full_speed():
    """Opt this process out of Windows power throttling (EcoQoS), as in 2d/experiments/neural/common.py."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetProcessInformation.restype = wintypes.BOOL
    state = PowerThrottlingState(1, 0x1, 0x0)
    kernel.SetProcessInformation(kernel.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))


def load_settings():
    settings = json.loads(STUDY.read_text(encoding="utf-8"))
    settings["defaults"], settings["levels"] = {}, {}
    for family, path in LADDERS.items():
        ladder = json.loads(path.read_text(encoding="utf-8"))
        settings["defaults"][family] = ladder["defaults"]
        for name, level in ladder["levels"].items():
            settings["levels"][name] = dict(level, family=family)
    return settings


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def level_settings(settings, name):
    """Family defaults, then the ``base`` level's resolved settings, then the level's own."""
    level = settings["levels"][name]
    if "base" in level:
        resolved = level_settings(settings, level["base"])
    else:
        resolved = copy.deepcopy(settings["defaults"][level["family"]])
        resolved["computational_domain"] = copy.deepcopy(settings["computational_domain"])
    own = {k: v for k, v in level.items() if k not in ("base", "technique", "family")}
    resolved = merge(resolved, own)
    resolved.update(family=level["family"], technique=level["technique"])
    resolved.pop("cap", None)
    resolved["cap"] = level.get("cap", "default")
    return resolved


def ranges():
    return load_domain()["ranges"]


def _cached(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists():
        return torch.load(path, weights_only=False)
    value = build()
    torch.save(value, path)
    return value


def evaluation_set(kind):
    """Monitor, validation or short-validation contracts with twin reference values."""
    settings = load_settings()
    count = settings["monitor_contracts"] if kind == "monitor" else settings["validation_contracts"]
    seed = settings["seeds"][kind]
    tau = settings["short_validation_tau"] if kind == "short_validation" else None

    def build():
        x, put = sample_domain(ranges(), count, seed, tau)
        t = perf_counter()
        ref = reference_values(x, put)
        return dict(inputs=(x, put), reference=ref, label_seconds=perf_counter() - t)
    return _cached(f"{kind}_{count}_{seed}.pt", build)


def training_data(count, seed_name="data", short_share=0.0):
    settings = load_settings()
    seed = settings["seeds"][seed_name]
    tag = f"_short{short_share:g}" if short_share else ""
    return _cached(f"{seed_name}_{count}_{seed}{tag}.pt",
                   lambda: labelled_data(ranges(), count, seed, short_share))


def checkpoint_path(level, seed):
    return LADDER / f"{level}_seed{seed}.pt"

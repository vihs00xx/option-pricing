"""Settings, cached data sets and paths shared by the four-stock PINN and DeepONet ladder scripts."""

import copy
import json
import sys

import torch

from experiments.records import ROOT
from src.neural.coordinates import Stocks, protocol_domain, sample_domain
from src.neural.reference import reference_values
from src.neural.training import labelled_data

CONFIG = ROOT/"experiments/config/neural"
STUDY = CONFIG/"study.json"
LADDERS = {"pinn": CONFIG/"pinn_ladder.json", "deeponet": CONFIG/"deeponet_ladder.json"}
SETTINGS_FILES = (STUDY, LADDERS["pinn"], LADDERS["deeponet"])
RESULTS = ROOT/"results/neural/pinn_deeponet_v1"
STOCKS = Stocks(4)
LADDER = RESULTS/"ladder"
CACHE = ROOT/"artifacts/neural"      # rebuildable data sets and resume files; ignored by git

torch.set_default_dtype(torch.float64)


def keep_full_speed():
    """Opt this process out of Windows power throttling (EcoQoS).

    Windows runs windowless background processes on efficiency cores at low clock
    speeds, which slowed training about fifteen-fold here. The hint applies to this
    process only and ends with it; other platforms ignore it.
    """
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
    # ProcessPowerThrottling = 4; control the execution-speed bit and leave it off.
    state = PowerThrottlingState(1, 0x1, 0x0)
    kernel.SetProcessInformation(kernel.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))


def load_settings():
    """The study settings with both ladders: ``defaults`` per family and every level."""
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
    own = {k: v for k, v in level.items() if k not in ("base", "technique", "family")}
    resolved = merge(resolved, own)
    resolved.update(family=level["family"], technique=level["technique"],
                    computational_domain=settings["computational_domain"])
    return resolved


def domain():
    return protocol_domain(ROOT/load_settings()["protocol"])


def _cached(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE/name
    if path.exists():
        return torch.load(path, weights_only=False)
    value = build()
    torch.save(value, path)
    return value


def evaluation_set(kind):
    """Monitor, validation or short-validation contracts with reference values."""
    settings = load_settings()
    box, strike = domain()
    count = settings["monitor_contracts"] if kind == "monitor" else settings["validation_contracts"]
    seed = settings["seeds"][kind]
    tau = settings["short_validation_tau"] if kind == "short_validation" else None

    def build():
        x, put = sample_domain(STOCKS, box, strike, count, seed, tau)
        nodes = settings["reference_nodes"]["evaluation"]
        return dict(inputs=(x, put), reference=reference_values(STOCKS, x, put, nodes=nodes))
    return _cached(f"{kind}_{count}_{seed}.pt", build)


def training_data(count, seed_name="data"):
    """Reference-labelled contracts from the training domain (DeepONet data, PINN anchors)."""
    settings = load_settings()
    box, strike = domain()
    seed = settings["seeds"][seed_name]
    nodes = settings["reference_nodes"]["labels"]
    return _cached(f"{seed_name}_{count}_{seed}.pt", lambda: labelled_data(STOCKS, box, strike, count, seed, nodes))


def checkpoint_path(level, seed):
    return LADDER/f"{level}_seed{seed}.pt"

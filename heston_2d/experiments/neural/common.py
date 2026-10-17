"""Settings, cached data sets and paths shared by the Heston basket PINN and DeepONet ladder scripts.

The number of stocks is the protocol's (experiments/config/protocol/*.json)."""

import copy
import hashlib
import inspect
import json
import sys

import torch

from experiments.records import ROOT, digest
from src.neural import coordinates, reference
from src.neural.coordinates import Stocks, protocol_domain, protocol_path, sample_domain
from src.neural.reference import reference_values
from src.neural.training import labelled_data

CONFIG = ROOT/"experiments/config/neural"
STUDY = CONFIG/"study.json"
LADDERS = {"pinn": CONFIG/"pinn_ladder.json", "deeponet": CONFIG/"deeponet_ladder.json"}
SETTINGS_FILES = (STUDY, LADDERS["pinn"], LADDERS["deeponet"])
RESULTS = ROOT/"results/neural/pinn_deeponet_v1"
PROTOCOL = protocol_path()
STOCKS = Stocks(int(json.loads(PROTOCOL.read_text(encoding="utf-8"))["stocks"]))
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


def machine_load():
    """Other neural training runs on this machine and the CPU busy share over one second.

    Training times are compared only between runs made one at a time on an otherwise idle
    machine; train.py records this at the start and end of each run so that a contended
    timing shows in its summary instead of needing a correction factor afterwards.
    """
    try:
        import psutil
    except ImportError:
        return None
    me, others = psutil.Process().pid, 0
    for process in psutil.process_iter(["pid", "cmdline"]):
        modules = [a for a in process.info["cmdline"] or [] if a.startswith("experiments.neural.")]
        if process.info["pid"] != me and modules:
            others += 1
    return dict(other_training_runs=others, cpu_percent=psutil.cpu_percent(interval=1.0))


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
    return protocol_domain(PROTOCOL)


def label_sources():
    """What the labels and references are computed with: the sampler, the engine, the label code."""
    engine = {f"engine_{p.stem}": digest(p) for p in sorted((ROOT/"src/heston_basket").glob("*.py"))}
    return dict(coordinates=digest(coordinates.__file__), reference=digest(reference.__file__), **engine,
                labelled_data=hashlib.sha256(inspect.getsource(labelled_data).encode()).hexdigest())


def _cached(name, identity, build):
    """Load ``name`` from the cache if it was built from ``identity``, else build and save it.

    ``identity`` holds every setting and source the data set depends on. Its fingerprint
    is part of the file name, so a changed node count, domain or reference code builds a
    new file instead of loading old labels; the identity is also stored and checked.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    identity = json.loads(json.dumps(dict(identity, sources=label_sources())))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = CACHE/f"{name}_{fingerprint[:12]}.pt"
    if path.exists():
        value = torch.load(path, weights_only=False)
        if value.get("identity") != identity:
            raise RuntimeError(f"{path.name} was built from different settings or sources; delete it")
        return value
    value = dict(build(), identity=identity)
    torch.save(value, path)
    return value


def label_settings():
    settings = load_settings()
    return dict(points=settings["reference"]["labels"]["points"], steps=settings["reference"]["labels"]["steps"],
                seed=settings["seeds"]["label_normals"])


def evaluation_set(kind, workers=None):
    """Monitor, validation or short-validation contracts with engine references and standard errors."""
    settings = load_settings()
    box, strike = domain()
    count = settings["monitor_contracts"] if kind == "monitor" else settings["validation_contracts"]
    seed = settings["seeds"][kind]
    tau = settings["short_validation_tau"] if kind == "short_validation" else None
    rule = {k: v for k, v in settings["reference"]["evaluation"].items() if k != "note"}
    normals = settings["seeds"]["evaluation_normals"]+(0 if kind == "validation" else 100000*(1+(kind == "short_validation")))

    def build():
        x, put = sample_domain(STOCKS, box, strike, count, seed, tau)
        values, errors = reference_values(STOCKS, x, put, rule, normals, workers)
        return dict(inputs=(x, put), reference=values, standard_errors=errors)
    identity = dict(kind=kind, count=count, seed=seed, tau=tau, rule=rule, normals=normals, domain=box,
                    strike=strike, d=STOCKS.d)
    return _cached(f"{kind}_{count}_{seed}", identity, build)


def training_data(count, seed_name="data"):
    """Simulation-labelled contracts from the training domain (DeepONet data, PINN anchors)."""
    settings = load_settings()
    box, strike = domain()
    seed = settings["seeds"][seed_name]
    rule = label_settings()
    identity = dict(kind=seed_name, count=count, seed=seed, rule=rule, domain=box, strike=strike, d=STOCKS.d)
    return _cached(f"{seed_name}_{count}_{seed}", identity,
                   lambda: labelled_data(STOCKS, box, strike, count, seed, rule))


def checkpoint_path(level, seed):
    return LADDER/f"{level}_seed{seed}.pt"

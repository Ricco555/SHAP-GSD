"""
Regression tests for three "declared but never wired" config/CLI fixes,
the same defect class as the ``--config-template`` bug fixed earlier this
session (``scripts/run_dataset.py``).

Fix 1 — ``scripts/10_baselines.py``'s ``--gnn-lr`` flag was parsed into
    ``args.gnn_lr`` but never threaded into the ``kwargs`` dict passed to
    ``_run_baseline``/``_run_one_flow``, so it never reached
    ``run_gnnexplainer_with_model``'s ``lr`` parameter. Both defaults happen
    to be 0.01, so only a non-default ``--gnn-lr`` value exposed the bug.

Fix 2 — ``configs/default.yaml`` had a ``balancer.compute_weights_on:
    "original"`` key that no code ever read; class weights are (correctly,
    per CLAUDE.md CRITICAL INVARIANT #4) ALWAYS computed from the original
    unbalanced label distribution, so the key implied a toggle that never
    existed. Fixed by removing the key (not wiring it in, since wiring it in
    as a genuine toggle would let a caller violate the invariant) and adding
    an explanatory comment at the actual call site.

Fix 3 — ``src/model/trainer.py``'s two ``DataLoader(...)`` construction call
    sites (inside ``Trainer._run_epoch`` and ``Trainer._evaluate``) never
    passed ``num_workers``/``pin_memory``, so ``compute.num_workers`` /
    ``compute.pin_memory`` in config were always ignored in favor of
    PyTorch's own defaults. Pure performance knobs — no effect on any
    computed value.

Tests:
  1 — ``_run_one_flow`` in ``scripts/10_baselines.py`` threads a non-default
      ``gnnexplainer_lr`` through to ``run_gnnexplainer_with_model``'s ``lr``
      kwarg (direct functional proof, wrapper mocked out).
  2 — the wrapper default value (0.01) is also passed *explicitly* -- proves
      the value is threaded on purpose, not coincidentally equal to the
      wrapper's own default when the caller omits it.
  3 — ``main()``'s ``kwargs = dict(...)`` block in ``scripts/10_baselines.py``
      source actually assigns ``gnnexplainer_lr=args.gnn_lr`` (source-level
      guard against the wiring silently regressing at the CLI-to-kwargs
      boundary, mirroring test_tuning_config.py's landmine-removal checks).
  4 — ``configs/default.yaml`` no longer has a ``balancer.compute_weights_on``
      key.
  5 — no file in the repo (docs, configs, tests, src) references
      ``compute_weights_on`` any more.
  6 — ``TemporalBalancer.get_class_weights`` still computes from whatever
      label array it is given (i.e. behavior is caller-controlled by which
      array is passed in, not by any config toggle) — a light regression
      guard that the removed key's absence didn't break the method itself.
  7 — ``scripts/01_preprocess.py`` calls ``get_class_weights`` with the
      original (pre-balancing) ``train_data["labels"]`` array, not a
      balanced/resampled variable (source-level guard for INVARIANT #4).
  8 — ``Trainer._run_epoch``'s ``DataLoader`` construction receives
      ``num_workers``/``pin_memory`` sourced from ``cfg["compute"]``.
  9 — ``Trainer._evaluate``'s ``DataLoader`` construction receives the same.
  10 — missing ``compute`` block in cfg falls back to safe defaults
      (``num_workers=0``, ``pin_memory=False``) rather than raising.
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINES_SCRIPT = REPO_ROOT / "scripts" / "10_baselines.py"
DEFAULT_CFG_PATH = REPO_ROOT / "configs" / "default.yaml"
TRAINER_PATH = REPO_ROOT / "src" / "model" / "trainer.py"


# ══ Fix 1 — --gnn-lr wiring ════════════════════════════════════════════════

_BASELINES_MODULE = None


def _load_baselines_script():
    """Import ``scripts/10_baselines.py`` as a module (``scripts/`` is not a
    package), cached across tests in this session."""
    global _BASELINES_MODULE
    if _BASELINES_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "baselines_phase10_wiring", BASELINES_SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASELINES_MODULE = module
    return _BASELINES_MODULE


def test_gnn_lr_reaches_gnnexplainer_wrapper(monkeypatch) -> None:
    """A non-default gnnexplainer_lr passed into _run_one_flow reaches
    run_gnnexplainer_with_model's lr= kwarg, not the wrapper's own 0.01 default."""
    pytest.importorskip("torch")
    pytest.importorskip("dgl")
    module = _load_baselines_script()

    monkeypatch.setattr(
        "src.baselines.adapter.build_flow_context", lambda **kw: object()
    )

    captured: dict = {}

    def _fake_run_gnnexplainer_with_model(**kwargs):
        captured.update(kwargs)
        return {"stub": True}

    monkeypatch.setattr(
        "src.baselines.gnnexplainer_wrapper.run_gnnexplainer_with_model",
        _fake_run_gnnexplainer_with_model,
    )

    result = module._run_one_flow(
        baseline_name="gnnexplainer",
        global_eid=0,
        model=None,
        g_test=None,
        nsm=None,
        fs_test=None,
        sampler=None,
        feature_groups={},
        background=None,
        device=None,
        gnnexplainer_epochs=17,
        gnnexplainer_lr=0.0777,
    )

    assert result == {"stub": True}
    assert captured.get("lr") == 0.0777, (
        f"--gnn-lr must reach run_gnnexplainer_with_model's lr kwarg; got {captured}"
    )
    assert captured.get("epochs") == 17


def test_gnn_lr_default_still_reaches_wrapper(monkeypatch) -> None:
    """Even the default gnnexplainer_lr (0.01) must be explicitly passed
    through -- not merely coincide with the wrapper's own default."""
    pytest.importorskip("torch")
    pytest.importorskip("dgl")
    module = _load_baselines_script()

    monkeypatch.setattr(
        "src.baselines.adapter.build_flow_context", lambda **kw: object()
    )
    captured: dict = {}

    def _fake_run_gnnexplainer_with_model(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "src.baselines.gnnexplainer_wrapper.run_gnnexplainer_with_model",
        _fake_run_gnnexplainer_with_model,
    )

    module._run_one_flow(
        baseline_name="gnnexplainer",
        global_eid=0, model=None, g_test=None, nsm=None, fs_test=None,
        sampler=None, feature_groups={}, background=None, device=None,
    )
    assert "lr" in captured, "lr must be passed explicitly, not omitted"
    assert captured["lr"] == 0.01


def test_source_kwargs_dict_assigns_gnnexplainer_lr() -> None:
    """Source-level guard: main()'s `kwargs = dict(...)` block must assign
    gnnexplainer_lr=args.gnn_lr, mirroring the already-correct
    gnnexplainer_epochs=args.gnn_epochs pattern. Fails if this wiring is
    silently removed/regressed in a future edit."""
    tree = ast.parse(BASELINES_SCRIPT.read_text())

    kwargs_call = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            kw_names = {kw.arg for kw in node.keywords}
            if "gnnexplainer_epochs" in kw_names:
                kwargs_call = node
                break

    assert kwargs_call is not None, "could not locate the kwargs = dict(...) block"

    def _kwarg_source(name: str) -> str:
        for kw in kwargs_call.keywords:
            if kw.arg == name:
                return ast.unparse(kw.value)
        raise AssertionError(f"kwarg {name!r} not found in kwargs dict")

    assert _kwarg_source("gnnexplainer_epochs") == "args.gnn_epochs"
    assert _kwarg_source("gnnexplainer_lr") == "args.gnn_lr", (
        "gnnexplainer_lr=args.gnn_lr is missing from the kwargs dict -- "
        "the --gnn-lr flag is once again a no-op"
    )


# ══ Fix 2 — compute_weights_on removed, not wired ══════════════════════════

def test_compute_weights_on_key_removed_from_default_yaml() -> None:
    """The misleading, never-read balancer.compute_weights_on key is gone."""
    with open(DEFAULT_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert "compute_weights_on" not in cfg.get("balancer", {})


# Top-level dirs to prune when walking the filesystem directly (mirrors
# .gitignore, plus .git itself and other non-source dirs that are never
# meant to be read as text): the fallback used when no .git is present,
# e.g. on an HPC deployment that is a plain file copy, not a clone.
_WALK_EXCLUDE_DIRS = {
    ".git", "__pycache__", "visualisation", "artifacts", "feature_store",
    "graphs", "node_state_snapshots", "outputs", "runs", "external", "data",
    "datasets", "local", "backups", ".pytest_cache", "node_modules", ".venv",
}


def _walk_repo_like_git_ls_files(repo_root: Path) -> list[str]:
    """Filesystem-walk fallback for ``git ls-files`` when no ``.git`` is
    present (e.g. a plain-copy HPC deployment). Prunes the same dirs
    ``.gitignore`` excludes so multi-GB runtime artifacts are never read."""
    results = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _WALK_EXCLUDE_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            results.append(str(path.relative_to(repo_root)))
    return results


def test_compute_weights_on_not_referenced_anywhere() -> None:
    """No file under the repo (excluding .git/backups) still mentions the
    removed key -- nothing left dangling per the fix instructions."""
    import subprocess

    this_file = Path(__file__).resolve()

    # git-tracked files only -- avoids walking gitignored runtime dirs
    # (feature_store/, graphs/, artifacts/, outputs/, runs/, data/, ...),
    # some of which contain multi-GB memmaps not meant to be read as text.
    # Falls back to a filesystem walk (pruning the same dirs) when no .git
    # is present, e.g. on an HPC deployment that is a plain file copy.
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        tracked = _walk_repo_like_git_ls_files(REPO_ROOT)
    # specs/ is gitignored (local-only, per CLAUDE.md) so git ls-files misses
    # it -- check its markdown files explicitly since specs/02_graph_model.md
    # is a real place this key could reappear.
    specs_dir = REPO_ROOT / "specs"
    extra = [str(p.relative_to(REPO_ROOT)) for p in specs_dir.glob("*.md")] \
        if specs_dir.is_dir() else []

    hits = []
    for rel in tracked + extra:
        path = REPO_ROOT / rel
        if not path.is_file() or path.resolve() == this_file:
            continue
        try:
            text = path.read_text(errors="ignore")
        except (UnicodeDecodeError, OSError):
            continue
        if "compute_weights_on" in text:
            hits.append(str(path))
    assert hits == [], f"compute_weights_on still referenced in: {hits}"


def test_get_class_weights_uses_whatever_labels_it_is_given() -> None:
    """No config toggle governs the label source; the caller's array is
    authoritative. This is the invariant CLAUDE.md #4 requires: the caller
    (scripts/01_preprocess.py, scripts/fix_labels.py) must always pass the
    ORIGINAL unbalanced labels -- verified by their source at the call site,
    not by any config value."""
    from src.data.balancer import TemporalBalancer

    balancer = TemporalBalancer(seed=0)
    original_labels = np.array([0] * 90 + [1] * 10)
    balanced_labels = np.array([0] * 50 + [1] * 50)

    w_original = balancer.get_class_weights(original_labels, log_weights=False)
    w_balanced = balancer.get_class_weights(balanced_labels, log_weights=False)

    # Weights differ depending on which distribution is passed in -- proving
    # the method has no hidden config-driven override forcing one or the other.
    assert not np.allclose(w_original.numpy(), w_balanced.numpy())
    # Balanced (50/50) input yields ~equal weights; original (90/10) does not.
    assert np.allclose(w_balanced.numpy(), w_balanced.numpy()[0], atol=1e-3)
    assert w_original.numpy()[1] > w_original.numpy()[0]


def test_preprocess_script_calls_get_class_weights_with_original_labels() -> None:
    """scripts/01_preprocess.py must pass train_data['labels'] (original,
    pre-balancing) into get_class_weights -- source-level guard."""
    script = REPO_ROOT / "scripts" / "01_preprocess.py"
    text = script.read_text()
    idx = text.index("get_class_weights(")
    # The very next non-whitespace token after the call must be the original
    # labels array, not a balanced/resampled variable.
    snippet = text[idx: idx + 200]
    assert 'train_data["labels"]' in snippet


# ══ Fix 3 — compute.num_workers / compute.pin_memory wired into DataLoader ═

def _make_trainer(monkeypatch, num_workers: int, pin_memory: bool) -> "Trainer":
    """Construct a minimal real Trainer with a distinctive compute config."""
    torch = pytest.importorskip("torch")
    from src.model.trainer import Trainer

    class _DummyGraph:
        def num_edges(self) -> int:
            return 4

    cfg = {
        "model": {
            "batch_size": 2,
            "max_epochs": 1,
            "patience": 1,
            "fanouts": [2],
            "learning_rate": 0.001,
        },
        "compute": {"num_workers": num_workers, "pin_memory": pin_memory},
    }
    model = torch.nn.Linear(2, 2)
    trainer = Trainer(
        model=model,
        g_train=_DummyGraph(),
        g_val=_DummyGraph(),
        fs_train=None,
        fs_val=None,
        nsm=None,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    return trainer


class _RecordingLoader:
    """Stand-in for torch.utils.data.DataLoader: records kwargs, yields nothing."""

    calls: list = []

    def __init__(self, dataset, **kwargs) -> None:
        _RecordingLoader.calls.append(kwargs)

    def __iter__(self):
        return iter(())


def test_run_epoch_dataloader_receives_compute_config(monkeypatch) -> None:
    """Trainer._run_epoch's DataLoader(...) call site passes num_workers/
    pin_memory sourced from cfg['compute'], not PyTorch defaults."""
    torch = pytest.importorskip("torch")
    import src.model.trainer as trainer_mod

    _RecordingLoader.calls = []
    monkeypatch.setattr(trainer_mod, "DataLoader", _RecordingLoader)

    trainer = _make_trainer(monkeypatch, num_workers=3, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()
    trainer._run_epoch(
        g=None, fs=None, local_eids=np.arange(4), criterion=criterion, is_train=False
    )

    assert len(_RecordingLoader.calls) == 1
    kwargs = _RecordingLoader.calls[0]
    assert kwargs.get("num_workers") == 3
    assert kwargs.get("pin_memory") is True


def test_evaluate_dataloader_receives_compute_config(monkeypatch) -> None:
    """Trainer._evaluate's DataLoader(...) call site passes num_workers/
    pin_memory sourced from cfg['compute']."""
    torch = pytest.importorskip("torch")
    import src.model.trainer as trainer_mod

    _RecordingLoader.calls = []
    monkeypatch.setattr(trainer_mod, "DataLoader", _RecordingLoader)

    trainer = _make_trainer(monkeypatch, num_workers=5, pin_memory=False)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        trainer._evaluate(criterion=criterion)
    except ValueError:
        # The stub loader yields zero batches, so the downstream
        # f1_score(all_labels, all_preds) call sees empty arrays and raises.
        # Irrelevant here -- the DataLoader(...) call (and its kwargs) already
        # happened before that point, which is what this test verifies.
        pass

    assert len(_RecordingLoader.calls) == 1
    kwargs = _RecordingLoader.calls[0]
    assert kwargs.get("num_workers") == 5
    assert kwargs.get("pin_memory") is False


def test_trainer_defaults_compute_config_when_missing(monkeypatch) -> None:
    """A cfg with no 'compute' block at all must not raise -- falls back to
    safe defaults (num_workers=0, pin_memory=False)."""
    torch = pytest.importorskip("torch")
    import src.model.trainer as trainer_mod
    from src.model.trainer import Trainer

    class _DummyGraph:
        def num_edges(self) -> int:
            return 4

    cfg = {
        "model": {
            "batch_size": 2, "max_epochs": 1, "patience": 1, "fanouts": [2],
            "learning_rate": 0.001,
        },
        # no "compute" key at all
    }
    model = torch.nn.Linear(2, 2)
    trainer = Trainer(
        model=model, g_train=_DummyGraph(), g_val=_DummyGraph(),
        fs_train=None, fs_val=None, nsm=None, cfg=cfg, device=torch.device("cpu"),
    )
    assert trainer.num_workers == 0
    assert trainer.pin_memory is False

    _RecordingLoader.calls = []
    monkeypatch.setattr(trainer_mod, "DataLoader", _RecordingLoader)
    criterion = torch.nn.CrossEntropyLoss()
    trainer._run_epoch(
        g=None, fs=None, local_eids=np.arange(4), criterion=criterion, is_train=False
    )
    kwargs = _RecordingLoader.calls[0]
    assert kwargs.get("num_workers") == 0
    assert kwargs.get("pin_memory") is False

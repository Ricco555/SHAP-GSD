"""Shared statistical helpers for ``explore/evaluation/``.

Implements specs/64 Part II §15, which in turn implements Part I §5's
statistical-testing conventions **once**, here, so no ``evalNN_*.py`` module
calls a scipy test directly and no module invents its own correction-family
semantics.

Conventions this module enforces (specs/64 Part I §5):
  - Per-class paired comparisons across datasets/seeds/windows use the
    Wilcoxon signed-rank test (non-parametric; small per-class n, no
    justified normality assumption).
  - Comparisons across more than two groups use the Friedman test.
  - Multiple-comparison correction is Benjamini-Hochberg FDR, applied per
    comparison *family*, with the family named in the output.
  - alpha = 0.05 on **corrected** p-values, exposed as :data:`DEFAULT_ALPHA`,
    never hardcoded inline.
  - Every statistical output carries **both** ``p_raw`` and ``p_corrected``;
    neither ever overwrites the other.

Dependency policy (specs/64 D7): ``requirements.txt`` pins ``scipy`` but
neither ``statsmodels`` nor ``scikit-posthocs``. ``scipy.stats`` supplies
``spearmanr``, ``kendalltau``, ``wilcoxon`` and ``friedmanchisquare``;
Benjamini-Hochberg is hand-rolled below. The Nemenyi post-hoc needs the
studentized range distribution and is therefore **deferred, spec-only** — a
significant Friedman omnibus is reported with
``posthoc = "not_computed_no_dependency"`` rather than silently dropped.

Undefined results are ``NaN`` with a note, never ``0.0``. A rank correlation
against a constant vector is *undefined*, not "uncorrelated"; collapsing the
two is acceptable in a figure and not acceptable in a source-of-truth CSV
(specs/64 §14.5.1). Labelling *why* it is undefined is the calling module's
job (its ``rho_undefined_reason`` column); producing an honest ``NaN`` is
this module's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

#: Significance threshold applied to **corrected** p-values (Part I §5).
DEFAULT_ALPHA: float = 0.05

#: Minimum number of non-zero-difference pairs below which the exact Wilcoxon
#: signed-rank test cannot attain significance at alpha = 0.05 (2 ** -6 =
#: 0.0156 two-sided is the smallest attainable p at n = 6).
MIN_WILCOXON_PAIRS: int = 6

#: Emitted as the ``posthoc`` field of :class:`FriedmanResult` (specs/64 D7).
POSTHOC_UNAVAILABLE: str = "not_computed_no_dependency"


@dataclass(frozen=True)
class WilcoxonResult:
    """Outcome of a paired Wilcoxon signed-rank test.

    Attributes:
        statistic: The test statistic, ``NaN`` when not computed.
        p_raw: Uncorrected two-sided p-value, ``NaN`` when not computed.
        n_pairs: Number of finite pairs the test saw.
        note: ``""`` on a normal result, else why it was not computed.
    """

    statistic: float
    p_raw: float
    n_pairs: int
    note: str


@dataclass(frozen=True)
class FriedmanResult:
    """Outcome of a Friedman omnibus test across more than two groups.

    Attributes:
        statistic: The chi-square statistic, ``NaN`` when not computed.
        p_raw: Uncorrected p-value, ``NaN`` when not computed.
        n_groups: Number of groups compared.
        n_blocks: Number of complete blocks (paired observations) used.
        posthoc: :data:`POSTHOC_UNAVAILABLE` — Nemenyi is deferred (D7).
        note: ``""`` on a normal result, else why it was not computed.
    """

    statistic: float
    p_raw: float
    n_groups: int
    n_blocks: int
    posthoc: str
    note: str


@dataclass(frozen=True)
class RankAgreement:
    """Rank- and overlap-agreement between two ranked score vectors.

    Attributes:
        spearman_rho: Spearman's rho, ``NaN`` when undefined.
        p_raw: Uncorrected p-value for rho, ``NaN`` when undefined.
        kendall_tau: Kendall's tau, computed alongside at negligible cost and
            reported, never substituted for rho.
        jaccard_topk: Jaccard index of the two top-k index sets, ``NaN`` when
            both sets are empty.
        k: The k actually used (clamped to the vector length).
        n: Number of aligned elements compared.
        note: ``""`` on a normal result, else why a value is ``NaN``.
    """

    spearman_rho: float
    p_raw: float
    kendall_tau: float
    jaccard_topk: float
    k: int
    n: int
    note: str


def benjamini_hochberg(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction of a vector of p-values.

    Hand-rolled because scipy at this repo's pin does not supply it and
    specs/64 D7 refuses a new dependency for it. The procedure is: sort
    ascending, scale each by ``n / rank``, enforce monotonicity by a reverse
    cumulative minimum (walking from the *largest* p downward), clip at 1.0,
    restore the original order.

    NaN-safe: ``NaN`` entries pass through as ``NaN`` and are excluded from
    ``n``, so a family containing uncomputable tests is not silently
    over-corrected.

    Args:
        p: Raw p-values, possibly containing ``NaN``.

    Returns:
        A float array of corrected p-values, same length and order as ``p``.
    """
    raw = np.asarray(p, dtype=float)
    out = np.full(raw.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(raw)
    n = int(finite_mask.sum())
    if n == 0:
        return out

    values = raw[finite_mask]
    order = np.argsort(values, kind="mergesort")     # stable => deterministic
    ranks = np.arange(1, n + 1, dtype=float)
    scaled = values[order] * n / ranks
    # Monotonicity: q[i] = min(q[i], q[i+1], ..., q[n-1]) — accumulate the
    # minimum from the largest p-value downward, then flip back.
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)

    corrected = np.empty(n, dtype=float)
    corrected[order] = scaled
    out[finite_mask] = corrected
    return out


def apply_family(
    df: pd.DataFrame,
    family_name: str,
    p_col: str = "p_raw",
    corrected_col: str = "p_corrected",
    family_col: str = "correction_family",
) -> pd.DataFrame:
    """Write BH-corrected p-values and the family name onto a result frame.

    The single place ``p_corrected`` and ``correction_family`` are written, so
    no module can invent its own family semantics (Part I §5). The raw column
    is never overwritten.

    Args:
        df: Result frame carrying ``p_col``. Corrected in place on a copy.
        family_name: Name of the comparison family these p-values form, e.g.
            ``"dataset=UNSW;space=feature_group"``. This is what the emitted
            ``.md`` must state.
        p_col: Column holding raw p-values.
        corrected_col: Column to write corrected p-values into.
        family_col: Column to write ``family_name`` into.

    Returns:
        A copy of ``df`` with ``corrected_col`` and ``family_col`` set.

    Raises:
        KeyError: When ``p_col`` is absent from ``df``.
    """
    if p_col not in df.columns:
        raise KeyError(
            f"apply_family: frame has no {p_col!r} column (columns: "
            f"{list(df.columns)!r})"
        )
    out = df.copy()
    out[corrected_col] = benjamini_hochberg(out[p_col].to_numpy(dtype=float))
    out[family_col] = family_name
    return out


def is_significant(p_corrected: float, alpha: float = DEFAULT_ALPHA) -> bool:
    """Whether a **corrected** p-value clears the significance threshold.

    Args:
        p_corrected: A BH-corrected p-value; ``NaN`` is never significant.
        alpha: Threshold, defaulting to :data:`DEFAULT_ALPHA`.

    Returns:
        ``True`` when ``p_corrected`` is finite and below ``alpha``.
    """
    value = float(p_corrected)
    return bool(np.isfinite(value) and value < alpha)


def paired_wilcoxon(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    min_pairs: int = MIN_WILCOXON_PAIRS,
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank test with explicit not-computable outcomes.

    Pairs where either side is non-finite are dropped before testing. The test
    is refused — ``p_raw = NaN`` plus a note — when fewer than ``min_pairs``
    pairs remain (below which the exact test cannot attain significance at
    alpha = 0.05) or when every difference is zero (scipy raises).

    Args:
        a: First paired sample.
        b: Second paired sample, same length as ``a``.
        min_pairs: Minimum usable pairs; defaults to
            :data:`MIN_WILCOXON_PAIRS`.

    Returns:
        The :class:`WilcoxonResult`.

    Raises:
        ValueError: When ``a`` and ``b`` differ in length.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"paired_wilcoxon needs equal-length samples, got {arr_a.shape} "
            f"and {arr_b.shape}"
        )
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a, arr_b = arr_a[mask], arr_b[mask]
    n_pairs = int(arr_a.size)

    if n_pairs < min_pairs:
        return WilcoxonResult(
            statistic=float("nan"), p_raw=float("nan"), n_pairs=n_pairs,
            note=(f"not computed: {n_pairs} usable pairs < {min_pairs}, below "
                  f"which the exact signed-rank test cannot attain "
                  f"significance at alpha={DEFAULT_ALPHA}"),
        )
    if np.allclose(arr_a - arr_b, 0.0):
        return WilcoxonResult(
            statistic=float("nan"), p_raw=float("nan"), n_pairs=n_pairs,
            note="not computed: all paired differences are zero",
        )
    try:
        statistic, p_raw = stats.wilcoxon(arr_a, arr_b)
    except ValueError as exc:  # pragma: no cover - scipy edge cases
        return WilcoxonResult(
            statistic=float("nan"), p_raw=float("nan"), n_pairs=n_pairs,
            note=f"not computed: scipy.stats.wilcoxon raised: {exc}",
        )
    return WilcoxonResult(
        statistic=float(statistic), p_raw=float(p_raw), n_pairs=n_pairs, note="",
    )


def friedman(*groups: Sequence[float] | np.ndarray) -> FriedmanResult:
    """Friedman omnibus test across three or more paired groups.

    Blocks (columns) where any group is non-finite are dropped. The
    Nemenyi post-hoc is **not** run: it needs the studentized range
    distribution, i.e. a dependency this repo does not carry (specs/64 D7), so
    ``posthoc`` is always :data:`POSTHOC_UNAVAILABLE`. A significant omnibus
    is still reported — never silently omitted.

    Args:
        *groups: Three or more equal-length samples, one per group.

    Returns:
        The :class:`FriedmanResult`.

    Raises:
        ValueError: When fewer than three groups are given, or their lengths
            differ.
    """
    if len(groups) < 3:
        raise ValueError(
            f"friedman needs at least 3 groups, got {len(groups)}; use "
            f"paired_wilcoxon for a two-group comparison"
        )
    arrays = [np.asarray(g, dtype=float) for g in groups]
    lengths = {a.shape for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"friedman needs equal-length groups, got shapes {lengths}")

    stacked = np.vstack(arrays)
    mask = np.all(np.isfinite(stacked), axis=0)
    stacked = stacked[:, mask]
    n_blocks = int(stacked.shape[1])

    if n_blocks < 3:
        return FriedmanResult(
            statistic=float("nan"), p_raw=float("nan"), n_groups=len(arrays),
            n_blocks=n_blocks, posthoc=POSTHOC_UNAVAILABLE,
            note=(f"not computed: {n_blocks} complete blocks < 3 required by "
                  f"scipy.stats.friedmanchisquare"),
        )
    try:
        statistic, p_raw = stats.friedmanchisquare(*stacked)
    except ValueError as exc:  # pragma: no cover - scipy edge cases
        return FriedmanResult(
            statistic=float("nan"), p_raw=float("nan"), n_groups=len(arrays),
            n_blocks=n_blocks, posthoc=POSTHOC_UNAVAILABLE,
            note=f"not computed: scipy.stats.friedmanchisquare raised: {exc}",
        )
    return FriedmanResult(
        statistic=float(statistic), p_raw=float(p_raw), n_groups=len(arrays),
        n_blocks=n_blocks, posthoc=POSTHOC_UNAVAILABLE, note="",
    )


def jaccard(set_a: Iterable, set_b: Iterable) -> float:
    """Jaccard index of two iterables treated as sets.

    Args:
        set_a: First collection.
        set_b: Second collection.

    Returns:
        ``|A n B| / |A u B|``, or ``NaN`` when both are empty (the index is
        undefined there, not 1.0 and not 0.0).
    """
    a, b = set(set_a), set(set_b)
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def top_k_indices(scores: Sequence[float] | np.ndarray, k: int) -> list[int]:
    """Indices of the ``k`` largest finite scores, descending.

    Ties break by ascending index, so the result is deterministic
    (CLAUDE.md CODING STANDARDS 6).

    Args:
        scores: Score vector; non-finite entries are never selected.
        k: How many indices to return; clamped to the number of finite scores.

    Returns:
        Up to ``k`` indices into ``scores``.
    """
    arr = np.asarray(scores, dtype=float)
    finite = [i for i in range(arr.size) if np.isfinite(arr[i])]
    finite.sort(key=lambda i: (-arr[i], i))
    return finite[: max(0, int(k))]


def rank_agreement(
    vec_a: Sequence[float] | np.ndarray,
    vec_b: Sequence[float] | np.ndarray,
    k: int,
) -> RankAgreement:
    """Rank and top-k overlap agreement between two aligned score vectors.

    Spearman + Jaccard is the pairing specs/64 Part I §4.3 requires, for
    consistency with the cited cross-seed/cross-window stability check in the
    literature. Kendall's tau is computed alongside and reported, never
    substituted for rho.

    Undefined outcomes return ``NaN``, never ``0.0``: a constant input vector
    makes rho undefined (there is no variation to correlate), and that is a
    materially different statement from "these two rankings are uncorrelated".

    Args:
        vec_a: First score vector.
        vec_b: Second score vector, same length as ``vec_a`` and aligned to it
            element-by-element (same feature group / node at each index).
        k: Top-k size for the Jaccard overlap; clamped to the vector length.

    Returns:
        The :class:`RankAgreement`.

    Raises:
        ValueError: When the two vectors differ in length.
    """
    arr_a = np.asarray(vec_a, dtype=float)
    arr_b = np.asarray(vec_b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"rank_agreement needs aligned, equal-length vectors, got "
            f"{arr_a.shape} and {arr_b.shape}"
        )
    k_eff = int(min(max(0, k), arr_a.size))
    jac = jaccard(top_k_indices(arr_a, k_eff), top_k_indices(arr_b, k_eff))

    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    a, b = arr_a[mask], arr_b[mask]
    n = int(a.size)

    if n < 3:
        return RankAgreement(
            spearman_rho=float("nan"), p_raw=float("nan"),
            kendall_tau=float("nan"), jaccard_topk=jac, k=k_eff, n=n,
            note=f"rho undefined: only {n} aligned finite elements (<3)",
        )
    if np.all(a == a[0]) or np.all(b == b[0]):
        which = "a" if np.all(a == a[0]) else "b"
        return RankAgreement(
            spearman_rho=float("nan"), p_raw=float("nan"),
            kendall_tau=float("nan"), jaccard_topk=jac, k=k_eff, n=n,
            note=(f"rho undefined: input vector {which} is constant, so it has "
                  f"no rank variation to correlate (this is NOT the same as "
                  f"rho = 0)"),
        )

    rho_res = stats.spearmanr(a, b)
    tau_res = stats.kendalltau(a, b)
    return RankAgreement(
        spearman_rho=float(rho_res.statistic),
        p_raw=float(rho_res.pvalue),
        kendall_tau=float(tau_res.statistic),
        jaccard_topk=jac,
        k=k_eff,
        n=n,
        note="",
    )

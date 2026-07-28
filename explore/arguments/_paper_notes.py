"""Loader for optional, local-only paper-strategy notes.

The REVIEWER CHALLENGE / COUNTER-ARGUMENT / PAPER SECTION PLACEMENT sections
that argument scripts append to their `.txt` output are internal paper-writing
strategy (anticipated reviewer objections and how the article answers them),
not part of the public, reproducible analysis. That content lives in
``local/`` (git-ignored, same convention as CLAUDE.md/specs/) rather than in
these tracked scripts.

On a fresh clone (no ``local/paper_arguments/<STEM>.txt`` present), scripts
using this loader simply omit that section from their output — the CAPTION
section (the actual computed, reproducible figure caption) is unaffected.
"""

from pathlib import Path

_NOTES_DIR = Path(__file__).resolve().parent.parent.parent / "local" / "paper_arguments"


def load_paper_notes(stem: str, **fmt_vars) -> list[str]:
    """Return local-only paper-strategy note lines for `stem`, formatted with fmt_vars.

    Args:
        stem: script STEM, used to locate ``local/paper_arguments/<stem>.txt``.
        fmt_vars: values substituted into ``{name}``-style placeholders in the
            note text (e.g. a computed rate the note's argument references).

    Returns:
        The note's lines, or ``[]`` if the local-only file does not exist.
    """
    path = _NOTES_DIR / f"{stem}.txt"
    if not path.exists():
        return []
    raw = path.read_text()
    if fmt_vars:
        try:
            raw = raw.format(**fmt_vars)
        except (KeyError, IndexError):
            pass  # fall back to the unformatted note rather than crashing a figure script
    return raw.splitlines()

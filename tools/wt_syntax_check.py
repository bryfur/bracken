#!/usr/bin/env python3
"""Compile-check a bracken worktree without building a tree.

The one blessed build tree (`/home/bryan/git/bracken/build`) already carries
9.5 GB of built dependencies; a fresh `build/` in a git worktree would rebuild
SDL_shadercross/DXC from scratch. This script instead reuses that tree's
`compile_commands.json` flags, rewrites the source roots at the worktree, and
runs every translation unit through `-fsyntax-only` in parallel. It links
nothing and touches no build directory, so several worktrees can be checked
concurrently and none of them can disturb the shared tree.

New files an agent added are not in `compile_commands.json`; they get a
synthesized command copied from a template entry for the same target
(engine library, test suite, or example app).

Usage:
    tools/wt_syntax_check.py <worktree-root> [--only substr ...] [--jobs N]

Exit code is non-zero if any translation unit fails to compile.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REF_ROOT = Path("/home/bryan/git/bracken")
REF_DB = REF_ROOT / "build" / "compile_commands.json"


def load_entries() -> list[dict]:
    with open(REF_DB) as f:
        return json.load(f)


def is_ours(path: str) -> bool:
    """A source we own, as opposed to a fetched dependency's."""
    return path.startswith(f"{REF_ROOT}/src/") or path.startswith(f"{REF_ROOT}/examples/")


def rewrite(text: str, wt: Path) -> str:
    """Point the source roots at the worktree, leaving _deps paths alone."""
    for sub in ("src", "examples"):
        text = text.replace(f"{REF_ROOT}/{sub}", f"{wt}/{sub}")
    return text


def to_syntax_only(command: str, source: str) -> list[str]:
    argv = shlex.split(command)
    out: list[str] = []
    skip = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == "-o":
            skip = True
            continue
        if tok == "-c":
            continue
        if i and tok == source:
            continue
        out.append(tok)
    out += ["-fsyntax-only", source]
    return out


def classify(path: str) -> str:
    if path.endswith(".test.cpp"):
        return "tests"
    if path.endswith(".bench.cpp"):
        return "bench"
    if "/examples/" in path:
        return "example"
    return "lib"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("worktree")
    ap.add_argument("--only", nargs="*", default=[], help="substring filter on the source path")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    wt = Path(args.worktree).resolve()
    if not (wt / "src" / "bracken").is_dir():
        print(f"error: {wt} does not look like a bracken checkout", file=sys.stderr)
        return 2

    entries = load_entries()
    templates: dict[str, dict] = {}
    known: dict[str, dict] = {}
    for e in entries:
        src = e["file"]
        if not is_ours(src):
            continue
        known[src] = e
        templates.setdefault(classify(src), e)

    # Every source that exists in the worktree today, including files the
    # reference tree has never seen.
    jobs: list[tuple[str, list[str]]] = []
    for root in ("src", "examples"):
        for src in sorted((wt / root).rglob("*.cpp")):
            rel = str(src)
            ref = rel.replace(f"{wt}/", f"{REF_ROOT}/", 1)
            entry = known.get(ref) or templates.get(classify(rel))
            if entry is None:
                continue
            if args.only and not any(s in rel for s in args.only):
                continue
            cmd = rewrite(entry["command"], wt)
            jobs.append((rel, to_syntax_only(cmd, rel)))

    failures: list[tuple[str, str]] = []

    def run(job: tuple[str, list[str]]):
        src, argv = job
        p = subprocess.run(argv, capture_output=True, text=True, cwd=str(wt))
        return src, p.returncode, p.stderr

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for src, rc, err in ex.map(run, jobs):
            if rc != 0:
                failures.append((src, err))
            elif err.strip():
                # -Werror is on, so a clean exit with output is unusual; show it.
                print(f"[warn] {src}\n{err}", file=sys.stderr)

    print(f"checked {len(jobs)} translation units, {len(failures)} failed")
    for src, err in failures:
        print(f"\n===== FAILED: {src} =====\n{err}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

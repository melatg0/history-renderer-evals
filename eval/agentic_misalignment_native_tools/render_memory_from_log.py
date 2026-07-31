#!/usr/bin/env python
"""Render a prior-session memory asset from an Inspect native-tool eval log."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_memory import asset_from_log, write_asset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Inspect .eval log to read")
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON asset path",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--description", default=None)
    args = parser.parse_args()

    asset = asset_from_log(
        args.log,
        sample_index=args.sample_index,
        session_id=args.session_id,
        description=args.description,
    )
    out = write_asset(asset, args.out)
    print(f"Wrote {len(asset['events'])} native tool event(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

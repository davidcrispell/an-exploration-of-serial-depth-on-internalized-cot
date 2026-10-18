#!/usr/bin/env python3
"""Resume many small SCP uploads in parallel when a host resets large transfers."""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("remote_dir")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    files = sorted(path for path in args.local_dir.iterdir() if path.is_file())
    listing = subprocess.run(
        [
            "ssh",
            "-i",
            args.key,
            "-p",
            args.port,
            args.host,
            f"mkdir -p {args.remote_dir}; find {args.remote_dir} -maxdepth 1 -type f -printf '%f %s\\n'",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    remote_sizes = {}
    for line in listing.stdout.splitlines():
        name, size = line.rsplit(" ", 1)
        remote_sizes[name] = int(size)
    complete = {
        path.name for path in files if remote_sizes.get(path.name) == path.stat().st_size
    }
    pending = [path for path in files if path.name not in complete]
    print(f"existing={len(complete)} pending={len(pending)} total={len(files)}", flush=True)

    def upload(path: Path) -> tuple[str, bool, str]:
        command = [
            "scp",
            "-q",
            "-i",
            args.key,
            "-P",
            args.port,
            "-o",
            "ConnectTimeout=10",
            str(path),
            f"{args.host}:{args.remote_dir}/{path.name}",
        ]
        error = ""
        for _ in range(args.retries):
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                return path.name, True, ""
            error = result.stderr.strip()
        return path.name, False, error

    done = len(complete)
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(upload, path) for path in pending]
        for future in as_completed(futures):
            name, ok, error = future.result()
            if ok:
                done += 1
                if done % 50 == 0 or done == len(files):
                    print(f"uploaded={done}/{len(files)}", flush=True)
            else:
                failures.append((name, error))
    if failures:
        for name, error in failures:
            print(f"FAILED {name}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

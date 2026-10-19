#!/usr/bin/env python3
"""Resume a large Runpod download through verified small chunks."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=capture,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_file")
    parser.add_argument("local_file", type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--remote-chunk-dir", default="/tmp/codex-download-chunks")
    args = parser.parse_args()

    ssh = ["ssh", "-i", args.key, "-p", args.port, args.host]
    remote_name = Path(args.remote_file).name
    remote_dir = f"{args.remote_chunk_dir}/{remote_name}"
    manifest_command = (
        f"mkdir -p '{remote_dir}' && "
        f"split -b {args.chunk_bytes} -d -a 6 "
        f"'{args.remote_file}' '{remote_dir}/part-' && "
        f"cd '{remote_dir}' && sha256sum part-*"
    )
    manifest_result = run(ssh + [manifest_command], capture=True)
    manifest: list[tuple[str, str]] = []
    for line in manifest_result.stdout.splitlines():
        digest, name = line.split(maxsplit=1)
        manifest.append((name.removeprefix("*"), digest))
    if not manifest:
        raise SystemExit("remote chunk manifest is empty")

    part_dir = args.local_file.parent / f".{args.local_file.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    def valid(path: Path, digest: str) -> bool:
        if not path.is_file():
            return False
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest() == digest

    pending = [(name, digest) for name, digest in manifest if not valid(part_dir / name, digest)]
    print(f"existing={len(manifest) - len(pending)} pending={len(pending)} total={len(manifest)}", flush=True)

    def download(item: tuple[str, str]) -> tuple[str, bool, str]:
        name, digest = item
        destination = part_dir / name
        command = [
            "scp",
            "-q",
            "-i",
            args.key,
            "-P",
            args.port,
            "-o",
            "ConnectTimeout=10",
            f"{args.host}:{remote_dir}/{name}",
            str(destination),
        ]
        error = ""
        for _ in range(args.retries):
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and valid(destination, digest):
                return name, True, ""
            error = result.stderr.strip() or "downloaded chunk failed SHA256 verification"
        return name, False, error

    failures: list[tuple[str, str]] = []
    completed = len(manifest) - len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item) for item in pending]
        for future in as_completed(futures):
            name, ok, error = future.result()
            if ok:
                completed += 1
                if completed % 25 == 0 or completed == len(manifest):
                    print(f"downloaded={completed}/{len(manifest)}", flush=True)
            else:
                failures.append((name, error))
    if failures:
        for name, error in failures:
            print(f"FAILED {name}: {error}")
        raise SystemExit(1)

    temporary = args.local_file.with_suffix(args.local_file.suffix + ".assembling")
    hasher = hashlib.sha256()
    with temporary.open("wb") as output:
        for name, digest in manifest:
            part = part_dir / name
            if not valid(part, digest):
                raise SystemExit(f"chunk verification failed before assembly: {name}")
            with part.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    hasher.update(block)
    temporary.replace(args.local_file)

    remote_sha = run(
        ssh + [f"sha256sum '{args.remote_file}'"], capture=True
    ).stdout.split()[0]
    local_sha = hasher.hexdigest()
    if local_sha != remote_sha:
        raise SystemExit(f"final SHA256 mismatch: local={local_sha} remote={remote_sha}")
    print(f"sha256={local_sha} file={args.local_file}", flush=True)


if __name__ == "__main__":
    main()

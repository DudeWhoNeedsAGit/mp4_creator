#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("aeneas-align")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals", required=True, type=Path)
    ap.add_argument("--lyrics", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lang", default="eng")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        log.info(f"Using existing ASS: {args.out}")
        return

    if not args.vocals.exists():
        raise FileNotFoundError(args.vocals)
    if not args.lyrics.exists():
        raise FileNotFoundError(args.lyrics)

    task = (
        f"task_language={args.lang}"
        "|is_text_type=plain"
        "|os_task_file_format=json"  # <-- was 'ass'
        "|task_adjust_boundary_algorithm=percent"
        "|task_adjust_boundary_percent_value=50"
    )


    cmd = [
        "python", "-m", "aeneas.tools.execute_task",
        str(args.vocals),
        str(args.lyrics),
        task,
        str(args.out)
    ]

    log.info("Running Aeneas forced alignment")
    subprocess.run(cmd, check=True)
    log.info(f"Aligned ASS written to {args.out}")

if __name__ == "__main__":
    main()

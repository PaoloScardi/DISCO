# disco/cli.py
from __future__ import annotations
import argparse
from pathlib import Path

from cfactor.main import run  # you’ll create this function (see next section)

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="discocf", description="DISCO contrast factor calculator")
    p.add_argument("input", nargs="?", help="Input JSON file (e.g. contrast_factors.json)")
    p.add_argument("-o", "--output-dir", dest="output_dir", default=None,
                   help="Directory where outputs will be written (default: <input_dir>/output_fit)")
    args = p.parse_args(argv)

    input_path = Path(args.input) if args.input else None
    out_dir = Path(args.output_dir) if args.output_dir else None

    run(input_path=input_path, output_dir=out_dir)
    return 0

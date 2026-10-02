from __future__ import annotations

import argparse

from . import config as C
from . import data, export, pipeline


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mccast", description="Markov chain + cellular automata urban growth simulation")
    p.add_argument("--config", help="path to config.yaml (default: repo root)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="clip MapBiomas rasters to the study area (data/raw)")
    d.add_argument("--all-years", action="store_true", help="every year start..validation_end, not only the snapshots")
    d.add_argument("--overwrite", action="store_true")
    sub.add_parser("train", help="fit Markov matrix + suitability on the training period")
    sub.add_parser("validate", help="simulate train_end -> validation_end and score against observed")
    sub.add_parser("project", help="project validation_end -> horizon")
    sub.add_parser("export", help="write figures/tables to outputs/")
    sub.add_parser("all", help="download (snapshots) -> train -> validate -> project -> export")

    args = p.parse_args(argv)
    cfg = C.load(args.config)
    y = cfg["years"]
    snapshots = [y["start"], y["train_end"], y["validation_end"]]

    def fetch(years, overwrite=False):
        for yr in years:
            print(f"{yr}...")
            data.download_year(cfg, yr, overwrite)
            data.prepare_year(cfg, yr, overwrite)

    if args.cmd == "download":
        fetch(range(y["start"], y["validation_end"] + 1) if args.all_years else snapshots, args.overwrite)
    elif args.cmd == "train":
        pipeline.train(cfg)
    elif args.cmd == "validate":
        pipeline.validate(cfg)
    elif args.cmd == "project":
        pipeline.project(cfg)
    elif args.cmd == "export":
        export.export(cfg)
    elif args.cmd == "all":
        fetch(snapshots)
        pipeline.train(cfg)
        pipeline.validate(cfg)
        pipeline.project(cfg)
        export.export(cfg)


if __name__ == "__main__":
    main()

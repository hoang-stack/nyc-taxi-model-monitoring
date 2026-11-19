"""Command line: ``taximon <command>``.

Study (reproduces every number in the README):
    download   fetch the monthly TLC files and keep a 3% uniform sample of each
    study      train the 2019 H1 model, log every monitoring signal week by week
    evaluate   score each signal against the real error changes
    policies   replay retraining policies over 2020-2022
    report     figures + results/results.md

Operations:
    train      train on a date range and register a new model version
    promote    point models/current at a version (also how you roll back)
    versions   list registered versions
    serve      run the prediction service
    monitor    batch drift / error check over the service logs (exit 1 on alert)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="taximon", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", default="2019-01")
    d.add_argument("--end", default="2022-12")
    d.add_argument("--fraction", type=float, default=0.03)
    d.add_argument("--data", default="data")
    for name in ("study", "evaluate", "policies", "report"):
        s = sub.add_parser(name)
        s.add_argument("--data", default="data")
        s.add_argument("--results", default="results")
    t = sub.add_parser("train")
    t.add_argument("--start", required=True, help="first pickup week, e.g. 2022-07-04")
    t.add_argument("--end", required=True, help="last pickup week, e.g. 2022-12-26")
    t.add_argument("--version", default=None)
    t.add_argument("--data", default="data")
    t.add_argument("--models", default="models")
    t.add_argument("--promote", action="store_true")
    pr = sub.add_parser("promote")
    pr.add_argument("version")
    pr.add_argument("--models", default="models")
    v = sub.add_parser("versions")
    v.add_argument("--models", default="models")
    sub.add_parser("serve")
    m = sub.add_parser("monitor")
    m.add_argument("--logs", default="logs")
    m.add_argument("--model", default="models/current")
    m.add_argument("--psi-alert", type=float, default=0.2)
    m.add_argument("--mae-alert", type=float, default=None)
    args = p.parse_args(argv)

    if args.cmd == "download":
        from taximon.download import download

        download(args.start, args.end, args.fraction, args.data)
    elif args.cmd == "study":
        from taximon.study import run

        run(args.data, args.results)
    elif args.cmd == "evaluate":
        import pandas as pd

        from taximon.evaluate import score

        weekly = pd.read_csv(Path(args.results) / "weekly.csv", index_col=0, parse_dates=True)
        table = score(weekly)
        table.to_csv(Path(args.results) / "detectors.csv")
        print(table.round(3).to_string())
    elif args.cmd == "policies":
        from taximon.study import run_policies

        run_policies(args.data, args.results)
    elif args.cmd == "report":
        from taximon.report import build_report

        print(f"report written to {build_report(args.results)}")
    elif args.cmd == "train":
        import pandas as pd

        from taximon import data, registry
        from taximon.model import ETAModel

        df = data.clean(data.load(args.data))
        df = df[(df["week"] >= pd.Timestamp(args.start)) & (df["week"] <= pd.Timestamp(args.end))]
        zones = (
            pd.read_csv(Path(args.data) / "taxi_zone_lookup.csv")
            .set_index("LocationID")["Borough"]
            .to_dict()
        )
        model = ETAModel().fit(df, train_weeks=[args.start, args.end])
        path = registry.register(model, df, zones, args.models, args.version)
        print(f"registered {path} ({len(df):,} trips)")
        if args.promote:
            registry.promote(path.name, args.models)
            print(f"promoted {path.name}")
    elif args.cmd == "promote":
        from taximon import registry

        print(f"current -> {registry.promote(args.version, args.models).resolve().name}")
    elif args.cmd == "versions":
        from taximon import registry

        for row in registry.versions(args.models):
            print(("* " if row["current"] else "  ") + json.dumps(row, default=str))
    elif args.cmd == "serve":
        from taximon.service import main as serve

        serve()
    elif args.cmd == "monitor":
        from taximon import monitor_logs

        report = monitor_logs.run(args.logs, args.model, args.psi_alert, args.mae_alert)
        print(json.dumps(report, indent=1))
        sys.exit(1 if report["alerts"] else 0)


if __name__ == "__main__":
    main()

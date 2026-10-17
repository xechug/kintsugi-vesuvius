import argparse
import json

from vesuvius_code.smoke_test import config_from_json, run_smoke_test


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a local, audited Kintsugi diagnostic on Apple MPS"
    )
    parser.add_argument("--config", default="configs/kintsugi_local_baseline.json")
    parser.add_argument("--output-dir", default="runs/local_kintsugi")
    parser.add_argument("--steps", type=int, default=512)
    return parser


def main():
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    report = run_smoke_test(
        config_from_json(args.config, args.output_dir, args.steps)
    )
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "model": report["model"],
                "train_steps": report["train_steps"],
                "validation": report["validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

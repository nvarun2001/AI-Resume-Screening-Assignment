"""CLI entrypoint.

Usage:
    python main.py --input ./resumes --output ./output/results.json

Reads LLM/GitHub configuration from the environment (.env). If no LLM is
configured, or --no-llm is passed, the pipeline runs with deterministic
heuristics so it still produces a ranked result.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from resume_screener.config import Settings
from resume_screener.llm_adapter import LLMClient
from resume_screener.pipeline import Pipeline


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI resume screening and ranking system")
    parser.add_argument("--input", required=True, help="Directory containing resume files")
    parser.add_argument(
        "--output", default="./output/results.json", help="Path to write the JSON results"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM and score deterministically with heuristics",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable info logging")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env()
    client = None
    if not args.no_llm:
        if settings.llm_configured:
            client = LLMClient(settings)
        else:
            print(
                "Warning: LLM not configured in .env; using deterministic heuristics.",
                file=sys.stderr,
            )

    result = Pipeline(llm_client=client, settings=settings).run(args.input)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(exclude_none=True), indent=2), encoding="utf-8"
    )

    summary = result.summary
    print(
        f"Screened {summary.total_resumes} resume(s): "
        f"{summary.eligible} eligible, {summary.rejected} rejected, "
        f"{summary.failed} failed/unreadable."
    )
    print(f"Results written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI entrypoint. Usage: python main.py --input ./resumes --output ./output/results.json"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main() -> None:
    raise SystemExit("Pipeline not implemented yet.")


if __name__ == "__main__":
    main()

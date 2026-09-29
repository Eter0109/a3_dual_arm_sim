"""Print package compatibility facts without importing model weights or credentials."""

import argparse
from importlib.metadata import PackageNotFoundError, requires, version


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        prog="a3-sim inspect environment", description=__doc__,
    ).parse_args(argv)
    for name in ("lerobot", "transformers", "huggingface-hub", "peft", "torch"):
        try:
            print(name, version(name))
            for requirement in requires(name) or []:
                if any(key in requirement.lower() for key in (
                    "transformers", "huggingface", "smolvla", "pi05", "torchvision",
                )):
                    print(" ", requirement)
        except PackageNotFoundError:
            print(name, "not installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

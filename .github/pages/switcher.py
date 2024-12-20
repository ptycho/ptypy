"""Create/modify switcher.json to allow docs to switch between different versions."""

import json, os
from argparse import ArgumentParser
from pathlib import Path


def get_versions(root: str) -> list[str]:
    """Generate a list of versions."""
    versions = sorted([ f.name for f in os.scandir(root) if f.is_dir() ])
    print(f"Sorted versions: {versions}")
    return versions


def write_json(path: Path, repository: str, versions: list[str]):
    """Write the JSON switcher to path."""
    org, repo_name = repository.split("/")
    struct = [
        {"version": version, "url": f"https://{org}.github.io/{repo_name}/{version}/"}
        for version in versions
    ]
    text = json.dumps(struct, indent=2)
    print(f"JSON switcher:\n{text}")
    path.write_text(text, encoding="utf-8")


def main(args=None):
    """Parse args and write switcher."""
    parser = ArgumentParser(description="Make a versions.json file")
    parser.add_argument("root", type=Path, help="Path to root directory with all versions of docs")
    parser.add_argument("repository", help="The GitHub org and repository name: ORG/REPO")
    parser.add_argument("output", type=Path, help="Path of write switcher.json to")
    args = parser.parse_args(args)

    # Write the versions file
    versions = get_versions(args.root)
    write_json(args.output, args.repository, versions)


if __name__ == "__main__":
    main()
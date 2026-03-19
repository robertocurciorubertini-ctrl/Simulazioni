from pathlib import Path
import argparse
import re
import json

SIM_METADATA_PATTERN = re.compile(
    r"window\.SIM_METADATA\s*=\s*({.*?})\s*;",
    re.DOTALL
)

def find_html_files(root: Path, exclude_dirs: set[str] | None = None) -> list[Path]:
    if exclude_dirs is None:
        exclude_dirs = set()

    html_files = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in {".html", ".htm"}:
            continue

        # Controlla se il file è dentro una directory da escludere
        relative_parts = p.relative_to(root).parts
        if any(part in exclude_dirs for part in relative_parts[:-1]):
            continue

        html_files.append(p)

    return sorted(html_files)


def extract_metadata(path: Path):
    text = path.read_text(encoding="utf-8")
    match = SIM_METADATA_PATTERN.search(text)
    if not match:
        return None

    raw = match.group(1)

    # conversione semplice JS → JSON
    raw = re.sub(r'([{\s,])([a-zA-Z_]\w*)\s*:', r'\1"\2":', raw)
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)

    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".")
    parser.add_argument("--output", default="simulations.json")
    args = parser.parse_args()

    root = Path(args.input)

    # 🔴 QUI LA PARTE IMPORTANTE
    html_files = find_html_files(
        root,
        exclude_dirs={"theory", ".git", "__pycache__"}
    )

    simulations = []

    for file in html_files:
        meta = extract_metadata(file)
        if not meta:
            continue

        simulations.append(meta)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(simulations, f, indent=2, ensure_ascii=False)

    print(f"Trovate {len(simulations)} simulazioni")


if __name__ == "__main__":
    main()
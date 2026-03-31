#!/usr/bin/env python3
"""
build_catalog.py

Scansiona ricorsivamente una cartella di file HTML, cerca un blocco statico del tipo:

<head>
  <script>
  window.SIM_METADATA = {
    "title": "Conservazione dell'energia",
    "description": "Visualizza le trasformazioni tra energia cinetica e potenziale.",
    "icon": "♻️",
    "order": 8,
    "tags": ["energia", "energia cinetica"],
    "level": "base"
  };
  </script>
</head>

e aggiorna la sezione "simulations" di un file JSON esistente,
preservando "categories" e qualsiasi altra chiave già presente.

Caratteristiche principali:
- supporta sottocartelle;
- deduce la categoria dalla prima cartella del percorso, se il campo "category"
  non è presente nei metadati oppure se si usa --category-source folder;
- evita duplicati sia per "file" sia per "title";
- conserva chiavi extra già presenti nel JSON quando possibile.

Uso tipico:
    py build_catalog.py
    py build_catalog.py --category-source auto
    py build_catalog.py --category-source folder
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


METADATA_REGEX = re.compile(
    r"window\.SIM_METADATA\s*=\s*\{(?P<body>.*?)\}\s*;",
    re.DOTALL | re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggiorna la sezione 'simulations' di simulations.json leggendo i metadati dai file HTML."
    )
    parser.add_argument(
        "--input",
        default=".",
        help="Cartella da scandire ricorsivamente alla ricerca di file HTML. Default: cartella corrente.",
    )
    parser.add_argument(
        "--json",
        default="simulations.json",
        help="File JSON da aggiornare. Default: simulations.json",
    )
    parser.add_argument(
        "--category-source",
        choices=["auto", "metadata", "folder"],
        default="auto",
        help=(
            "Come determinare la categoria: "
            "'metadata' usa sempre category dai metadati, "
            "'folder' usa sempre la prima cartella del percorso, "
            "'auto' usa category se presente, altrimenti deduce dalla cartella."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Interrompe l'esecuzione al primo file con metadati invalidi.",
    )
    return parser.parse_args()


def find_html_files(root: Path) -> list[Path]:
    html_files = []
    excluded_dirs = {"theory", "nome_cartella"}

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in {".html", ".htm"}:
            continue

        relative_parts = p.relative_to(root).parts
        if any(part in excluded_dirs for part in relative_parts[:-1]):
            continue

        html_files.append(p)

    return sorted(html_files)


def extract_metadata_block(text: str) -> str | None:
    match = METADATA_REGEX.search(text)
    if not match:
        return None
    return "{" + match.group("body") + "}"


def js_object_to_json(js_text: str) -> str:
    js_text = re.sub(r"//.*?$", "", js_text, flags=re.MULTILINE)
    js_text = re.sub(r"/\*.*?\*/", "", js_text, flags=re.DOTALL)

    js_text = re.sub(
        r'([\{\s,])([A-Za-z_][A-Za-z0-9_\-]*)\s*:',
        r'\1"\2":',
        js_text,
    )

    # Non convertire automaticamente stringhe tra apici singoli.
    # Questa trasformazione rompe casi perfettamente validi come:
    # "forme d'onda"
    # quando l'apostrofo compare dentro una stringa già delimitata da doppi apici.

    js_text = re.sub(r",(\s*[}\]])", r"\1", js_text)
    return js_text


def normalize_order(value: Any, path: Path) -> int | float:
    try:
        if isinstance(value, bool):
            raise TypeError
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        if isinstance(value, str):
            n = float(value)
            return int(n) if n.is_integer() else n
        raise TypeError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Il campo 'order' deve essere numerico in '{path}'.") from exc


def title_case_from_slug(value: str) -> str:
    text = value.replace("-", " ").replace("_", " ").strip()
    return " ".join(word[:1].upper() + word[1:] for word in text.split())


def infer_category_from_path(relative_path: Path) -> str:
    parts = list(relative_path.parts)
    if len(parts) <= 1:
        return "Senza categoria"
    return title_case_from_slug(parts[0])


def choose_category(raw: dict[str, Any], relative_path: Path, mode: str) -> str:
    from_folder = infer_category_from_path(relative_path)
    from_metadata = str(raw.get("category", "")).strip()

    if mode == "folder":
        return from_folder
    if mode == "metadata":
        return from_metadata or from_folder
    return from_metadata or from_folder


def load_metadata_from_html(path: Path, root: Path, category_source: str) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8")
    block = extract_metadata_block(text)
    if block is None:
        return None

    json_like = js_object_to_json(block)

    try:
        raw = json.loads(json_like)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Metadati non validi in '{path}': {exc}. "
            f"Consiglio: usa chiavi e stringhe tra doppi apici e separa ogni riga con una virgola."
        ) from exc

    required = ["title", "description", "icon", "order"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(
            f"Metadati mancanti in '{path}': {', '.join(missing)}"
        )

    relative_path = path.relative_to(root)
    result = dict(raw)
    result["file"] = relative_path.as_posix()
    result["title"] = str(raw["title"]).strip()
    result["description"] = str(raw["description"]).strip()
    result["icon"] = str(raw["icon"]).strip()
    result["order"] = normalize_order(raw["order"], path)
    result["category"] = choose_category(raw, relative_path, category_source)
    result["tags"] = [str(tag).strip() for tag in raw.get("tags", []) if str(tag).strip()]
    result["level"] = str(raw.get("level", "")).strip()
    return result


def load_existing_json(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        return {"categories": [], "simulations": []}

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Il file JSON '{json_path}' non è valido: {exc}. "
            f"Controlla che inizi con '{{' oppure '[' e che non ci siano virgole mancanti."
        ) from exc

    if isinstance(data, dict):
        if "categories" not in data or not isinstance(data["categories"], list):
            data["categories"] = []
        if "simulations" not in data or not isinstance(data["simulations"], list):
            data["simulations"] = []
        return data

    if isinstance(data, list):
        return {"categories": [], "simulations": data}

    raise ValueError(
        f"Il file JSON '{json_path}' deve contenere un oggetto oppure una lista JSON alla radice."
    )


def _norm_title(value: Any) -> str:
    return str(value).strip().casefold()


def merge_simulations(existing: list[dict[str, Any]], extracted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    extracted_by_file: dict[str, dict[str, Any]] = {
        str(item["file"]): dict(item) for item in extracted if isinstance(item, dict) and "file" in item
    }
    extracted_titles: set[str] = {
        _norm_title(item.get("title", "")) for item in extracted_by_file.values()
    }

    merged: dict[str, dict[str, Any]] = {}

    for item in existing:
        if not isinstance(item, dict) or "file" not in item:
            continue

        file_key = str(item["file"])
        title_key = _norm_title(item.get("title", ""))

        if file_key in extracted_by_file:
            current = dict(item)
            current.update(extracted_by_file[file_key])
            merged[file_key] = current
            continue

        if title_key and title_key in extracted_titles:
            continue

        merged[file_key] = dict(item)

    for file_key, item in extracted_by_file.items():
        if file_key in merged:
            current = dict(merged[file_key])
            current.update(item)
            merged[file_key] = current
        else:
            merged[file_key] = dict(item)

    dedup_by_title: dict[str, dict[str, Any]] = {}
    for item in merged.values():
        title_key = _norm_title(item.get("title", ""))
        if title_key:
            dedup_by_title[title_key] = item
        else:
            dedup_by_title[f"__file__:{item.get('file','')}"] = item

    def sort_key(x: dict[str, Any]) -> tuple:
        category = str(x.get("category", "")).casefold()
        order = x.get("order", 10**9)
        try:
            order_num = float(order)
        except (TypeError, ValueError):
            order_num = 10**9
        title = str(x.get("title", "")).casefold()
        return (category, order_num, title)

    return sorted(dedup_by_title.values(), key=sort_key)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    json_path = Path(args.json).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Errore: la cartella di input non esiste o non è una directory: {input_dir}", file=sys.stderr)
        return 1

    html_files = find_html_files(input_dir)
    extracted: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in html_files:
        try:
            metadata = load_metadata_from_html(path, input_dir, args.category_source)
            if metadata is not None:
                extracted.append(metadata)
        except Exception as exc:
            if args.strict:
                print(f"Errore: {exc}", file=sys.stderr)
                return 1
            warnings.append(str(exc))

    if warnings:
        print("Avvisi durante la scansione:", file=sys.stderr)
        for w in warnings:
            print(f" - {w}", file=sys.stderr)

    try:
        data = load_existing_json(json_path)
    except Exception as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 1

    existing_simulations = data.get("simulations", [])
    if not isinstance(existing_simulations, list):
        print("Errore: la chiave 'simulations' del JSON deve contenere una lista.", file=sys.stderr)
        return 1

    data["simulations"] = merge_simulations(existing_simulations, extracted)

    try:
        write_json(data, json_path)
    except Exception as exc:
        print(f"Errore durante la scrittura del JSON: {exc}", file=sys.stderr)
        return 1

    print(f"Aggiornato '{json_path.name}' con {len(extracted)} simulazioni lette dai file HTML.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

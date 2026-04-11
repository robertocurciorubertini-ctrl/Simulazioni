#!/usr/bin/env python3
"""
generate_sim.py
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

CONFIG_SECTION_MARKER = "Simulazioni:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggiorna simulations.json leggendo i metadati dai file HTML e, opzionalmente, aggiorna un file di configurazione testuale."
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
        "--config",
        default="configurazione.txt",
        help=(
            "File di configurazione testuale da aggiornare sostituendo solo la sezione finale "
            "che parte da 'Simulazioni:'. Default: configurazione.txt"
        ),
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Non aggiorna il file di configurazione testuale.",
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
    excluded_dirs = {"theory"}

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


def strip_js_comments(js_text: str) -> str:
    js_text = re.sub(r"/\*.*?\*/", "", js_text, flags=re.DOTALL)
    js_text = re.sub(r"//.*?$", "", js_text, flags=re.MULTILINE)
    return js_text


def quote_unquoted_keys(js_text: str) -> str:
    """
    Converte chiavi JavaScript non quotate in chiavi JSON quotate,
    senza alterare testo interno alle stringhe.
    Esempio:
      { title: "abc", order: 2 }
    ->
      { "title": "abc", "order": 2 }
    """
    out = []
    i = 0
    n = len(js_text)
    in_string = False
    string_delim = ""
    escape = False

    while i < n:
        ch = js_text[i]

        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == string_delim:
                in_string = False
            i += 1
            continue

        if ch in ('"', "'"):
            in_string = True
            string_delim = ch
            out.append(ch)
            i += 1
            continue

        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (js_text[j].isalnum() or js_text[j] in "_-"):
                j += 1

            k = j
            while k < n and js_text[k].isspace():
                k += 1

            prev_nonspace_idx = len(out) - 1
            while prev_nonspace_idx >= 0 and out[prev_nonspace_idx].isspace():
                prev_nonspace_idx -= 1
            prev_nonspace = out[prev_nonspace_idx] if prev_nonspace_idx >= 0 else ""

            token = js_text[i:j]

            if k < n and js_text[k] == ":" and prev_nonspace in {"", "{", ",", "\n"}:
                out.append(f'"{token}"')
                i = j
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def normalize_single_quoted_strings(js_text: str) -> str:
    """
    Converte stringhe delimitate da apici singoli in stringhe JSON con doppi apici,
    lasciando inalterate quelle già con doppi apici.
    """
    out = []
    i = 0
    n = len(js_text)
    in_string = False
    string_delim = ""
    escape = False

    while i < n:
        ch = js_text[i]

        if not in_string:
            if ch == '"':
                in_string = True
                string_delim = '"'
                out.append(ch)
            elif ch == "'":
                in_string = True
                string_delim = "'"
                out.append('"')
            else:
                out.append(ch)
            i += 1
            continue

        if escape:
            if string_delim == "'":
                if ch == '"':
                    out.append('\\"')
                else:
                    out.append(ch)
            else:
                out.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\":
            escape = True
            out.append(ch)
            i += 1
            continue

        if ch == string_delim:
            out.append('"' if string_delim == "'" else ch)
            in_string = False
            i += 1
            continue

        if string_delim == "'" and ch == '"':
            out.append('\\"')
        else:
            out.append(ch)
        i += 1

    return "".join(out)


def js_object_to_json(js_text: str) -> str:
    js_text = strip_js_comments(js_text)

    # Primo tentativo: se è già JSON valido o quasi valido, preservalo il più possibile
    candidate = re.sub(r",(\s*[}\]])", r"\1", js_text).strip()
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    # Fallback: conversione da object literal JS a JSON
    js_text = normalize_single_quoted_strings(js_text)
    js_text = quote_unquoted_keys(js_text)
    js_text = re.sub(r",(\s*[}\]])", r"\1", js_text)
    return js_text.strip()


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


def is_wip_path(relative_path: Path) -> bool:
    parts = [part.casefold() for part in relative_path.parts]
    return len(parts) > 1 and parts[0] == "wip"


def choose_category(raw: dict[str, Any], relative_path: Path, mode: str) -> str:
    if is_wip_path(relative_path):
        return "WIP"

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
    wip_flag = is_wip_path(relative_path)

    result = dict(raw)
    result["file"] = relative_path.as_posix()
    result["title"] = str(raw["title"]).strip()
    result["description"] = str(raw["description"]).strip()
    result["icon"] = str(raw["icon"]).strip()
    result["order"] = normalize_order(raw["order"], path)
    result["category"] = choose_category(raw, relative_path, category_source)
    result["tags"] = [str(tag).strip() for tag in raw.get("tags", []) if str(tag).strip()]
    result["level"] = str(raw.get("level", "")).strip()
    result["isWip"] = wip_flag
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


def ensure_wip_category(data: dict[str, Any]) -> None:
    categories = data.get("categories", [])
    if not isinstance(categories, list):
        data["categories"] = []
        categories = data["categories"]

    has_wip = any(
        isinstance(cat, dict) and str(cat.get("name", "")).strip().casefold() == "wip"
        for cat in categories
    )

    if not has_wip:
        categories.append({
            "name": "WIP",
            "description": "Simulazioni in sviluppo o in fase di test.",
            "order": 999999
        })


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def category_order_map(data: dict[str, Any]) -> dict[str, tuple[int | float, str]]:
    mapping: dict[str, tuple[int | float, str]] = {}
    categories = data.get("categories", [])
    if not isinstance(categories, list):
        return mapping

    for idx, category in enumerate(categories):
        if not isinstance(category, dict):
            continue
        name = str(category.get("name", "")).strip()
        if not name:
            continue
        raw_order = category.get("order", 10**9 + idx)
        try:
            order = float(raw_order)
        except (TypeError, ValueError):
            order = float(10**9 + idx)
        mapping[name.casefold()] = (order, name)

    return mapping


def build_config_index_text(simulations: list[dict[str, Any]], data: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sim in simulations:
        if not isinstance(sim, dict):
            continue
        category = str(sim.get("category", "Senza categoria")).strip() or "Senza categoria"
        grouped.setdefault(category, []).append(sim)

    order_map = category_order_map(data)

    def category_sort_key(name: str) -> tuple[float, str]:
        if name.casefold() in order_map:
            return (order_map[name.casefold()][0], name.casefold())
        return (float(10**12), name.casefold())

    lines = ["Simulazioni:", ""]
    for category in sorted(grouped, key=category_sort_key):
        lines.append(f"{category}:")
        sims = grouped[category]

        def sim_sort_key(item: dict[str, Any]) -> tuple[float, str]:
            raw_order = item.get("order", 10**9)
            try:
                order = float(raw_order)
            except (TypeError, ValueError):
                order = float(10**9)
            title = str(item.get("title", "")).casefold()
            return (order, title)

        for sim in sorted(sims, key=sim_sort_key):
            title = str(sim.get("title", "")).strip()
            if title:
                lines.append(f"- {title}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def update_config_file(config_path: Path, simulations: list[dict[str, Any]], data: dict[str, Any]) -> None:
    marker = "Simulazioni:"
    new_section = build_config_index_text(simulations, data)

    if config_path.exists():
        original = config_path.read_text(encoding="utf-8")
    else:
        original = ""

    if marker in original:
        prefix = original.split(marker, 1)[0].rstrip()
        if prefix:
            updated = prefix + "\n\n" + new_section
        else:
            updated = new_section
    else:
        base = original.rstrip()
        if base:
            updated = base + "\n\n" + new_section
        else:
            updated = new_section

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(updated, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    json_path = Path(args.json).resolve()
    config_path = Path(args.config).resolve()

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

    if extracted:
        print(f"Simulazioni trovate: {len(extracted)}")
        for item in extracted:
            print(f" - {item.get('title', '<senza titolo>')}  [{item.get('file', '<percorso sconosciuto>')}]")
    else:
        print("Simulazioni trovate: nessuna")

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
    ensure_wip_category(data)

    try:
        write_json(data, json_path)
    except Exception as exc:
        print(f"Errore durante la scrittura del JSON: {exc}", file=sys.stderr)
        return 1

    if not args.no_config:
        try:
            update_config_file(config_path, data["simulations"], data)
            print(f"Aggiornato il file '{config_path.name}'.")
        except Exception as exc:
            print(f"Errore durante l'aggiornamento del file di configurazione: {exc}", file=sys.stderr)
            return 1

    print(f"Aggiornato il file '{json_path.name}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
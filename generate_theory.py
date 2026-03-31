#!/usr/bin/env python3
"""
Genera theory/theory.json scansionando automaticamente la cartella theory/.

Convenzione consigliata:
in ogni pagina teorica HTML inserisci un blocco come questo, preferibilmente nel <head>:

<script>
window.THEORY_METADATA = {
  title: "Moto rettilineo uniforme",
  description: "Definizione, legge oraria e rappresentazione grafica del moto rettilineo uniforme.",
  section: "Meccanica",
  sectionSlug: "meccanica",
  order: 1,
  level: "base",
  tags: ["cinematica", "velocità costante", "legge oraria"]
};
</script>

Il programma:
- cerca tutti i file .html dentro theory/
- estrae THEORY_METADATA
- costruisce theory.json ordinato per sezione e per argomento

Uso:
    python generate_theory_json.py
oppure:
    python generate_theory_json.py /percorso/alla/repository
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


METADATA_PATTERN = re.compile(
    r"window\.THEORY_METADATA\s*=\s*({.*?})\s*;",
    re.DOTALL
)

TRAILING_COMMA_PATTERN = re.compile(r",(\s*[}\]])")


def js_object_to_json(js_text: str) -> str:
    """
    Converte un oggetto JavaScript semplice in JSON valido.
    Supporta chiavi non quotate e virgole finali.
    """
    js_text = re.sub(r"/\*.*?\*/", "", js_text, flags=re.DOTALL)
    js_text = re.sub(r"//.*", "", js_text)

    js_text = re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        js_text
    )

    js_text = TRAILING_COMMA_PATTERN.sub(r"\1", js_text)
    return js_text

def extract_metadata(html_path: Path) -> Dict[str, Any] | None:
    text = html_path.read_text(encoding="utf-8")
    match = METADATA_PATTERN.search(text)
    if not match:
        return None

    js_object = match.group(1)
    json_text = js_object_to_json(js_object)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Metadati non validi in {html_path}: {exc}") from exc

    return data


def scan_theory(theory_dir: Path) -> List[Dict[str, Any]]:
    entries = []

    for html_file in theory_dir.rglob("*.html"):
        # Escludi l'indice generale della teoria; puoi cambiare la regola se vuoi
        if html_file.name == "index.html" and html_file.parent == theory_dir:
            continue

        metadata = extract_metadata(html_file)
        if metadata is None:
            continue

        rel_path = html_file.relative_to(theory_dir).as_posix()

        section_title = metadata.get("section", "Senza sezione")
        section_slug = metadata.get("sectionSlug") or section_title.lower().replace(" ", "-")

        entry = {
            "title": metadata["title"],
            "file": rel_path,
            "description": metadata.get("description", ""),
            "order": metadata.get("order", 9999),
            "level": metadata.get("level", "base"),
            "tags": metadata.get("tags", []),
            "section": section_title,
            "sectionSlug": section_slug,
        }
        entries.append(entry)

    return entries


def build_structure(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    sections_map: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        slug = entry["sectionSlug"]
        if slug not in sections_map:
            sections_map[slug] = {
                "title": entry["section"],
                "slug": slug,
                "description": "",
                "order": 9999,
                "topics": []
            }

        topic = {
            "title": entry["title"],
            "file": entry["file"],
            "description": entry["description"],
            "order": entry["order"],
            "level": entry["level"],
            "tags": entry["tags"],
        }
        sections_map[slug]["topics"].append(topic)

        # Usa il minimo order degli argomenti come ordine della sezione,
        # se non hai metadati separati per le sezioni
        sections_map[slug]["order"] = min(
            sections_map[slug]["order"],
            int(entry["order"])
        )

    sections = list(sections_map.values())

    for section in sections:
        section["topics"].sort(key=lambda x: (int(x["order"]), x["title"].lower()))

    sections.sort(key=lambda x: (int(x["order"]), x["title"].lower()))

    return {"sections": sections}


def main() -> None:
    if len(sys.argv) > 1:
        repo_root = Path(sys.argv[1]).resolve()
    else:
        repo_root = Path.cwd()

    theory_dir = repo_root / "theory"
    output_path = theory_dir / "theory.json"

    if not theory_dir.exists():
        raise SystemExit(f"Cartella non trovata: {theory_dir}")

    entries = scan_theory(theory_dir)
    structure = build_structure(entries)

    output_path.write_text(
        json.dumps(structure, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Creato: {output_path}")
    print(f"Sezioni trovate: {len(structure['sections'])}")
    print(f"Argomenti trovati: {sum(len(s['topics']) for s in structure['sections'])}")


if __name__ == "__main__":
    main()

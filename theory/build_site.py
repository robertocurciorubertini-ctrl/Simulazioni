#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_site.py

Esegue in sequenza:
1) build_catalog.py          -> aggiorna simulations.json
2) generate_theory_json.py   -> aggiorna il JSON della teoria

Uso:
    py build_site.py
    py build_site.py --skip-simulations
    py build_site.py --skip-theory
    py build_site.py --only-simulations
    py build_site.py --only-theory

Note:
- Questo script NON sostituisce gli altri due: li coordina.
- I file build_catalog.py e generate_theory_json.py devono stare
  nella stessa cartella di build_site.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path) -> int:
    """
    Esegue uno script Python usando lo stesso interprete attuale.

    Parametri
    ----------
    script_path : Path
        Percorso assoluto o relativo dello script da eseguire.

    Restituisce
    -----------
    int
        Codice di uscita del processo:
        - 0 se tutto va bene
        - diverso da 0 in caso di errore
    """
    print(f"\n==> Esecuzione di: {script_path.name}")
    print("-" * 60)

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            check=False
        )
        return completed.returncode
    except Exception as exc:
        print(f"Errore durante l'esecuzione di {script_path.name}: {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lancia la build completa del sito: simulazioni + teoria."
    )

    parser.add_argument(
        "--skip-simulations",
        action="store_true",
        help="Salta la build delle simulazioni."
    )
    parser.add_argument(
        "--skip-theory",
        action="store_true",
        help="Salta la build della teoria."
    )
    parser.add_argument(
        "--only-simulations",
        action="store_true",
        help="Esegue solo la build delle simulazioni."
    )
    parser.add_argument(
        "--only-theory",
        action="store_true",
        help="Esegue solo la build della teoria."
    )

    args = parser.parse_args()

    if args.only_simulations and args.only_theory:
        print("Errore: non puoi usare contemporaneamente --only-simulations e --only-theory.")
        return 2

    base_dir = Path(__file__).resolve().parent
    build_catalog = base_dir / "generate_sim.py"
    generate_theory = base_dir / "generate_theory.py"

    if not build_catalog.exists():
        print(f"Errore: file non trovato -> {build_catalog}")
        return 1

    if not generate_theory.exists():
        print(f"Errore: file non trovato -> {generate_theory}")
        return 1

    run_simulations = True
    run_theory = True

    if args.only_simulations:
        run_theory = False
    elif args.only_theory:
        run_simulations = False
    else:
        if args.skip_simulations:
            run_simulations = False
        if args.skip_theory:
            run_theory = False

    if not run_simulations and not run_theory:
        print("Nessuna build da eseguire.")
        return 0

    print("Avvio build del sito...")
    print(f"Cartella di lavoro: {base_dir}")

    if run_simulations:
        code = run_script(build_catalog)
        if code != 0:
            print("\nBuild interrotta: errore nella generazione del catalogo simulazioni.")
            return code

    if run_theory:
        code = run_script(generate_theory)
        if code != 0:
            print("\nBuild interrotta: errore nella generazione del catalogo teoria.")
            return code

    print("\nBuild completata con successo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
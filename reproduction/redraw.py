"""Regenerate the ten manuscript figures in an output folder."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    1: ("fig01", "draw_fig01_v5.py", "Fig01_Overall_Architecture.pdf"),
    2: ("fig02", "draw_fig02_unified.py", "Fig04_Community_Objectives.pdf"),
    3: ("fig03", "draw_fig03_final.py", "Fig02_HCTF_Structural_Block.pdf"),
    4: ("fig04", "draw_safeblend_v9.py", "Fig03_SafeBlend_Protocol.pdf"),
    5: ("statistical", "mechanism", "pdf/hctf_mechanism_ablation.pdf"),
    6: ("fig06", "draw_factorial_validation.py", "factorial_validation_effects.pdf"),
    7: ("fig07", "draw_fig07.py", "pdf/Fig05_Paired_Stress_Effects.pdf"),
    8: ("statistical", "safe_holdout", "pdf/safe_holdout_forest.pdf"),
    9: ("statistical", "source/render_runtime.py", "pdf/end_to_end_scaling.pdf"),
    10: ("fig10", "draw_fig10.py", "pdf/Fig07_Qualitative_Layout_Comparison.pdf"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=int, choices=FIGURES, nargs="+")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MPLBACKEND="Agg")
    for number in args.figure or FIGURES:
        folder, entry, result = FIGURES[number]
        work = out / f"figure_{number:02d}"
        shutil.copytree(ROOT / "figure_sources" / folder, work)
        if folder == "statistical" and "/" not in entry:
            call = [
                sys.executable, "-c",
                "import runpy,sys; routines=runpy.run_path(sys.argv[1]); routines[sys.argv[2]]()",
                "source/render_figures.py", entry,
            ]
        else:
            call = [sys.executable, entry]
        subprocess.run(call, cwd=work, env=env, check=True)
        generated = work / result
        if not generated.is_file():
            raise FileNotFoundError(generated)
        published = next((ROOT / "paper_figures").glob(f"Fig{number:02d}_*.pdf"))
        shutil.copy2(generated, out / published.name)
        print(f"Rendered Figure {number}", flush=True)


if __name__ == "__main__":
    main()

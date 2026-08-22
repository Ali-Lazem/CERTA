# manuscript/

LaTeX sources for the paper and its Supplementary Information.

| File | Description |
|------|-------------|
| `main.tex` | Main manuscript. Uses the Springer Nature journal class (`sn-jnl.cls`, Nature-family submission template). |
| `SI.tex` | Supplementary Information. Self-contained `article` class; compiles on its own. 12 sections, 6 figures, 16 tables. |
| `si_tables.tex` | All Supplementary tables, **auto-generated** by `../src/si_tables.py`. Do not edit by hand; regenerate and copy in. |
| `fig1_architecture.tex` | TikZ source for Figure 1 (the certify-or-escalate architecture). |
| `certa.bib` | Bibliography (includes `zhao2023large` for PMC-Patients). |
| `cover_letter_NM.tex` | Cover letter to the editor (Nature Medicine submission). |

## Building

The SI is standalone and compiles with a normal TeX distribution:

```bash
cd manuscript
pdflatex SI.tex && pdflatex SI.tex   # two passes resolve refs; -> SI.pdf (21 pp)
```

The main text uses the **Springer Nature** LaTeX template, which is not redistributed here. Download `sn-jnl.cls` and the `sn-bibliography` style from the Springer Nature LaTeX package, place them alongside `main.tex`, then:

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`main.tex` and `SI.tex` share cross-references via `xr-hyper`; build `SI.tex` first so `SI.aux` exists when `main.tex` is compiled.

## Regenerating tables

`si_tables.tex` is a build product. To refresh it after any pipeline change:

```bash
cd ..                      # repo root
python src/si_compute.py   # -> si_data.pkl (+ si_compute2.py, si_compute3.py)
python src/si_tables.py    # -> si_tables.tex
cp si_tables.tex manuscript/
```


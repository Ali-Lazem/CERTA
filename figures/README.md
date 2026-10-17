# figures/

Publication-ready PDFs. Main-text figures fig2–fig5 are produced by `../src/build_figures.py`; supplementary figures figS1–figS6 by `../src/build_si_figures.py`. All values are recomputed from `certa_core.py`; none are hardcoded. (Figure 1 is a TikZ architecture schematic maintained with the paper sources, not a data figure, and is not reproduced here.)

## Main text

| File | Content |
|------|---------|
| `fig2.pdf` | Decision certification across the eleven models; the gap between raw agreement and what is statistically certifiable; and the resulting hardware footprint. |
| `fig3.pdf` | How the risk level determines the deployable footprint; certificate calibration; and the small cost of the class-conditional guarantee. |
| `fig4.pdf` | The quantizer determines whether four-bit compression is certifiable (NF4 vs RTN). |
| `fig5.pdf` | Deployed configurations preserve external task accuracy — complementary to, not the same as, reference agreement. |

## Supplementary

| File | Content |
|------|---------|
| `figS1.pdf` | Certified compression frontiers for all eleven models under the marginal certificate. |
| `figS2.pdf` | Prevalence-shift stress test, all eleven models. |
| `figS3.pdf` | Escalated cases are enriched for low-margin, high-disagreement inputs (NF4, α = 5%). |
| `figS4.pdf` | Two mechanisms of reject-class escalation (resolution-limited vs risk-limited). |
| `figS5.pdf` | Statistical robustness (confidence-parameter and calibration-size sensitivity). |
| `figS6.pdf` | Design-choice robustness of the marginal frontier (threshold-grid and repeated-split stability). |

## Regenerating

```bash
# from the repository root (scripts resolve data/ and results/ automatically)
python src/build_figures.py     # -> fig2.pdf .. fig5.pdf   (written to the working dir)
python src/build_si_figures.py  # -> figS1.pdf .. figS6.pdf (written to the working dir)
```
Both scripts recompute every value from `certa_core.py` and the recorded outputs in `data/verify/`; no figure value is hardcoded. `fig5.pdf` additionally reads the external disease-NER benchmark in `data/verify/bench.csv`.

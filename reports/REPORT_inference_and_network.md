# DeepSLP — Genome-wide SL Inference & Network Report

**Date:** 2026-06-25
**Scope:** Genome-wide synthetic-lethality (SL) prediction from the 10-model CV2
ensemble, plus construction of a de-hubbed inferred SL network.
**Pipeline:** `reproduce_cv2/` (`infer_all_pairs.py`, `build_dehubbed_network.py`,
`analysis_top_pairs.py`). **Outputs:** `data/interim/all_pairs_pred/`.

---

## 1. Genome-wide inference
10-model ensemble (mean predicted probability), GIV pairwise features built on the
fly, scoring every unique gene pair except the ~4M already screened.

- Gene universe: **17,840** genes (KO ∩ Expression embeddings).
- Candidate pairs C(N,2) = 159,123,880; screened masked = 3,746,864.
- **Scored (written) = 155,377,016 pairs.** Runtime ≈ 29 min (single GPU).

**Deliverables (`data/interim/all_pairs_pred/`):**
| file | content | size / rows |
|---|---|---|
| `all_pairs_ensemble10_3L.tsv` | full: gene1, gene2, prob_fold1…10, mean (4 dp) | 14 GB / 155.4M |
| `all_pairs_ensemble10_3L_mean_only.tsv` | trimmed: gene1, gene2, mean_pred_proba | 3.0 GB / 155.4M |
| `all_pairs_ensemble10_3L_top0.3pct_mean_only.tsv` | top 0.3% by mean, 3 cols | 9 MB / 466,132 |
| `all_pairs_ensemble10_3L_top0.3pct.tsv` | top 0.3% with all 10 fold cols | 42 MB |
| `screened_mask_keys_3L.npy` | reusable screened-pair mask | 30 MB |

Ensemble-mean score distribution over the top 0.3%: min 0.120 (the 0.3% cutoff),
median 0.130, max **0.401**.

---

## 2. Hub structure of the raw top list
Treating the top-0.3% list as a graph shows it is strongly **hub-dominated** — a
small number of genes pair with almost everything:

| metric | value |
|---|---|
| edges (gene pairs) | 466,132 |
| unique genes | 9,258 |
| max degree | **MEN1 = 6,693** |
| degree Gini | 0.813 |
| connected components | 1 (giant) |

This arises because the pairwise GIV feature is **additive per gene**, so a gene
whose contribution aligns with the model's SL-positive direction lifts the score of
nearly every partner. A raw probability threshold therefore yields a hub star rather
than a usable network — motivating the degree correction below.

---

## 3. De-hubbed SL network
**Method:** degree-corrected score `c(i,j) = mean − 0.5·(m_i + m_j)`, where `m_g` is
gene g's mean ensemble probability over all of its unscreened partners (its baseline
propensity), followed by a **mutual top-25** reciprocal filter (keep an edge only if
each gene is in the other's top-25 by corrected score). This caps every gene's degree
and requires reciprocity.

| metric | raw top-0.3% | de-hubbed | high-confidence (score > 0.05) |
|---|---|---|---|
| edges | 466,132 | 1,894 | **1,789** |
| genes | 9,258 | 866 | 748 |
| max degree | 6,693 | 25 (capped) | 25 |
| degree Gini | 0.813 | 0.555 | — |
| components | 1 giant | — | 25 (largest 681) |
| corrected score range | — | −0.004 … 0.320 | 0.056 … 0.320 |

**Files:**
- `dehubbed_network_3L.tsv` — full de-hubbed (gene1, gene2, score=corrected, raw_mean, m_g1, m_g2)
- `dehubbed_network_3L_hiconf.tsv` — corrected score > 0.05 subset
- `dehubbed_network_3L_hiconf_edgelist.txt` — **Cytoscape edge list**
  (source, target, interaction=`predicted_SL`, corrected_score, raw_mean)
- `dehubbed_network_3L_gene_degree.tsv`, `dehubbed_network_3L.tsv.summary.txt`

To load in Cytoscape: Import → Network from File → `dehubbed_network_3L_hiconf_edgelist.txt`
(source = Source, target = Target, interaction = Interaction Type; use
`corrected_score` for edge weight/width).

---

## 4. Biology — a vesicle/Golgi trafficking module
After de-hubbing, the network is dominated by a **coherent membrane-trafficking
module**. Top genes by degree: `RAB18, TSSC1, PI4KB, RIC8A, UNC50, VPS52, COG3,
COG7, TMED2, TBC1D20` (Rab GTPases, the COG complex, VPS, PI4KB).

Strongest high-confidence edges:
| edge | corrected | raw mean |
|---|---|---|
| RAB18 – UNC50 | 0.320 | 0.401 |
| COG3 – RAB18 | 0.315 | 0.379 |
| RAB18 – TSSC1 | 0.292 | 0.368 |
| COG7 – RAB18 | 0.287 | 0.351 |
| TSSC1 – UNC50 | 0.287 | 0.359 |
| RAB18 – VPS51 | 0.254 | 0.322 |

Genes from the same trafficking complexes (COG complex, RAB18 regulators, VPS) being
predicted as mutually SL is the kind of pathway-coherent structure expected of a real
SL network, supporting the biological plausibility of the top predictions.

---

## 5. Recommendations
1. For downstream network work, prefer the **degree-corrected, mutual-top-k** network
   (`dehubbed_network_3L_hiconf*`) over the raw top-0.3%, which is hub-dominated.
2. Tune the corrected-score threshold (currently 0.05) and `topk` (currently 25) to
   trade off network size vs stringency for your use case.
3. Treat the per-gene hub effect as inherent to the additive pairwise-feature design;
   a future non-additive pair encoder or a positive-aware loss/resampling scheme could
   reduce it and is worth evaluating.
4. Validate selected high-confidence edges against external complex/pathway gold
   standards (e.g. CORUM/STRING/Reactome) before experimental follow-up.

**Scripts:** `reproduce_cv2/infer_all_pairs.py`, `build_dehubbed_network.py`,
`analysis_top_pairs.py`. **Analysis figures:** `data/interim/all_pairs_pred/analysis_3L/`.

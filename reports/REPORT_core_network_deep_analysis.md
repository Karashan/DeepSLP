# DeepSLP — Core SL Network: Deep Analysis (degree, modules, CORUM)

**Date:** 2026-06-26
**Input:** binarized core SL network `SL_core_network_cutoff0.162.tsv`
(`mean_pred_proba ≥ 0.162`; 24,491 edges, 3,672 genes).
**Script:** `reproduce_cv2/core_network_deep_analysis.py`.
**Outputs:** `data/interim/all_pairs_pred/core_network_analysis/`.

---

## (a) Degree vs. mean incident-edge score — are hubs high-confidence or threshold-grazing?
- Spearman(degree, mean incident score) = **+0.575** (p ≈ 0): higher-degree genes do tend
  to have slightly higher incident scores.
- **But in absolute terms the hubs are threshold-grazing:**
  - top-50 hubs: mean incident score **0.1795** (median 0.1792)
  - low-degree (≤3) genes: mean incident score 0.1692
  - core median edge score 0.1726; cutoff 0.162; **max 0.401**

So hub edges sit just above the 0.162 cutoff (≈0.18), far from the strong-edge range
(0.3–0.4). The hubs are mostly an accumulation of **many weak, threshold-grazing edges**
— consistent with the per-gene marginal artifact (a hub gene nudges many partners just
over the cutoff) rather than a few high-confidence interactions.
*Plot:* `degree_vs_incident_score.png`; *table:* `gene_incident_stats.tsv`.

---

## (b) Community structure & module functional summary
Louvain (weighted) on the core network: **4 modules ≥ 3 genes**, modularity **Q = 0.293**
(modest — the network does not split into many clean small modules at this resolution).

| module | size | top genes | theme |
|---|---|---|---|
| C0 | 1,569 | UBR4, PAXIP1, PDS5A, CTNNA1, RIC8A, INTS6, MEN1, TOE1 | mixed: ubiquitin / chromatin / cohesin |
| C1 | 1,473 | UNC50, RAB18, DNAJC9, TSSC1, KLF5, COG3, VPS52 | **vesicle/Golgi trafficking** |
| C2 | 367 | MCL1, UBE2A, PFDN1, HECTD1, ING3, DYNLL1, DDX5, HIRA | chromatin / proteostasis |
| C3 | 263 | VPS33A, NAPG, CDK4, SYS1, TMED2, COG7, KDM2A, CHMP5 | **Golgi/HOPS/COG trafficking** |

> Caveat: the automatic "dominant CORUM complex" label uses max raw overlap, which is biased
> toward very large complexes (e.g. Complex I, spliceosome) and is not informative for these
> big modules; the top-gene themes above are the more meaningful read-out. Trafficking
> (RAB18/COG/VPS/UNC50/TSSC1) is the clearest coherent program.
*Plot:* `community_sizes.png`; *tables:* `community_summary.tsv`, `community_membership.tsv`.

---

## (c) CORUM co-complex enrichment (gold-standard validation)
CORUM v5.3 human: 5,628 complexes, 5,150 genes (4,830 predictable). Co-complex
gene pairs among predictable genes: 43,358 (background rate 3.7e‑3).

| quantity | value |
|---|---|
| core SL edges with both genes CORUM-annotated (E) | 6,949 |
| of those, co-complex pairs (E_cc) | **43** |
| observed co-complex rate | 0.62% |
| **fold enrichment** | **1.7×** |
| hypergeometric p-value | **1.2e‑3** |
| recall of known co-complex pairs | 0.10% (43 / 43,358) |

The core SL edges are **significantly enriched** for known protein-complex pairs
(1.7×, p = 1.2e‑3) — modest in magnitude but real. Top validated edges are bona fide
complexes:
`UBE2A–UBR4` (RAD6A–KCMF1–UBR4), `PAGR1–PAXIP1` (PTIP-HMT), `VPS33A–VPS41` (HOPS),
`COG6–COG7`, `COG7–COG8`, `COG6–COG8` (COG complex).
*Plot:* `corum_enrichment.png`; *table:* `corum_validated_core_edges.tsv` (43 edges).

---

## Summary & interpretation
1. **Hubs are threshold-grazing** — the raw core network's high-degree genes accumulate
   many weak (~0.18) edges, the per-gene marginal effect. For clean modular biology use
   the degree-corrected `dehubbed_network_3L*` rather than the raw core hubs.
2. **Coherent trafficking program** recurs (modules C1/C3; RAB18/COG/VPS/HOPS), matching
   the strongest individual edges.
3. **Statistically significant CORUM enrichment** (1.7×, p = 1.2e‑3) with concrete
   complex-level hits (COG, HOPS, PTIP, UBR4) — positive but limited-precision biological
   validation. Recall of all known complexes is low (~0.1%), i.e. the network captures a
   small, enriched slice rather than the full complexome.

**Recommendation:** treat the core network as an enriched-but-noisy hypothesis set; rank
candidates by score and CORUM/complex membership, and prefer the de-hubbed network for
module-level interpretation.

---
### Output inventory (`data/interim/all_pairs_pred/core_network_analysis/`)
- `gene_incident_stats.tsv`, `degree_vs_incident_score.png` — (a)
- `community_summary.tsv`, `community_membership.tsv`, `community_sizes.png` — (b)
- `corum_validated_core_edges.tsv`, `corum_enrichment.png` — (c)
- `deep_analysis_stats.txt` — combined text summary
- (from prior step) `network_stats.txt`, `gene_degree.tsv`, `degree_distribution.png`,
  `top_hubs.png`, `edge_score_distribution.png`, `top_subnetwork.png`

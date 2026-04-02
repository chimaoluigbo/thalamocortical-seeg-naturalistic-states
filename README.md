# thalamocortical-seeg-naturalistic-states

**Analysis code for:**
> Oluigbo CO, Wittanayakorn N, Xie H, Cohen NT, Karakas C, Boonkrongsak R, Anwar S, Berl MM, Gaillard WD, Sepeta L.
> "A low-dimensional thalamocortical state axis in humans."
> *Nature Communications* (submitted 2026)

Children's National Research Institute and Children's National Hospital, Washington, DC
Correspondence: coluigbo@childrensnational.org

---

## Overview

This repository contains all analysis scripts used to produce the results, figures, and statistics reported in the manuscript. The study quantified thalamocortical imaginary coherence (iCoh) from stereoencephalography (SEEG) recordings in 27 children, teenagers, and young adults during naturalistic inpatient monitoring across nine behavioral states. Principal component analysis revealed a dominant low-dimensional axis (PC1 = 54.0%) along which thalamocortical connectivity varies systematically with behavior.

---

## Repository Structure

```
thalamocortical-seeg-naturalistic-states/
│
├── README.md
├── requirements.txt
│
├── ── PREPROCESSING ──────────────────────────────────────────────────
│
├── build_thomas_ground_truth.py        Build atlas-assigned thalamic contact list
│                                        using the Saranathan et al. (2021) THOMAS atlas.
│                                        Expands contact count from 112 → 161 by applying
│                                        atlas labels to all contacts whose MNI coordinates
│                                        fall within thalamic nuclei.
│
├── add_grey_matter_flag.py             Merge grey matter classification into the contact
│                                        list. Assigns tissue_class (cortical_grey /
│                                        white_matter / subcortical / unknown) and
│                                        is_grey_matter flag using DKT atlas (threshold
│                                        >= 50% probability).
│
├── ── MAIN PIPELINE (run steps 1–6 in order) ─────────────────────────
│
├── step1_edf_inventory.py              Walk the edf/ folder, read file headers (no
│                                        signal data loaded), output subject × state
│                                        event count matrix and duration table.
│                                        Saves: outputs/results/edf_inventory.csv
│
├── step2_inspect_edf.py                Load one EDF and verify channel names, amplitude
│                                        ranges, and thalamic contact matching before
│                                        preprocessing begins. Inspection/QC only —
│                                        does not save processed data.
│
├── step3_preprocess_one_clip.py        Preprocess a single EDF clip and save a
│                                        verification NPZ. Used to confirm preprocessing
│                                        logic (POL stripping, filtering, bipolar
│                                        referencing) before running on all subjects.
│                                        Saves: outputs/results/step3_verification.npz
│
├── step4_preprocess_all.py             Full preprocessing of every EDF across all 27
│                                        subjects: strip POL prefix, drop non-SEEG
│                                        channels (EKG/DC/status), resample to 256 Hz,
│                                        bandpass filter 0.5–100 Hz + notch 60 Hz,
│                                        bipolar re-reference within electrode shafts.
│                                        Saves: processed_npz/*.npz
│
├── step5_connectivity.py               Read processed NPZ files, compute iCoh and
│                                        spectral Granger causality per epoch across six
│                                        frequency bands, average across clips per
│                                        subject × state × nucleus.
│                                        Saves: outputs/results/step5_connectivity.csv
│
├── step6_statistics.py                 Pairwise Mann-Whitney U tests across all state
│                                        pairs and metrics, FDR correction (Benjamini-
│                                        Hochberg), Kruskal-Wallis per nucleus, PCA
│                                        manifold computation.
│                                        *** AUTHORITATIVE FILE: 73/252 significant ***
│                                        Saves: outputs/results/step6_statistics.csv
│
├── master_analysis_pipeline.py         Documents and re-runs every analysis step in
│                                        order. Does NOT overwrite step6_statistics.csv.
│                                        Use as the definitive reproducibility record.
│
├── robustness_analyses.py              Four core robustness checks:
│                                        (1) Leave-one-subject-out PCA (27 folds)
│                                        (2) Label-shuffle null (1,000 permutations)
│                                        (3) Phase-randomized null (1,000 permutations)
│                                        (4) Power-matched Granger for crying
│
├── ── FIGURE GENERATION ──────────────────────────────────────────────
│
├── generate_figures.py                 All main manuscript figures: Fig 2 (PCA
│                                        manifold), Fig 3 (iCoh heatmap), Fig 4
│                                        (Granger directionality), Fig 5 (nucleus
│                                        dissociation), Fig 6 (spectral radar),
│                                        FigS1 (coverage heatmap), FigS2 (significant
│                                        contrasts dot plot).
│
├── plot_pca_anatomy_v2.py              PCA scores (PC1 and PC2) mapped onto thalamic
│                                        nucleus MNI centroids, annotated by functional
│                                        group (motor relay / integrative /
│                                        limbic-association).
│
├── plot_figS4_functional_gradient.py   Figure S4: functional gradient (thalamic circuit
│                                        class → PC1, Spearman r = +0.869, p = 0.005)
│                                        vs anatomical gradient (A-P position → PC1,
│                                        r = −0.476, p = 0.233, n.s.)
│
├── plot_figS5_scree.py                 Figure S5: scree plot with phase-randomized null
│                                        CI ribbon. PC3–PC6 each fall below the null
│                                        median, confirming two-dimensional structure.
│
├── interpret_pca_dimensions.py         Figure S6: PC1 loadings (broadband coupling
│                                        magnitude), PC2 loadings (slow vs fast spectral
│                                        composition), four-quadrant state space with
│                                        behavioral centroids.
│
├── ── ROBUSTNESS ANALYSES ────────────────────────────────────────────
│
├── null_pc2.py                         Extend phase-randomized null to PC2. Reports
│                                        PC1 null median 20.7% and PC2 null median
│                                        18.8% (both p < 0.001).
│
├── spatial_permutation_null_v2.py      State-label permutation null: shuffle state
│                                        labels within each subject (correct approach —
│                                        see note below). Confirms observed PC1 = 54.0%
│                                        significantly exceeds null (p < 0.001).
│
├── sensitivity_complete_states.py      Restrict PCA to 7 states with complete coverage
│                                        across all 27 subjects (excludes crying and
│                                        reading, each 9/27 subjects). Result: PC1 =
│                                        54.4%, Spearman r = 1.000.
│
├── sensitivity_nucleus_coverage.py     Test whether PC1 is driven by VLP dominance.
│                                        Three restricted nucleus subsets: well-sampled
│                                        only (r = 0.967), excluding VLP (r = 0.983),
│                                        motor relay only (r = 0.933), limbic only
│                                        (r = 0.867).
│
├── split_half_cortical_reliability.py  Random split of 2,482 cortical contacts,
│                                        1,000 iterations. Mean r = 0.998 (95% CI
│                                        0.997–0.999, min r = 0.996). CV = 28.5%
│                                        confirms global state signal despite local
│                                        contact variability.
│
├── regional_icoh_breakdown.py          Lobe-specific mean broadband iCoh: frontal
│                                        0.0699, temporal 0.0718, parietal 0.0732,
│                                        occipital 0.0701. KW p < 0.001 but η² =
│                                        0.0009 (statistically significant, biologically
│                                        negligible).
│
└── lme_state_lobe.py                   State × lobe permutation test (1,000
                                         permutations). All 9 states: p = 0.152 (n.s.).
                                         Excluding crying: p = 0.925. Confirms
                                         uniformity for 7/9 states; crying exception
                                         reflects pulvinar preferential projections.
```

> **Note on `spatial_permutation_null_v2.py`:** An earlier version shuffled nucleus labels across thalamic contacts before re-aggregating to the all-thalamus mean. This is a null operation — reaggregating to the all-thalamus mean after nucleus-label shuffling yields identical values regardless of permutation. The corrected v2 shuffles *state labels* within each subject, which is the appropriate test for whether the observed state structure exceeds chance.

> **Note on `step6_statistics.csv`:** This file is the authoritative record of 73 significant pairwise contrasts. `master_analysis_pipeline.py` does **not** overwrite it. If the file ever reverts to showing 25 contrasts, restore it by running `python step6_statistics.py` directly.

---

## Key Results

| Metric | Value |
|--------|-------|
| PC1 explained variance | 54.0% |
| PC2 explained variance | 32.9% |
| Combined (PC1 + PC2) | 86.9% |
| PC1 null median | 20.7% (p < 0.001) |
| PC2 null median | 18.8% (p < 0.001) |
| LOOCV PC1 range | 52.7–55.6% |
| LOOCV Spearman r (min) | 0.950, all p < 1e-4 |
| Pairwise significant contrasts | 73/252 (FDR q < 0.05) |
| 7-state sensitivity PC1 | 54.4%, r = 1.000 |
| VLP-excluded Spearman r | 0.983 |
| Split-half cortical r | 0.998 (min 0.996) |
| State × lobe permutation p | 0.152 (n.s.) |
| Functional gradient r | +0.869, p = 0.005 |
| Anatomical gradient r | −0.476, p = 0.233 (n.s.) |
| IRB number | 00003724 |

---

## Installation

**Python >= 3.9 is required.**

```bash
# 1. Clone the repository
git clone https://github.com/chimaoluigbo/thalamocortical-seeg-naturalistic-states.git
cd thalamocortical-seeg-naturalistic-states

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Reproducing the Analysis

All scripts from step 5 onward read from processed data files available on Zenodo. Steps 1–4 require the raw EDF recordings, which cannot be shared due to patient privacy (see Data Availability). Set `DATA_DIR` at the top of each script to point to your local data directory.

### Preprocessing (requires raw EDF files)

```bash
# Build atlas-assigned thalamic contact list
python build_thomas_ground_truth.py

# Add grey matter classification
python add_grey_matter_flag.py

# Inventory raw EDF files
python step1_edf_inventory.py

# Inspect one file before processing (edit SUBJECT/STATE at top of script)
python step2_inspect_edf.py

# Test preprocessing on one clip (edit SUBJECT/STATE at top of script)
python step3_preprocess_one_clip.py

# Full preprocessing of all subjects (~1–2 hours)
python step4_preprocess_all.py --skip_existing
```

### Connectivity computation (requires processed NPZ files)

```bash
# Compute iCoh and Granger causality (~2–4 hours, parallelised)
python step5_connectivity.py --skip_existing

# Compute authoritative pairwise statistics (73/252 significant)
python step6_statistics.py
```

### Robustness analyses

```bash
python robustness_analyses.py          # LOOCV, label-shuffle, phase null, power Granger
python null_pc2.py                     # PC1 + PC2 null model
python spatial_permutation_null_v2.py  # State-label permutation null
python sensitivity_complete_states.py  # 7-state sensitivity
python sensitivity_nucleus_coverage.py # Nucleus coverage sensitivity
python split_half_cortical_reliability.py
python regional_icoh_breakdown.py
python lme_state_lobe.py
```

### Figure generation

```bash
python generate_figures.py                  # Fig 2–6, FigS1–S2
python plot_pca_anatomy_v2.py               # PCA on thalamic MNI centroids
python plot_figS4_functional_gradient.py    # FigS4
python plot_figS5_scree.py                  # FigS5
python interpret_pca_dimensions.py          # FigS6
```

---

## Data Availability

Raw SEEG recordings cannot be shared due to patient privacy requirements (IRB #00003724). The following processed data files required to reproduce all results from step 5 onward are deposited at Zenodo:

- `step5_connectivity.csv` — thalamocortical iCoh and Granger causality per subject × state × nucleus
- `thalamic_nuclei_ground_truth.csv` — atlas-assigned thalamic contact list (Saranathan et al. 2021)
- `grey_matter_contacts_all27.csv` — cortical grey matter contact classifications for all 27 subjects, generated using the **Brainstorm MATLAB toolbox** (Tadel et al. 2011; https://neuroimage.usc.edu/brainstorm). Tissue class and DKT atlas region labels were assigned by querying each contact's MNI coordinates against the DKT probabilistic atlas within Brainstorm.
- `step6_statistics.csv` — authoritative pairwise statistics (73/252 significant)

**Zenodo:** [https://doi.org/10.5281/zenodo.19392389](https://doi.org/10.5281/zenodo.19392389)

Requests for de-identified data should be directed to the corresponding author.

---

## Atlas

Thalamic contact assignments used the **Saranathan et al. (2021)** high-resolution structural MRI atlas of human thalamic nuclei:

> Saranathan M, Iglehart C, Monti M, Tourdias T & Rutt B. In vivo high-resolution structural MRI-based atlas of human thalamic nuclei. *Sci. Data* 8, 275 (2021).

Atlas download: Saranathan M, Iglehart C, Monti M, Tourdias T & Rutt B. Data for: In vivo structural MRI-based atlas of human thalamic nuclei (v4.3). Zenodo (2021). [https://doi.org/10.5281/zenodo.5499504](https://doi.org/10.5281/zenodo.5499504)

---

## IRB and Ethics

This study was approved by the Children's National Hospital Institutional Review Board (IRB #00003724). Written informed consent was obtained from adult participants; parental permission and age-appropriate assent (for participants ≥ 7 years) were obtained for minors.

---

## Citation

If you use this code, please cite:

```
Oluigbo CO, Wittanayakorn N, Xie H, Cohen NT, Karakas C, Boonkrongsak R,
Anwar S, Berl MM, Gaillard WD, Sepeta L. A low-dimensional thalamocortical
state axis in humans. Nature Communications (2026).
DOI: [to be added upon acceptance]
```

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Chima O. Oluigbo, MD, PhD
Children's National Research Institute
Washington, DC
coluigbo@childrensnational.org

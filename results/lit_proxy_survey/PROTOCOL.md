# Proxy-use survey protocol (2026-10-21)

Goal: estimate how often published data-driven SOH estimation studies use proxy inputs.

## Sampling (reproducible)
- Source: OpenAlex API (https://api.openalex.org/works), query on title+abstract:
  "state of health" AND (lithium-ion OR battery) AND (estimation) AND (machine learning OR deep learning OR neural network OR data-driven),
  publication_year 2019–2025, type article, sorted by cited_by_count desc.
- Inclusion: estimates SOH/capacity of the CURRENT cycle from measured signals; uses at least one public dataset
  (NASA PCoE, CALCE, MIT/Severson/Toyota, Oxford, XJTU, TJU, HUST, Sandia, others named).
- Exclusion: RUL / cycle-life / early-life prediction only; reviews; pure model-based (ECM/KF) without ML; no full text accessible.
- Take candidates in citation order until the stratum quota is met. Record every skipped candidate with reason.

## Coding per paper (one CSV row)
- id, doi, title, year, venue, cited_by, datasets, access (full_text / abstract_only)
- inputs: short description of model inputs
- P1_charge_capacity: same-cycle charge capacity or any capacity-equivalent (CC/CV charge capacity, Ah throughput of a full charge) used as input (yes/no/unclear)
- P2_full_curve: features computed from the full (or nearly full) same-cycle discharge curve, incl. IC/DV features over the full discharge, discharge time to cutoff under constant current (= Q/I), full Q(V) (yes/no/unclear); note voltage window / coverage if partial
- P2c_full_charge_curve: features from a same-cycle full CC charge starting from a fully discharged state (e.g. CC charge time) (yes/no/unclear)
- P3_future_info: per-cell whole-life normalization, end-of-life cycle, life fraction (yes/no/unclear)
- split: cell_held_out / within_cell / unclear
- simple_baseline: linear/tree/mean baseline reported under same protocol (yes/no)
- evidence: <=15-word quote + section for each "yes"
- notes

Rules: never infer from memory; "unclear" when the text does not say. Every DOI checked against Crossref/OpenAlex.

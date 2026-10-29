# Verification pass (2026-10-21)
Independent re-check of every paper with at least one "yes" in P1/P2/P2c/P3.
For each such paper and each "yes" item:
- locate the quoted evidence in the extracted full text (raw_*/); read the methods/feature section around it;
- verdict: confirmed / downgraded_to_no / downgraded_to_unclear;
- strength (for confirmed P2/P2c): strong = capacity-equivalent (full Q(V), time-to-cutoff or full CC time under constant current, Ah/Wh over a full charge or discharge, IC/DV area over the full curve);
  weak = summary statistic or local feature computed over a full curve (mean voltage, entropy, IC peak height/position, SVD of resampled curve);
- also re-check: is the task really CURRENT-cycle SOH estimation; is the evidence about the final model inputs (not just a candidate pool) — note if unknown.
Output: verify_<stratum>.csv with columns id, doi, item, original, verdict, strength, input_role (final/candidate_pool/unknown), quote(<=15 words), location, note.

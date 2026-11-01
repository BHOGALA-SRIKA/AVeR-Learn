# member3_selectors (test)

Evidence selectors on one document per question. Token counter: `approx-v1`. Significance: paired randomization test vs `emte` on cov@512, Holm-adjusted.

## Evidence delivered

| Selector | cov@256 | cov@512 | cov@1024 | allEss@512 | gold share@512 | tokens@512 | regions@512 | diff vs emte | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_sections | 0.869 | 0.983 | 0.983 | 0.967 | 0.102 | 413 | 7.2 | +0.042 (2W/0L) | 1.000 |
| granularity=section | 0.802 | 0.956 | 0.983 | 0.933 | 0.070 | 505 | 8.4 | +0.015 (1W/0L) | 1.000 |
| emte | 0.895 | 0.941 | 0.956 | 0.900 | 0.089 | 430 | 7.2 | baseline |  |
| emte(agg=sum) | 0.895 | 0.941 | 0.956 | 0.900 | 0.089 | 430 | 7.2 | +0.000 (0W/0L) | 1.000 |
| emte(agg=max) | 0.895 | 0.941 | 0.956 | 0.900 | 0.089 | 430 | 7.2 | +0.000 (0W/0L) | 1.000 |
| emte(tau_up=0.002) | 0.829 | 0.861 | 0.861 | 0.800 | 0.103 | 362 | 8.4 | -0.080 (1W/4L) | 0.669 |
| hichunk | 0.714 | 0.793 | 0.838 | 0.667 | 0.090 | 409 | 13.9 | -0.148 (0W/8L) | 0.058 |
| hichunk_with_stop | 0.714 | 0.793 | 0.810 | 0.667 | 0.138 | 307 | 9.8 | -0.148 (0W/8L) | 0.058 |
| emte(min_raw_score=1.7) | 0.762 | 0.762 | 0.762 | 0.733 | 0.210 | 166 | 2.7 | -0.179 (0W/6L) | 0.190 |
| emte(tau_up=0.01) | 0.644 | 0.721 | 0.727 | 0.533 | 0.088 | 396 | 15.0 | -0.220 (1W/12L) | 0.006 |
| emte(agg=mean) | 0.652 | 0.715 | 0.741 | 0.567 | 0.079 | 471 | 17.2 | -0.225 (0W/16L) | 0.002 |
| section_then_paragraph | 0.609 | 0.703 | 0.758 | 0.567 | 0.104 | 428 | 20.7 | -0.238 (0W/16L) | 0.002 |
| emte(min_raw_score=2.0) | 0.695 | 0.695 | 0.695 | 0.667 | 0.232 | 137 | 2.2 | -0.246 (0W/8L) | 0.058 |
| granularity=paragraph | 0.593 | 0.692 | 0.745 | 0.533 | 0.076 | 509 | 22.2 | -0.249 (0W/19L) | 0.002 |
| plain_automerge | 0.619 | 0.619 | 0.619 | 0.467 | 0.234 | 131 | 4.3 | -0.322 (0W/16L) | 0.002 |
| emte(max_regions=1) | 0.618 | 0.618 | 0.618 | 0.533 | 0.314 | 70 | 1.0 | -0.323 (0W/12L) | 0.007 |
| emte(min_raw_score=2.5) | 0.613 | 0.613 | 0.613 | 0.567 | 0.238 | 100 | 1.6 | -0.328 (0W/12L) | 0.007 |
| adaptive_k | 0.558 | 0.574 | 0.574 | 0.367 | 0.133 | 242 | 11.0 | -0.367 (0W/21L) | 0.002 |
| inex | 0.487 | 0.546 | 0.626 | 0.300 | 0.042 | 508 | 18.8 | -0.395 (2W/19L) | 0.002 |
| score_ratio | 0.521 | 0.544 | 0.544 | 0.333 | 0.171 | 203 | 9.0 | -0.397 (0W/21L) | 0.002 |
| fixed_k | 0.462 | 0.462 | 0.462 | 0.200 | 0.196 | 114 | 5.0 | -0.479 (0W/24L) | 0.002 |
| score_gap | 0.448 | 0.461 | 0.472 | 0.300 | 0.195 | 149 | 6.8 | -0.480 (0W/22L) | 0.002 |
| granularity=subsection | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 61 | 1.0 | -0.941 (0W/30L) | 0.002 |

## Student vs book wording (cov@512)

| Selector | student | book |
|---|---:|---:|
| flat_sections | 1.000 (n=9) | 0.976 (n=21) |
| fixed_k | 0.314 (n=9) | 0.526 (n=21) |
| score_ratio | 0.369 (n=9) | 0.619 (n=21) |
| score_gap | 0.236 (n=9) | 0.557 (n=21) |
| adaptive_k | 0.382 (n=9) | 0.656 (n=21) |
| section_then_paragraph | 0.495 (n=9) | 0.792 (n=21) |
| granularity=paragraph | 0.494 (n=9) | 0.776 (n=21) |
| granularity=subsection | 0.000 (n=9) | 0.000 (n=21) |
| granularity=section | 1.000 (n=9) | 0.937 (n=21) |
| plain_automerge | 0.430 (n=9) | 0.700 (n=21) |
| hichunk | 0.570 (n=9) | 0.889 (n=21) |
| hichunk_with_stop | 0.570 (n=9) | 0.889 (n=21) |
| inex | 0.528 (n=9) | 0.554 (n=21) |
| emte | 0.951 (n=9) | 0.937 (n=21) |
| emte(agg=sum) | 0.951 (n=9) | 0.937 (n=21) |
| emte(agg=max) | 0.951 (n=9) | 0.937 (n=21) |
| emte(agg=mean) | 0.601 (n=9) | 0.764 (n=21) |
| emte(max_regions=1) | 0.265 (n=9) | 0.770 (n=21) |
| emte(tau_up=0.002) | 0.752 (n=9) | 0.908 (n=21) |
| emte(tau_up=0.01) | 0.554 (n=9) | 0.793 (n=21) |
| emte(min_raw_score=1.7) | 0.466 (n=9) | 0.889 (n=21) |
| emte(min_raw_score=2.0) | 0.355 (n=9) | 0.841 (n=21) |
| emte(min_raw_score=2.5) | 0.265 (n=9) | 0.762 (n=21) |

## Speed and memory (selection only; index already built)

| Selector | median ms | p95 ms | peak alloc MB |
|---|---:|---:|---:|
| flat_sections | 7.93 | 21.02 | 0.3 |
| fixed_k | 7.53 | 9.56 | 0.5 |
| score_ratio | 7.61 | 10.30 | 0.5 |
| score_gap | 8.10 | 12.68 | 0.5 |
| adaptive_k | 8.23 | 14.87 | 0.5 |
| section_then_paragraph | 13.99 | 21.61 | 0.5 |
| granularity=paragraph | 44.25 | 85.85 | 0.3 |
| granularity=subsection | 4.73 | 8.29 | 0.3 |
| granularity=section | 12.53 | 15.80 | 0.3 |
| plain_automerge | 15.00 | 22.44 | 0.3 |
| hichunk | 17.42 | 65.99 | 0.5 |
| hichunk_with_stop | 20.19 | 38.36 | 0.9 |
| inex | 95.09 | 147.23 | 0.5 |
| emte | 48.68 | 93.79 | 0.5 |
| emte(agg=sum) | 43.41 | 86.80 | 0.5 |
| emte(agg=max) | 43.19 | 90.98 | 0.5 |
| emte(agg=mean) | 77.32 | 216.82 | 0.5 |
| emte(max_regions=1) | 17.42 | 21.20 | 0.4 |
| emte(tau_up=0.002) | 50.36 | 111.47 | 0.5 |
| emte(tau_up=0.01) | 78.76 | 194.70 | 0.5 |
| emte(min_raw_score=1.7) | 17.59 | 57.57 | 0.4 |
| emte(min_raw_score=2.0) | 18.65 | 53.97 | 0.4 |
| emte(min_raw_score=2.5) | 12.50 | 38.69 | 0.4 |

Index build per document (Member 1 files + Member 2 BM25F): unit1_topic1 92 ms, unit1_topic2 51 ms, unit1_topic3 138 ms

## Out-of-scope questions (should abstain)

Each out-of-scope question is run on every document. A good selector abstains on these and not on in-scope questions.

| Selector | out-of-scope abstained (right) | in-scope abstained (wrong) | tokens@512 given for out-of-scope |
|---|---:|---:|---:|
| flat_sections | 1/12 | 0/30 | 142 |
| fixed_k | 0/12 | 0/30 | 144 |
| score_ratio | 1/12 | 0/30 | 167 |
| score_gap | 0/12 | 0/30 | 211 |
| adaptive_k | 0/12 | 0/30 | 278 |
| section_then_paragraph | 0/12 | 0/30 | 478 |
| granularity=paragraph | 0/12 | 0/30 | 509 |
| granularity=subsection | 4/12 | 10/30 | 58 |
| granularity=section | 0/12 | 0/30 | 501 |
| plain_automerge | 0/12 | 0/30 | 168 |
| hichunk | 1/12 | 0/30 | 108 |
| hichunk_with_stop | 1/12 | 0/30 | 108 |
| inex | 0/12 | 0/30 | 507 |
| emte | 1/12 | 0/30 | 156 |
| emte(agg=sum) | 1/12 | 0/30 | 156 |
| emte(agg=max) | 1/12 | 0/30 | 156 |
| emte(agg=mean) | 1/12 | 0/30 | 168 |
| emte(max_regions=1) | 1/12 | 0/30 | 67 |
| emte(tau_up=0.002) | 1/12 | 0/30 | 86 |
| emte(tau_up=0.01) | 1/12 | 0/30 | 128 |
| emte(min_raw_score=1.7) | 7/12 | 0/30 | 25 |
| emte(min_raw_score=2.0) | 10/12 | 2/30 | 10 |
| emte(min_raw_score=2.5) | 12/12 | 7/30 | 0 |

### Can a score threshold separate them? (top raw BM25F paragraph score)

| min_raw_score | out-of-scope abstained (right) | in-scope abstained (wrong) |
|---:|---:|---:|
| 0 | 0/12 | 0/30 |
| 1.0 | 1/12 | 0/30 |
| 1.5 | 1/12 | 0/30 |
| 1.7 | 7/12 | 0/30 |
| 2.0 | 10/12 | 2/30 |
| 2.5 | 12/12 | 7/30 |
| 3.0 | 12/12 | 9/30 |
| 4.0 | 12/12 | 19/30 |

In-scope top scores: 1.74, 1.84, 2.27, 2.34, 2.42, 2.43, 2.49, 2.51, 2.71, 3.01, 3.04, 3.15, 3.46, 3.49, 3.50, 3.74, 3.84, 3.90, 3.91, 4.25, 4.42, 4.64, 4.86, 4.89, 5.17, 5.52, 6.46, 6.68, 6.90, 8.21

Out-of-scope top scores: 0.00, 1.55, 1.55, 1.55, 1.58, 1.64, 1.65, 1.84, 1.84, 1.97, 2.01, 2.01

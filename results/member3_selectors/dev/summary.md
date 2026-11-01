# member3_selectors (dev)

Evidence selectors on one document per question. Token counter: `approx-v1`. Significance: paired randomization test vs `emte` on cov@512, Holm-adjusted.

## Evidence delivered

| Selector | cov@256 | cov@512 | cov@1024 | allEss@512 | gold share@512 | tokens@512 | regions@512 | diff vs emte | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_sections | 0.865 | 0.948 | 0.962 | 0.933 | 0.104 | 456 | 8.7 | +0.000 (0W/0L) | 1.000 |
| granularity=section | 0.815 | 0.948 | 0.962 | 0.933 | 0.079 | 506 | 9.5 | +0.000 (0W/0L) | 1.000 |
| emte | 0.798 | 0.948 | 0.962 | 0.933 | 0.101 | 439 | 7.8 | baseline |  |
| emte(agg=sum) | 0.798 | 0.948 | 0.962 | 0.933 | 0.101 | 439 | 7.8 | +0.000 (0W/0L) | 1.000 |
| emte(agg=max) | 0.798 | 0.948 | 0.962 | 0.933 | 0.101 | 439 | 7.8 | +0.000 (0W/0L) | 1.000 |
| emte(tau_up=0.002) | 0.765 | 0.859 | 0.873 | 0.800 | 0.118 | 398 | 11.4 | -0.089 (0W/2L) | 1.000 |
| hichunk | 0.631 | 0.841 | 0.933 | 0.733 | 0.094 | 484 | 16.9 | -0.107 (0W/4L) | 0.755 |
| emte(agg=mean) | 0.568 | 0.776 | 0.834 | 0.533 | 0.093 | 485 | 20.1 | -0.172 (0W/8L) | 0.116 |
| emte(tau_up=0.01) | 0.658 | 0.765 | 0.778 | 0.600 | 0.110 | 434 | 17.9 | -0.183 (0W/7L) | 0.202 |
| hichunk_with_stop | 0.631 | 0.761 | 0.761 | 0.600 | 0.135 | 282 | 9.3 | -0.188 (0W/6L) | 0.294 |
| inex | 0.605 | 0.727 | 0.829 | 0.533 | 0.059 | 508 | 20.0 | -0.221 (0W/7L) | 0.202 |
| emte(min_raw_score=1.7) | 0.691 | 0.691 | 0.691 | 0.667 | 0.206 | 172 | 3.2 | -0.258 (0W/5L) | 0.481 |
| granularity=paragraph | 0.547 | 0.668 | 0.778 | 0.400 | 0.077 | 509 | 24.7 | -0.280 (0W/10L) | 0.031 |
| emte(min_raw_score=2.0) | 0.665 | 0.665 | 0.665 | 0.667 | 0.205 | 131 | 2.4 | -0.283 (0W/5L) | 0.481 |
| section_then_paragraph | 0.577 | 0.647 | 0.684 | 0.467 | 0.115 | 398 | 20.3 | -0.302 (0W/10L) | 0.023 |
| emte(min_raw_score=2.5) | 0.598 | 0.598 | 0.598 | 0.600 | 0.223 | 86 | 1.5 | -0.350 (0W/6L) | 0.294 |
| adaptive_k | 0.533 | 0.579 | 0.595 | 0.333 | 0.128 | 271 | 13.0 | -0.369 (0W/10L) | 0.031 |
| plain_automerge | 0.552 | 0.569 | 0.569 | 0.333 | 0.236 | 132 | 4.3 | -0.379 (0W/9L) | 0.048 |
| score_ratio | 0.487 | 0.524 | 0.524 | 0.267 | 0.128 | 245 | 12.1 | -0.424 (0W/12L) | 0.004 |
| score_gap | 0.425 | 0.461 | 0.490 | 0.267 | 0.145 | 192 | 9.3 | -0.488 (0W/11L) | 0.009 |
| emte(max_regions=1) | 0.433 | 0.433 | 0.433 | 0.400 | 0.264 | 66 | 1.0 | -0.515 (0W/9L) | 0.062 |
| fixed_k | 0.426 | 0.426 | 0.426 | 0.133 | 0.181 | 110 | 5.0 | -0.522 (0W/13L) | 0.002 |
| granularity=subsection | 0.077 | 0.077 | 0.077 | 0.067 | 0.041 | 43 | 0.8 | -0.872 (0W/14L) | 0.002 |

## Student vs book wording (cov@512)

| Selector | student | book |
|---|---:|---:|
| flat_sections | 0.850 (n=5) | 0.997 (n=10) |
| fixed_k | 0.105 (n=5) | 0.587 (n=10) |
| score_ratio | 0.138 (n=5) | 0.717 (n=10) |
| score_gap | 0.188 (n=5) | 0.597 (n=10) |
| adaptive_k | 0.217 (n=5) | 0.761 (n=10) |
| section_then_paragraph | 0.267 (n=5) | 0.837 (n=10) |
| granularity=paragraph | 0.333 (n=5) | 0.836 (n=10) |
| granularity=subsection | 0.030 (n=5) | 0.100 (n=10) |
| granularity=section | 0.850 (n=5) | 0.997 (n=10) |
| plain_automerge | 0.155 (n=5) | 0.776 (n=10) |
| hichunk | 0.658 (n=5) | 0.932 (n=10) |
| hichunk_with_stop | 0.448 (n=5) | 0.917 (n=10) |
| inex | 0.515 (n=5) | 0.834 (n=10) |
| emte | 0.850 (n=5) | 0.997 (n=10) |
| emte(agg=sum) | 0.850 (n=5) | 0.997 (n=10) |
| emte(agg=max) | 0.850 (n=5) | 0.997 (n=10) |
| emte(agg=mean) | 0.600 (n=5) | 0.865 (n=10) |
| emte(max_regions=1) | 0.067 (n=5) | 0.616 (n=10) |
| emte(tau_up=0.002) | 0.583 (n=5) | 0.997 (n=10) |
| emte(tau_up=0.01) | 0.498 (n=5) | 0.898 (n=10) |
| emte(min_raw_score=1.7) | 0.400 (n=5) | 0.836 (n=10) |
| emte(min_raw_score=2.0) | 0.400 (n=5) | 0.797 (n=10) |
| emte(min_raw_score=2.5) | 0.200 (n=5) | 0.797 (n=10) |

## Speed and memory (selection only; index already built)

| Selector | median ms | p95 ms | peak alloc MB |
|---|---:|---:|---:|
| flat_sections | 5.48 | 13.90 | 0.2 |
| fixed_k | 4.58 | 6.61 | 0.4 |
| score_ratio | 4.19 | 6.79 | 0.4 |
| score_gap | 4.48 | 7.16 | 0.4 |
| adaptive_k | 4.65 | 7.05 | 0.4 |
| section_then_paragraph | 8.08 | 14.35 | 0.4 |
| granularity=paragraph | 23.33 | 55.38 | 0.3 |
| granularity=subsection | 1.76 | 5.40 | 0.2 |
| granularity=section | 6.76 | 10.97 | 0.3 |
| plain_automerge | 8.41 | 15.29 | 0.3 |
| hichunk | 8.85 | 40.56 | 0.4 |
| hichunk_with_stop | 7.91 | 16.85 | 0.8 |
| inex | 45.64 | 83.55 | 0.4 |
| emte | 26.19 | 61.70 | 0.4 |
| emte(agg=sum) | 29.28 | 55.21 | 0.4 |
| emte(agg=max) | 27.19 | 55.35 | 0.4 |
| emte(agg=mean) | 45.95 | 126.87 | 0.4 |
| emte(max_regions=1) | 10.45 | 13.71 | 0.4 |
| emte(tau_up=0.002) | 34.06 | 88.35 | 0.4 |
| emte(tau_up=0.01) | 54.46 | 128.67 | 0.4 |
| emte(min_raw_score=1.7) | 11.61 | 27.24 | 0.4 |
| emte(min_raw_score=2.0) | 8.45 | 27.64 | 0.4 |
| emte(min_raw_score=2.5) | 7.07 | 18.98 | 0.4 |

Index build per document (Member 1 files + Member 2 BM25F): unit1_topic1 50 ms, unit1_topic2 36 ms, unit1_topic3 94 ms

## Out-of-scope questions (should abstain)

Each out-of-scope question is run on every document. A good selector abstains on these and not on in-scope questions.

| Selector | out-of-scope abstained (right) | in-scope abstained (wrong) | tokens@512 given for out-of-scope |
|---|---:|---:|---:|
| flat_sections | 0/12 | 0/15 | 378 |
| fixed_k | 0/12 | 0/15 | 121 |
| score_ratio | 0/12 | 0/15 | 186 |
| score_gap | 0/12 | 0/15 | 236 |
| adaptive_k | 0/12 | 0/15 | 333 |
| section_then_paragraph | 0/12 | 0/15 | 488 |
| granularity=paragraph | 0/12 | 0/15 | 507 |
| granularity=subsection | 4/12 | 6/15 | 58 |
| granularity=section | 0/12 | 0/15 | 506 |
| plain_automerge | 0/12 | 0/15 | 160 |
| hichunk | 0/12 | 0/15 | 285 |
| hichunk_with_stop | 0/12 | 0/15 | 233 |
| inex | 0/12 | 0/15 | 508 |
| emte | 0/12 | 0/15 | 408 |
| emte(agg=sum) | 0/12 | 0/15 | 408 |
| emte(agg=max) | 0/12 | 0/15 | 408 |
| emte(agg=mean) | 0/12 | 0/15 | 418 |
| emte(max_regions=1) | 0/12 | 0/15 | 78 |
| emte(tau_up=0.002) | 0/12 | 0/15 | 295 |
| emte(tau_up=0.01) | 0/12 | 0/15 | 341 |
| emte(min_raw_score=1.7) | 4/12 | 2/15 | 116 |
| emte(min_raw_score=2.0) | 4/12 | 4/15 | 85 |
| emte(min_raw_score=2.5) | 9/12 | 4/15 | 26 |

### Can a score threshold separate them? (top raw BM25F paragraph score)

| min_raw_score | out-of-scope abstained (right) | in-scope abstained (wrong) |
|---:|---:|---:|
| 0 | 0/12 | 0/15 |
| 1.0 | 0/12 | 0/15 |
| 1.5 | 0/12 | 0/15 |
| 1.7 | 4/12 | 2/15 |
| 2.0 | 4/12 | 4/15 |
| 2.5 | 9/12 | 4/15 |
| 3.0 | 10/12 | 5/15 |
| 4.0 | 11/12 | 10/15 |

In-scope top scores: 1.65, 1.68, 1.85, 1.98, 2.69, 3.02, 3.08, 3.30, 3.64, 3.83, 4.16, 4.36, 4.39, 4.39, 4.85

Out-of-scope top scores: 1.51, 1.57, 1.64, 1.65, 2.01, 2.29, 2.33, 2.33, 2.35, 2.63, 3.46, 5.15

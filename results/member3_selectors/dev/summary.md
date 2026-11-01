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
| emte(agg=mean) | 0.568 | 0.776 | 0.834 | 0.533 | 0.093 | 485 | 20.1 | -0.172 (0W/8L) | 0.089 |
| emte(tau_up=0.01) | 0.658 | 0.765 | 0.778 | 0.600 | 0.110 | 434 | 17.9 | -0.183 (0W/7L) | 0.151 |
| hichunk_with_stop | 0.631 | 0.761 | 0.761 | 0.600 | 0.135 | 282 | 9.3 | -0.188 (0W/6L) | 0.206 |
| inex | 0.605 | 0.727 | 0.829 | 0.533 | 0.059 | 508 | 20.0 | -0.221 (0W/7L) | 0.151 |
| granularity=paragraph | 0.547 | 0.668 | 0.778 | 0.400 | 0.077 | 509 | 24.7 | -0.280 (0W/10L) | 0.025 |
| section_then_paragraph | 0.577 | 0.647 | 0.684 | 0.467 | 0.115 | 398 | 20.3 | -0.302 (0W/10L) | 0.019 |
| adaptive_k | 0.533 | 0.579 | 0.595 | 0.333 | 0.128 | 271 | 13.0 | -0.369 (0W/10L) | 0.025 |
| plain_automerge | 0.552 | 0.569 | 0.569 | 0.333 | 0.236 | 132 | 4.3 | -0.379 (0W/9L) | 0.038 |
| score_ratio | 0.487 | 0.524 | 0.524 | 0.267 | 0.128 | 245 | 12.1 | -0.424 (0W/12L) | 0.003 |
| score_gap | 0.425 | 0.461 | 0.490 | 0.267 | 0.145 | 192 | 9.3 | -0.488 (0W/11L) | 0.008 |
| emte(max_regions=1) | 0.433 | 0.433 | 0.433 | 0.400 | 0.264 | 66 | 1.0 | -0.515 (0W/9L) | 0.048 |
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

## Speed and memory (selection only; index already built)

| Selector | median ms | p95 ms | peak alloc MB |
|---|---:|---:|---:|
| flat_sections | 10.85 | 22.23 | 0.3 |
| fixed_k | 6.73 | 10.56 | 0.4 |
| score_ratio | 7.35 | 10.22 | 0.4 |
| score_gap | 7.47 | 11.20 | 0.4 |
| adaptive_k | 8.04 | 10.52 | 0.4 |
| section_then_paragraph | 13.62 | 30.09 | 0.4 |
| granularity=paragraph | 43.75 | 78.72 | 0.3 |
| granularity=subsection | 5.19 | 8.55 | 0.2 |
| granularity=section | 11.56 | 15.86 | 0.2 |
| plain_automerge | 14.77 | 19.17 | 0.2 |
| hichunk | 20.20 | 68.29 | 0.4 |
| hichunk_with_stop | 18.97 | 33.37 | 0.7 |
| inex | 79.15 | 131.82 | 0.4 |
| emte | 53.25 | 87.45 | 0.4 |
| emte(agg=sum) | 45.18 | 80.48 | 0.4 |
| emte(agg=max) | 48.38 | 81.97 | 0.4 |
| emte(agg=mean) | 118.24 | 229.88 | 0.4 |
| emte(max_regions=1) | 16.11 | 22.76 | 0.4 |
| emte(tau_up=0.002) | 73.10 | 159.35 | 0.4 |
| emte(tau_up=0.01) | 99.47 | 201.65 | 0.4 |

Index build per document (Member 1 files + Member 2 BM25F): unit1_topic1 92 ms, unit1_topic2 50 ms, unit1_topic3 117 ms

## Out-of-scope questions (should abstain)

No out-of-scope questions in the dev split.

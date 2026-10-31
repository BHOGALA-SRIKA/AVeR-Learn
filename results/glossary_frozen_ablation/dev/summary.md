# glossary_frozen_ablation (dev split)

15 in-scope questions, 0 out-of-scope, 55 paragraphs. Tokenizer 1.0: nfkc, lowercase, hyphen-join, i/o-kept, lucene+question stopwords, snowball-english, token counter approx-v1, run 2026-10-31T06:29:21+00:00. Sorted by nDCG@10.

## Ranking quality

| method | R@1 | R@3 | R@5 | R@10 | P@3 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| char_tfidf+exp | 0.689 | 0.900 | 0.900 | 0.900 | 0.444 | 0.889 | 0.870 |
| char_tfidf | 0.689 | 0.867 | 0.900 | 0.900 | 0.422 | 0.880 | 0.863 |
| rrf | 0.622 | 0.833 | 0.833 | 0.900 | 0.400 | 0.847 | 0.828 |
| bm25f | 0.622 | 0.811 | 0.833 | 0.867 | 0.378 | 0.840 | 0.821 |
| bm25 | 0.622 | 0.811 | 0.833 | 0.867 | 0.378 | 0.840 | 0.821 |
| rrf+exp | 0.622 | 0.778 | 0.833 | 0.900 | 0.356 | 0.850 | 0.818 |
| bm25+exp | 0.622 | 0.756 | 0.811 | 0.900 | 0.333 | 0.851 | 0.814 |
| bm25f+exp | 0.622 | 0.756 | 0.811 | 0.900 | 0.333 | 0.848 | 0.813 |

## Evidence coverage within a token budget

cov = share of the essential gold-quote words delivered; allEss = share of questions where every essential quote is at least 80% delivered; paras = paragraphs delivered.

| method | cov@256 | allEss@256 | paras@256 | cov@512 | allEss@512 | paras@512 | cov@1024 | allEss@1024 | paras@1024 |
|---|---|---|---|---|---|---|---|---|---|
| char_tfidf+exp | 0.774 | 0.667 | 2.333 | 0.875 | 0.867 | 4.867 | 0.892 | 0.867 | 10.533 |
| char_tfidf | 0.774 | 0.667 | 2.333 | 0.875 | 0.867 | 4.933 | 0.892 | 0.867 | 10.933 |
| rrf | 0.763 | 0.667 | 2.133 | 0.846 | 0.800 | 5.067 | 0.939 | 0.867 | 11.267 |
| bm25f | 0.763 | 0.667 | 2.133 | 0.863 | 0.800 | 5.067 | 0.939 | 0.867 | 10.800 |
| bm25 | 0.763 | 0.667 | 2.133 | 0.863 | 0.800 | 5.067 | 0.939 | 0.867 | 10.800 |
| rrf+exp | 0.766 | 0.667 | 2.333 | 0.863 | 0.800 | 5.200 | 0.939 | 0.867 | 11.267 |
| bm25+exp | 0.766 | 0.667 | 2.200 | 0.834 | 0.667 | 5.133 | 0.939 | 0.867 | 10.667 |
| bm25f+exp | 0.766 | 0.667 | 2.200 | 0.834 | 0.667 | 5.200 | 0.943 | 0.867 | 10.733 |

## nDCG@10 by question group

| method | all (n=15) | student wording (n=5) | book wording (n=10) | type: application (n=3) | type: comparative (n=2) | type: conceptual (n=2) | type: factual (n=8) |
|---|---|---|---|---|---|---|---|
| char_tfidf+exp | 0.870 | 0.628 | 0.992 | 1.000 | 1.000 | 0.069 | 0.990 |
| char_tfidf | 0.863 | 0.621 | 0.984 | 1.000 | 0.960 | 0.053 | 0.990 |
| rrf | 0.828 | 0.652 | 0.916 | 1.000 | 0.775 | 0.130 | 0.952 |
| bm25f | 0.821 | 0.635 | 0.913 | 1.000 | 0.775 | 0.089 | 0.948 |
| bm25 | 0.821 | 0.635 | 0.913 | 1.000 | 0.775 | 0.089 | 0.948 |
| rrf+exp | 0.818 | 0.654 | 0.901 | 1.000 | 0.754 | 0.135 | 0.937 |
| bm25+exp | 0.814 | 0.655 | 0.893 | 1.000 | 0.741 | 0.138 | 0.931 |
| bm25f+exp | 0.813 | 0.653 | 0.893 | 1.000 | 0.741 | 0.132 | 0.931 |

## Speed and memory

Machine: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel | 12 CPUs | 16 GB RAM | Windows-10-10.0.26300-SP0

index KB = what is stored per PDF; model MB = model file shared by all PDFs (0 for lexical). Process peak MB includes Python itself (baseline about 109 MB).

| method | build ms | query ms (median) | query ms (p95) | index KB | model MB | build alloc MB | process peak MB |
|---|---|---|---|---|---|---|---|
| char_tfidf+exp | 61.886 | 1.964 | 2.735 | 426.954 | 0.000 | 1.825 | 114.488 |
| char_tfidf | 42.494 | 1.307 | 2.582 | 422.984 | 0.000 | 1.824 | 114.102 |
| rrf | 84.145 | 6.097 | 7.994 | 497.929 | 0.000 | 2.028 | 115.348 |
| bm25f | 15.867 | 0.253 | 0.498 | 59.760 | 0.000 | 0.192 | 109.406 |
| bm25 | 12.659 | 0.107 | 0.166 | 60.209 | 0.000 | 0.221 | 109.445 |
| rrf+exp | 80.669 | 6.569 | 8.194 | 501.917 | 0.000 | 2.029 | 115.770 |
| bm25+exp | 8.655 | 0.289 | 0.622 | 64.179 | 0.000 | 0.222 | 109.418 |
| bm25f+exp | 14.043 | 0.495 | 0.923 | 63.722 | 0.000 | 0.193 | 109.320 |

## Is it better than bm25f? (nDCG@10, n = 15 questions)

Paired randomization test (10,000 sign flips), Holm-corrected over all methods. Significant = Holm p < 0.05.

| method | diff | 95% CI | wins/losses | p | Holm p | significant |
|---|---|---|---|---|---|---|
| char_tfidf+exp | 0.050 | [-0.016, +0.130] | 5/2 | 0.278 | 1.000 | no |
| char_tfidf | 0.042 | [-0.022, +0.122] | 4/2 | 0.339 | 1.000 | no |
| rrf | 0.008 | [+0.000, +0.021] | 2/0 | 0.494 | 1.000 | no |
| bm25 | 0.000 | [+0.000, +0.000] | 0/0 | 1.000 | 1.000 | no |
| rrf+exp | -0.002 | [-0.019, +0.016] | 1/2 | 1.000 | 1.000 | no |
| bm25+exp | -0.007 | [-0.031, +0.015] | 1/2 | 0.755 | 1.000 | no |
| bm25f+exp | -0.007 | [-0.031, +0.013] | 1/2 | 0.755 | 1.000 | no |

## Is it better than bm25f? (R@5, n = 15 questions)

Paired randomization test (10,000 sign flips), Holm-corrected over all methods. Significant = Holm p < 0.05.

| method | diff | 95% CI | wins/losses | p | Holm p | significant |
|---|---|---|---|---|---|---|
| char_tfidf+exp | 0.067 | [+0.000, +0.167] | 2/0 | 0.497 | 1.000 | no |
| char_tfidf | 0.067 | [+0.000, +0.167] | 2/0 | 0.497 | 1.000 | no |
| rrf | 0.000 | [+0.000, +0.000] | 0/0 | 1.000 | 1.000 | no |
| bm25 | 0.000 | [+0.000, +0.000] | 0/0 | 1.000 | 1.000 | no |
| rrf+exp | 0.000 | [+0.000, +0.000] | 0/0 | 1.000 | 1.000 | no |
| bm25+exp | -0.022 | [-0.067, +0.000] | 0/1 | 1.000 | 1.000 | no |
| bm25f+exp | -0.022 | [-0.067, +0.000] | 0/1 | 1.000 | 1.000 | no |

## Is it better than bm25f? (MRR@10, n = 15 questions)

Paired randomization test (10,000 sign flips), Holm-corrected over all methods. Significant = Holm p < 0.05.

| method | diff | 95% CI | wins/losses | p | Holm p | significant |
|---|---|---|---|---|---|---|
| char_tfidf+exp | 0.049 | [-0.013, +0.133] | 2/1 | 0.493 | 1.000 | no |
| char_tfidf | 0.040 | [-0.013, +0.120] | 2/1 | 0.493 | 1.000 | no |
| rrf | 0.007 | [+0.000, +0.022] | 1/0 | 1.000 | 1.000 | no |
| bm25 | 0.000 | [+0.000, +0.000] | 0/0 | 1.000 | 1.000 | no |
| rrf+exp | 0.010 | [+0.000, +0.029] | 1/0 | 1.000 | 1.000 | no |
| bm25+exp | 0.011 | [+0.000, +0.033] | 1/0 | 1.000 | 1.000 | no |
| bm25f+exp | 0.008 | [+0.000, +0.025] | 1/0 | 1.000 | 1.000 | no |

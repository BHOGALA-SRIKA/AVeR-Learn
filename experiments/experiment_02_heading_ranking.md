# Experiment 02: Heading-Aware Ranking and Token Normalization

## 1. Objective

To investigate whether incorporating section-heading information and token normalization improves the ranking of relevant paragraphs.

## 2. Starting Point

Experiment 01 showed that the query "what are interrupts" retrieved general operating-system content above the relevant introductory paragraph.

The relevant paragraphs identified during inspection were:

- `unit1_topic1_p00019`: Interrupts
- `unit1_topic1_p00020`: Interrupt Handling
- `unit1_topic1_p00021`: Interrupt Timeline

## 3. Modifications Investigated

### 3.1 Heading-Aware Ranking

An additional score component was introduced to reward overlap between query terms and section-heading terms.

### 3.2 Token Normalization

Simple plural normalization was introduced to improve matching between query terms and terms appearing in the indexed material.

## 4. Observed Results

For the query "what are interrupts", the latest observed output returned:

| Rank | Paragraph ID | Content |
|---|---|---|
| 1 | `unit1_topic1_p00020` | Steps in interrupt handling |
| 2 | `unit1_topic1_p00019` | Introduction to interrupts, ISR, traps and exceptions |
| 3 | `unit1_topic1_p00021` | Interrupt timeline |

All three results concerned interrupts, but the introductory paragraph was not ranked first.

## 5. Interpretation

The modifications improved the topical relevance of the returned results compared with the original output, which included unrelated operating-system and storage content.

However, the introductory paragraph did not achieve the desired rank for the broad conceptual query.

The results suggest that lexical matching and heading overlap alone may not always distinguish between a paragraph introducing a topic and one describing a related procedure.

## 6. Limitations

- Evaluation used a very small query set.
- No formal retrieval metrics were calculated.
- The relative contributions of heading-aware ranking and token normalization have not been isolated through controlled comparisons.
- Generalization to other subjects and unseen queries remains untested.

## 7. Next Steps

1. Create a manually labelled benchmark of questions and relevant paragraph IDs.
2. Compare standard BM25 with heading-aware BM25.
3. Evaluate each modification separately.
4. Measure Recall@3 and Mean Reciprocal Rank (MRR).
5. Test the approach on additional Operating Systems and DBMS material.

## 8. Status

Preliminary experiment completed. Further controlled evaluation is pending.
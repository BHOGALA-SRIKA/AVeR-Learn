# Experiment 01: Baseline BM25 Retrieval

## 1. Objective

To implement and evaluate a basic lexical retrieval system for retrieving relevant paragraphs from offline Operating Systems study material.

## 2. Dataset

- Subject: Operating Systems
- Source: `unit1_topic1.pdf`
- Number of PDF pages: 9
- Extracted paragraphs indexed: 28

## 3. Methodology

1. Extract text from the PDF using PyMuPDF.
2. Organize extracted content into paragraphs.
3. Tokenize paragraph text.
4. Build a BM25 index using the `rank-bm25` library.
5. Submit natural-language questions to the retriever.
6. Examine the top three retrieved paragraphs for relevance.

## 4. Test Queries

### Query 1: What are interrupts?

Observation: The initial ranking returned general operating-system content rather than the paragraph specifically introducing interrupts.

Result: Relevant introductory content was not ranked first.

### Query 2: What is a trap or exception?

Observation: The retriever returned a paragraph explaining that a trap or exception is a software-generated interrupt caused by errors or user requests, such as system calls.

Result: Relevant paragraph retrieved at rank 1.

### Query 3: What are the steps involved in interrupt handling?

Observation: The retriever returned a paragraph listing the five interrupt-handling steps.

Result: Relevant paragraph retrieved at rank 1.

## 5. Initial Findings

The baseline BM25 implementation successfully retrieves relevant paragraphs for questions containing distinctive terms.

However, broad conceptual queries can return irrelevant paragraphs above more suitable introductory content.

This indicates a need to investigate ranking improvements and evaluate retrieval on a larger benchmark.

## 6. Limitations

- Only one PDF was evaluated.
- Only three queries were tested.
- Relevance was assessed manually.
- No formal Recall@K or Mean Reciprocal Rank evaluation was performed.

## 7. Conclusion

The baseline confirms that offline paragraph retrieval using BM25 is feasible for the selected study material. Further experiments are needed to determine whether heading-aware ranking and token normalization improve retrieval quality.
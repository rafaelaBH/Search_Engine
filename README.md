# Semantic Symptom Search Engine

## Description
This project is a Python-based semantic symptom analysis engine that analyzes user-described symptoms and matches them to known medical conditions using natural language processing techniques. It features semantic similarity scoring with Sentence Transformers, risk classification based on symptom severity, and structured logging of search queries and results.

## Features
- Symptom dataset processing with severity mapping  
- Semantic similarity matching using **Sentence Transformers** and **cosine similarity**  
- Input parsing for multiple symptoms in free-text queries  
- Risk assessment based on detected symptoms  
- Logging of queries with timestamps, detected symptoms, and confidence scores  
- Graceful handling of unknown or low-confidence inputs  
- Fully version-controlled using Git  

## Build Instructions
This project runs directly in Python. Install dependencies with:

```bash
pip install pandas numpy scikit-learn sentence-transformers

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re
from datetime import datetime


class SymptomSearchEngine:
    MIN_MATCH_SCORE = 0.45  # Minimum cosine similarity score to accept a symptom match

    def __init__(self, data_path, log_path="search_history.txt"):
        self.log_path = log_path
        self.severity_map = {}
        self.symptoms_list = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.symptoms_embeddings = None
        self.__setup(data_path)

    def __setup(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file '{data_path}' missing")

        # creates a data frame
        df = pd.read_csv(data_path, header=None, names=['Symptom', 'Severity'])

        # maps symptom to severity
        self.severity_map = dict(zip(df['Symptom'], df['Severity']))

        # list of symptoms without repetitions
        self.symptoms_list = df['Symptom'].unique().tolist()

        # symptoms as vectors
        self.symptoms_embeddings = self.model.encode(self.symptoms_list)

    def __log_transaction(self, query, result, score):

        # saves the exact time of search (year-month-day-hour-minute-second) as a string
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Query: {query} | Match: {result} | Accuracy: {score:.2f}\n"
        with open(self.log_path, "a") as f:
            f.write(log_entry)

    def conclusion(self, severity):
        if 1 <= severity <= 3:
            return "little to no risk"
        elif 4 <= severity <= 5:
            return "mid risk"
        elif severity >= 6:
            return "high risk"
        else:
            return "cannot determine the level of risk"

    def __input_proccesor(self, given_input: str) -> list:

        # splits user input by key-words to assess each symptom separately
        key_words = r",| and | with | plus"
        user_symptoms = [s.strip() for s in re.split(key_words, given_input.lower()) if s.strip()]
        return user_symptoms

    def severity_assessment(self, given_input: str, k: int = 1):

        user_symptoms = self.__input_proccesor(given_input)
        max_severity = 0
        symptoms_to_return = []
        scores_to_return = []
        unknown = []

        for s in user_symptoms:

            # finding k most likely symptoms
            input_vector = self.model.encode([s])
            scores = cosine_similarity(input_vector, self.symptoms_embeddings)[0]

            # get top-k indices sorted by score descending
            top_k_indices = np.argsort(scores)[-k:][::-1]

            matched_any = False
            for idx in top_k_indices:
                if scores[idx] >= self.MIN_MATCH_SCORE:
                    scores_to_return.append(scores[idx])
                    symptom = self.symptoms_list[idx]
                    symptoms_to_return.append(symptom)
                    severity = self.severity_map[symptom]
                    max_severity = max(max_severity, severity)
                    self.__log_transaction(s, symptom, scores[idx])
                    matched_any = True

            if not matched_any:
                unknown.append(s)

        risk_report = self.conclusion(max_severity)
        return scores_to_return, symptoms_to_return, risk_report, unknown


if __name__ == "__main__":
    # Creates engine
    try:
        engine = SymptomSearchEngine('Symptom_severity.csv')
    except Exception as e:
        print(f"Failure: {e}")
        exit()

    print("\n" + "=" * 55)
    print("  SEMANTIC TEXT SEARCH ENGINE")
    print("  Powered by Sentence-Transformers & Cosine Similarity")
    print("=" * 55)

    # Get k from user once
    while True:
        try:
            k = int(input("\nHow many top results per symptom would you like? (default 1): ").strip() or "1")
            if k >= 1:
                break
            print("Please enter a number >= 1")
        except ValueError:
            print("Please enter a valid number")

    while True:
        given_input = input("\nDescribe symptoms (e.g. 'cough and fever' or 'q' to quit): ").strip()
        if given_input.lower() in ['q', 'quit']:
            print("Quited")
            break

        if not given_input:
            continue

        # Execute assessment
        scores, names, risk, unrecognized = engine.severity_assessment(given_input, k=k)

        print(f"\n[ANALYSIS SUMMARY]")
        print(f"Assessed risk: {risk.upper()}")

        if names:
            for score, name in zip(scores, names):
                # Removes underscores '_'
                display_name = name.replace('_', ' ').title()
                print(f"- Detected: {display_name:<20} | accuracy: {score:.1%}")
        else:
            print("- No valid medical symptoms identified.")

        # In case of an unknown symptom
        if unrecognized:
            print(f"\n[!] Note: The following terms were not recognized:")
            print(f"    {', '.join(unrecognized)}")

        print("-" * 55)

"""
ArvyaX ML Assignment
End-to-End Pipeline: Understand → Decide → Guide
Author: Gourab Barui
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from scipy.sparse import hstack


# CONFIG

TRAIN_URL = "https://docs.google.com/spreadsheets/d/1yLDum7yWr3IH0KivluCBEvqHGlfvFW_S/gviz/tq?tqx=out:csv"
TEST_URL  = "https://docs.google.com/spreadsheets/d/1lCvTufEhGgtDJp6b9oYyFXpCZqWPirSX/gviz/tq?tqx=out:csv"

NUM_COLS = ['sleep_hours', 'energy_level', 'stress_level']



# LOAD DATA

def load_data():
    train = pd.read_csv(TRAIN_URL)
    test = pd.read_csv(TEST_URL)
    return train, test



# PREPROCESSING

def preprocess(train, test):
    train['journal_text'] = train['journal_text'].fillna("")
    test['journal_text'] = test['journal_text'].fillna("")

    for col in NUM_COLS:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

    return train, test



# FEATURE ENGINEERING

def build_features(train, test):
    tfidf = TfidfVectorizer(max_features=5000)

    X_text_train = tfidf.fit_transform(train['journal_text'])
    X_text_test = tfidf.transform(test['journal_text'])

    X_meta_train = train[NUM_COLS].values
    X_meta_test = test[NUM_COLS].values

    X_train = hstack([X_text_train, X_meta_train])
    X_test = hstack([X_text_test, X_meta_test])

    return X_train, X_test



# TRAIN MODELS

def train_models(X_train, train):
    le = LabelEncoder()

    y_state = le.fit_transform(train['emotional_state'])
    y_intensity = train['intensity']

    clf = XGBClassifier(n_estimators=200, max_depth=6, verbosity=0)
    clf.fit(X_train, y_state)

    reg = XGBRegressor(n_estimators=200)
    reg.fit(X_train, y_intensity)

    return clf, reg, le



# DECISION ENGINE

def decide_action(stress, energy, time):
    if stress >= 7:
        return ("rest", "now") if energy <= 4 else ("box_breathing", "now")

    if energy >= 7:
        return "deep_work", "within_15_min"

    if str(time).lower() == "night":
        return "sleep", "tonight"

    return "light_planning", "later_today"



# PREDICTION PIPELINE
def predict(clf, reg, le, X_test, test):
    probs = clf.predict_proba(X_test)

    state_idx = np.argmax(probs, axis=1)
    states = le.inverse_transform(state_idx)

    intensity = np.clip(np.round(reg.predict(X_test)), 1, 5).astype(int)

    confidence = np.max(probs, axis=1)
    uncertain_flag = (confidence < 0.5).astype(int)

    actions, timings = [], []

    for i in range(len(test)):
        action, timing = decide_action(
            test.iloc[i]['stress_level'],
            test.iloc[i]['energy_level'],
            test.iloc[i]['time_of_day']
        )
        actions.append(action)
        timings.append(timing)

    return states, intensity, confidence, uncertain_flag, actions, timings



# SAVE & SHOW OUTPUT

def save_output(test, states, intensity, confidence, uncertain_flag, actions, timings):
    output = pd.DataFrame({
        "id": test['id'],
        "predicted_state": states,
        "predicted_intensity": intensity,
        "confidence": confidence,
        "uncertain_flag": uncertain_flag,
        "what_to_do": actions,
        "when_to_do": timings
    })

    output.to_csv("predictions.csv", index=False)

    print("\n" + "="*60)
    print("SAMPLE OUTPUT (FIRST 10 ROWS)")
    print("="*60)
    print(output.head(10).to_string(index=False))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total rows        : {len(output)}")
    print(f"Uncertain cases   : {output['uncertain_flag'].sum()}")
    print(f"Avg confidence    : {round(output['confidence'].mean(), 2)}")

    print("\n" + "="*60)
    print("UNCERTAIN CASES (TOP 5)")
    print("="*60)
    print(output[output['uncertain_flag'] == 1].head(5).to_string(index=False))

    print("\nFile saved as: predictions.csv")



# MAIN

def main():
    print("Starting ArvyaX Pipeline...\n")

    train, test = load_data()
    train, test = preprocess(train, test)

    X_train, X_test = build_features(train, test)

    clf, reg, le = train_models(X_train, train)

    states, intensity, confidence, uncertain_flag, actions, timings = predict(
        clf, reg, le, X_test, test
    )

    save_output(test, states, intensity, confidence, uncertain_flag, actions, timings)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
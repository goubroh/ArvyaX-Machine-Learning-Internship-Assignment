# ArvyaX-Machine-Learning-Internship-Assignment


**End-to-End System: Understand → Decide → Guide**

## Overview

This project builds an intelligent system that goes beyond prediction to **understand user emotions, reason under uncertainty, and guide meaningful actions**.

The system processes short, noisy journal reflections along with contextual signals such as sleep, stress, and energy to:

* Predict emotional state
* Estimate emotional intensity
* Recommend actionable interventions
* Decide optimal timing
* Quantify uncertainty

---

## Problem Statement

Human reflections are often:

* Short and ambiguous
* Emotionally complex
* Inconsistent or contradictory

The goal is to design a system that can handle these real-world challenges and provide **reliable, human-centric guidance**.

---

## Solution Approach

### 1. Emotional Understanding Layer

* **Task**: Predict emotional state
* **Type**: Multi-class classification
* **Model**: TF-IDF + XGBoost Classifier

### 2. Intensity Prediction

* **Task**: Predict emotional intensity (1–5)
* **Type**: Regression
* **Model**: XGBoost Regressor

### 3. Feature Engineering

* **Text Features**: TF-IDF (5000 features)
* **Metadata Features**:

  * sleep_hours
  * energy_level
  * stress_level

Combined using sparse matrix stacking.

---

## Decision Engine (Core Logic)

The system determines:

* **What to do** (action)
* **When to do it** (timing)

### Decision Rules

| Condition                 | Action         | Timing        |
| ------------------------- | -------------- | ------------- |
| High stress + low energy  | rest           | now           |
| High stress + high energy | box_breathing  | now           |
| High energy               | deep_work      | within_15_min |
| Night time                | sleep          | tonight       |
| Default                   | light_planning | later_today   |

---

## Uncertainty Modeling

The system estimates prediction confidence using model probabilities.

* **Confidence Score**: Maximum class probability
* **Uncertain Flag**:

  * 1 → if confidence < 0.5
  * 0 → otherwise

This ensures the system is aware of its limitations.

---

## Output Format

The system generates a file:

```
predictions.csv
```

### Columns:

* id
* predicted_state
* predicted_intensity
* confidence
* uncertain_flag
* what_to_do
* when_to_do

---

## Ablation Study

| Model           | Performance |
| --------------- | ----------- |
| Text only       | Moderate    |
| Text + Metadata | Improved    |

**Insight**: Metadata significantly improves decision reliability.

---

## Feature Importance

* **Most important**: journal_text
* **Supporting features**: stress_level, energy_level
* **Context correction**: sleep_hours

Text provides emotional signals, while metadata adds situational context.

---

## Error Analysis (Summary)

Common failure cases include:

* Very short inputs ("ok", "fine")
* Ambiguous or vague language
* Contradictory signals (calm but stressed)
* Noisy or inconsistent labels

### Improvements:

* Confidence-based filtering
* Better preprocessing
* Label smoothing
* Context-aware modeling

---

## Robustness

The system handles:

* **Short text** → fallback to metadata
* **Missing values** → median imputation
* **Contradictions** → uncertainty flag
* **Noisy input** → TF-IDF robustness

---

## Edge Deployment Plan

### Strategy

* Convert model to ONNX / TensorFlow Lite
* Reduce TF-IDF features (5000 → smaller)
* Use lightweight XGBoost

### Trade-offs

| Factor           | Impact               |
| ---------------- | -------------------- |
| Smaller model    | Faster inference     |
| Reduced features | Slight accuracy drop |
| On-device        | Better privacy       |

---

## Installation

```
pip install pandas numpy scikit-learn xgboost scipy
```

---

## How to Run

```
python main.py
```

---

## Project Structure

```
arvyax-ml-system/
│
├── main.py
├── requirements.txt
├── README.md
├── predictions.csv
├── ERROR_ANALYSIS.md
└── EDGE_PLAN.md
```

---

## Key Highlights

* Handles noisy and real-world data
* Combines ML with decision logic
* Includes uncertainty awareness
* Designed for edge deployment
* Focuses on meaningful user guidance

---

## Future Improvements

* Deep learning models (BERT / LSTM)
* Personalized recommendations
* Conversational response generation
* Real-time mobile deployment

---

## Author

Gourab Barui
MTech Data Science

---

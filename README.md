# Clinical NLP Risk Classification for Sickle Cell Emergency Department Notes

## Project Summary

This project applies natural language processing and machine learning to classify sickle cell emergency department visits using clinical notes and structured visit-level features.

The work focuses on two related binary classification tasks:

1. predicting whether a patient visit resulted in admission;
2. predicting whether a visit met a clinically defined high-risk profile.

The project combines free-text clinical notes with structured clinical variables such as haemoglobin, oxygen saturation, pain intensity, previous visits, previous admissions, and changes in laboratory values over time. Traditional machine learning models, LSTM-based deep learning models, and ClinicalBERT-based models were compared to assess how different modelling approaches perform on the two clinical classification tasks.

## Clinical Motivation

Sickle cell disease can lead to acute painful crises and other complications that require timely clinical assessment. In emergency and urgent care settings, clinicians often need to identify patients who may be at higher risk based on symptoms, laboratory findings, and clinical history.

This project explores whether machine learning models can use clinical text and structured clinical features to identify high-risk visits. The goal is not to replace clinical judgement, but to demonstrate how clinical NLP and predictive modelling can support risk stratification, triage research, and healthcare analytics.

## Dataset

The project uses the **Sickle Cell Clinical Notes** dataset from Kaggle.

The dataset contains visit-level records for patients with sickle cell disease. Each record includes structured variables such as:

* patient ID;
* visit date;
* facility type;
* pain type;
* pain intensity;
* admission status;
* haemoglobin;
* oxygen saturation;
* date of birth;
* diagnosis date;
* gender;
* free-text clinical note.

After preprocessing, the dataset contained **75,016 visits** from **5,000 patients**.

Two target variables were used:

* `admitted_bin`: the original admission outcome, indicating whether the visit resulted in admission;
* `high_risk`: an engineered clinical risk label based on haemoglobin, oxygen saturation, and pain intensity.

The original admission outcome was almost balanced, with approximately 50.1% of visits resulting in admission. The engineered high-risk label was positive in approximately 59.5% of visits.

## High-Risk Label Definition

A visit was classified as high risk if at least one of the following conditions was present:

* haemoglobin below 8.0 g/dL;
* oxygen saturation below 90%;
* pain intensity of 8 or above.

This label was created to represent a simple clinically interpretable risk profile using variables available in the dataset.

## Feature Engineering

Several clinical, temporal, and text-derived features were engineered before modelling.

### Clinical and Patient-History Features

The structured feature set included:

* age at visit;
* years since diagnosis;
* number of previous visits;
* number of previous admissions;
* days since last visit;
* previous haemoglobin;
* change in haemoglobin from previous visit;
* previous oxygen saturation;
* change in oxygen saturation from previous visit;
* pain intensity.

### Text-Derived Features

Information was also extracted from the clinical notes, including:

* genotype patterns;
* severity keywords;
* mild/severe/worsening symptom indicators;
* shortness of breath indicators;
* cleaned note text for NLP modelling.

## Exploratory Data Analysis

Exploratory analysis was performed to understand the structure of the dataset and the relationship between clinical features and the target labels.

The analysis included:

* class distribution for admission and high-risk labels;
* note length distribution;
* word frequency analysis;
* distributions of age, years since diagnosis, haemoglobin, oxygen saturation, and pain intensity;
* admission and high-risk rates by facility type;
* correlations between numeric features and target labels.

The analysis suggested that the engineered high-risk label was more directly related to clinical variables such as haemoglobin, oxygen saturation, and pain intensity, while the admission label appeared harder to predict from the available features alone.

## Train/Test Split

To reduce information leakage, the train/test split was performed at patient level rather than row level.

This means that all visits for a given patient were assigned either to the training set or the test set, but not both. This approach is more appropriate for repeated-visit healthcare data, where the same patient can appear multiple times.

The final split contained:

* 60,029 training visits;
* 14,987 test visits.

## Models Compared

The project compared several modelling approaches.

### Traditional Machine Learning

Traditional models were trained using TF-IDF features from the clinical notes combined with engineered structured features.

Models included:

* Logistic Regression;
* Linear Support Vector Machine.

### LSTM Models

LSTM-based neural networks were trained to model the cleaned clinical note text, with structured features incorporated into the modelling workflow.

Both admission and high-risk classification tasks were evaluated.

### ClinicalBERT + MLP

A ClinicalBERT-based approach was also explored using `emilyalsentzer/Bio_ClinicalBERT`.

ClinicalBERT was used to generate contextual embeddings from the raw clinical notes. These embeddings were combined with engineered numeric features and passed into a multi-layer perceptron classifier.

Due to computational cost, the ClinicalBERT experiments were run on a subset of the data.

## Key Findings

The admission prediction task was difficult across models. This suggests that admission decisions may depend on factors not fully captured in the dataset, such as clinician judgement, hospital policy, bed availability, pain management response, and wider clinical context.

The engineered high-risk prediction task was much more learnable. This is expected because the high-risk label was defined using clinical indicators available in the dataset.

The LSTM-based models performed strongly on the high-risk classification task, achieving high accuracy and ROC-AUC. ClinicalBERT with an MLP showed weaker performance on the subset experiment, with a tendency to favour the high-risk class. This suggests that the ClinicalBERT subset model captured some signal, but did not perform as strongly as the LSTM model trained on the fuller dataset.

## Interpretation

The results show the importance of carefully defining the clinical prediction target.

The original admission label reflects a real operational outcome, but it may not be determined by clinical note content and basic structured features alone. In contrast, the engineered high-risk label is more directly tied to measurable clinical indicators, making it more predictable from the available data.

This distinction is important in healthcare data science. A model can only learn patterns that are present in the data. If a target depends on unobserved factors, even advanced models may perform poorly.

## Limitations

This project has several limitations:

* the dataset is from Kaggle and may not fully represent real-world clinical complexity;
* the high-risk label was rule-based and simplified;
* ClinicalBERT was evaluated on a subset due to computational constraints;
* admission decisions may depend on unrecorded factors;
* the project is for educational and portfolio purposes and is not intended for clinical deployment.

## Skills Demonstrated

This project demonstrates practical skills in:

* clinical NLP;
* healthcare data science;
* Python data analysis;
* text preprocessing;
* feature engineering;
* patient-level train/test splitting;
* TF-IDF modelling;
* traditional machine learning;
* LSTM-based deep learning;
* transformer-based ClinicalBERT embeddings;
* model evaluation using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC;
* interpretation of model behaviour in a clinical context.

## Tools and Libraries

The project used:

* Python;
* pandas;
* NumPy;
* scikit-learn;
* TensorFlow/Keras;
* Hugging Face Transformers;
* Matplotlib;
* WordCloud.

## Repository Structure

```text
clinical-nlp-sickle-cell-risk-classification/
│
├── notebooks/
│   ├── clinical_nlp_sickle_cell_risk_classification.ipynb
│   └── clinicalbert_sickle_cell_risk_classification.ipynb
│
├── data/
│   └── README.md
│
├── outputs/
│   ├── model_metrics_summary.csv
│   └── model_metrics_summary_all.csv
│
├── README.md
└── requirements.txt
```

## How to Run

1. Clone the repository.

git clone https://github.com/your-username/clinical-nlp-sickle-cell-risk-classification.git
cd clinical-nlp-sickle-cell-risk-classification

2. Install the required packages.

pip install -r requirements.txt

3. Download the dataset from Kaggle and place it in the `data/` folder.

4. Open the notebook in Jupyter or Google Colab.

jupyter notebook


Due to the computational cost of the ClinicalBERT section, the notebook preserves the evaluated outputs used for comparison. Users may rerun the ClinicalBERT section if sufficient compute resources are available.

## Project Status

This project was developed as part of a wider healthcare data science portfolio. Future improvements could include:

* testing additional transformer models;
* using cross-validation with patient-level grouping;
* adding SHAP or other interpretability methods;
* evaluating calibration for risk prediction;
* comparing models on external or more realistic clinical datasets.

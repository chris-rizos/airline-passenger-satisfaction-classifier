---
title: Airline Passenger Satisfaction Classifier
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
---

# Airline Passenger Satisfaction Classifier — UoA Advanced Python Assignment

Advanced optional assignment for the **University of Athens BIS-Analytics** MSc course *"Python for Data Science, Machine Learning and Artificial Intelligence"*.

This project deploys a classification model that predicts **airline passenger satisfaction** based on passenger, travel and service-related features.

The app follows the **train-once / serve-many** deployment pattern. All model training happens offline in `train_and_save_model.ipynb`; `app.py` only loads the fitted artifact and serves predictions. The app never calls `.fit()`.

The deployed Gradio app contains three tabs:

* **EDA** — descriptive statistics and simple exploratory plots of the bundled sample.
* **Model Card** — static comparison of the three candidate algorithms trained offline on the same train/test split, plus the winner's name and a short justification. The numbers come from `model_comparison.csv`.
* **Predict** — enter a new passenger profile, select one of the fitted pipelines, and get a predicted satisfaction class with confidence. The default model is the F1-score winner loaded from `model.joblib`.

## Files

| File                                             | Purpose                                                                                                                     |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `app.py`                                         | Gradio app served on Hugging Face Spaces — load and predict only, no training                                               |
| `train_and_save_model.ipynb`                     | Offline training notebook — fits all candidate pipelines, selects the winner and writes the artifacts                       |
| `model.joblib`                                   | Bundle of all fitted `Pipeline`s plus metadata, such as feature names, target name, winner name and justification           |
| `model_comparison.csv`                           | Per-algorithm test-set scores, rendered statically in the Model Card tab                                                    |
| `data/airline_passenger_satisfaction_sample.csv` | Bundled sample of the airline passenger satisfaction dataset used by the app                                                |
| `requirements.txt`                               | Minimal dependencies for the Space; `scikit-learn` is required by `joblib.load`, even though `app.py` does not train models |

## Dataset

The app uses a smaller bundled sample of the airline passenger satisfaction classification dataset used in the mandatory final assignment.

The full dataset is not needed at runtime. The deployed app only needs the bundled sample for lightweight EDA display, the saved fitted model bundle and the model comparison table.

The target variable is `satisfaction`, with two possible classes:

* `satisfied`
* `neutral or dissatisfied`

The features include passenger characteristics, travel information, service ratings and delay information.

## Model Training

The model training process is performed in `train_and_save_model.ipynb`.

The notebook:

1. loads the bundled sample dataset,
2. creates one train/test split used by all candidate models,
3. defines three candidate classification pipelines,
4. fits each pipeline offline,
5. compares the models using test-set performance,
6. selects the F1-score winner,
7. saves the fitted pipelines and metadata to `model.joblib`,
8. saves the comparison table to `model_comparison.csv`.

The Gradio app does not retrain the models. It only loads the saved files.

## Reproducing the Model Locally

In a virtual environment with the listed requirements:

```bash
jupyter lab train_and_save_model.ipynb   # run all cells → writes model.joblib and model_comparison.csv
python app.py                            # serves locally on http://127.0.0.1:7860
```

## Deployment Logic

The project is designed for Hugging Face Spaces.

The expected folder structure is:

```text
app.py
train_and_save_model.ipynb
model.joblib
model_comparison.csv
requirements.txt
README.md
data/airline_passenger_satisfaction_sample.csv
```

The full original training dataset is not required for deployment.

## License

Code: MIT. Dataset: see the [original Kaggle source](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)- sample reproduced here for educational purposes.

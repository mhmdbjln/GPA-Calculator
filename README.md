# Student GPA Prediction with a Multi-Layer Perceptron

This project uses a supervised regression model to estimate students' Grade Point Average (GPA) from selected demographic and academic features. The implementation is presented in `notebook/AVG.ipynb`.

## Repository contents

- `notebook/AVG.ipynb`: data preparation, model training, evaluation, and prediction workflow.
- `data/train.csv` and `data/test.csv`: original training and test datasets.
- `data/processed_train.csv` and `data/processed_test.csv`: processed dataset files used or generated during the workflow.

## Method

The notebook:
1. Separates GPA from the input features.
2. Drops `StudentID`, `ParentalSupport`, and `ParentalEducation` in the current workflow.
3. Maps `Ethnicity` categories to integer values.
4. Applies `MinMaxScaler` to `StudyTimeWeekly`.
5. Splits the training data into training and validation subsets with `test_size=0.3` and `random_state=42`.
6. Trains an `MLPRegressor` with hidden layers `(100, 50)`, `tanh` activation, and up to 1,000 iterations.
7. Predicts GPA values for the separate test CSV.

## Reported result

The notebook reports an **R² score of approximately 0.9112** on its internal hold-out split.

**Metric clarification:** The notebook's summary refers to this result as “91% accuracy”, but the model is a regressor and the code computes `r2_score`. It should therefore be described as an R² score, not classification accuracy.

## Requirements

- Python 3
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install dependencies and open the notebook:

```bash
pip install -r requirements.txt
jupyter notebook notebook/AVG.ipynb
```

## Implementation note

The current notebook fits a separate `MinMaxScaler` on the test data rather than reusing the scaler fitted on the training data. This is a limitation of the current implementation and should be addressed before using the model for reliable inference.

## Data and license

The CSV files are included in this repository. Confirm the dataset's original source and redistribution terms before further distribution. No software license has been specified.

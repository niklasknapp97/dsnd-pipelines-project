# Fashion Forward Review Classification

![Alt Text](gfx/title_image.png)

This project develops a machine learning pipeline for **Fashion Forward**, a fictional fashion brand. The goal is to classify whether a customer review recommends the product (label `1`) or not (label `0`). It leverages text data, performs NLP preprocessing, and trains various models to achieve strong classification performance.

---

## Getting Started

These instructions will help you set up the project on your local machine for development and testing.

### Dependencies

```bash
numpy
pandas
scikit-learn
matplotlib
seaborn
nltk
```

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/stylesense-review-classifier.git
cd stylesense-review-classifier
```

### Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`
```

### Install the required packages:

```bash
pip install -r requirements.txt
```

## Testing

The model was evaluated using test data via the trained Random Forest Classifier. Metrics include accuracy, precision, recall, and F1-score.

### Break Down Tests

- classification_report is used to show precision, recall, and F1-score for both classes.

- Accuracy is calculated on test data to validate model generalization.

- A confusion matrix or additional metrics can be added for further insight.

```python
from sklearn.metrics import classification_report

y_pred = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Project Instructions

This notebook includes:

- **Data Inspection:** Shows structure, missing values, and statistical distributions.

- **Text Feature Engineering:** Adds review length and analyzes most common words.

- **Visualization:** Histogram of review lengths and label distribution.

- **Preprocessing:** Converts text using TF-IDF and removes noise.

- **Model Selection:** Tests multiple models and selects the best (Random Forest).

- **Hyperparameter Tuning:** Optimizes performance using GridSearchCV.

- **Final Evaluation:** Assesses best model on test data.

## Built With

- **scikit-learn** - Machine learning models and preprocessing

- **pandas** - Data manipulation

- **matplotlib** - Visualizations

- **seaborn** - Advanced statistical plotting

- **nltk** - Natural language processing tools

- **Jupyter Notebook** - Interactive coding environment

## License

[License](LICENSE.txt)

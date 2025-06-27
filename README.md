# Language Detection with TfidfVectorizer, Naive Bayes, Logistic Regression, and LinearSVC

This project is an NLP pipeline for **language detection** on short text samples, using classic machine learning algorithms. It demonstrates multiclass text classification with character-level n-grams and three classifiers: Multinomial Naive Bayes, Logistic Regression, and LinearSVC.

---

## Features

* Clean and explore language data
* TF-IDF vectorization with character-level n-grams
* Multinomial Naive Bayes classifier
* Logistic Regression classifier
* LinearSVC classifier
* Confusion matrix visualization
* Evaluation metrics: accuracy, precision, recall, F1-score

---

## Dataset

* Each language in this dataset contains 1000 rows/paragraphs.
* Each sample includes the text and its language label
* Classes are balanced, with roughly equal examples per language
* Source: [Language Identification Dataset on Kaggle](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst)

---

## Workflow

1. Load and inspect the dataset
2. Split data into training and testing sets
3. Vectorize the text using `TfidfVectorizer` with character-level n-grams
4. Train and evaluate:

   * Multinomial Naive Bayes
   * Logistic Regression
   * LinearSVC
5. Assess models with accuracy, precision, recall, and F1-score
6. Visualize misclassifications with a confusion matrix

---

## Requirements

* Python 3.x
* pandas
* scikit-learn
* seaborn
* matplotlib

Install them with:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## Future Work

* Hyperparameter tuning with `GridSearchCV`
* Trying other classifiers (e.g., Random Forest, XGBoost)
* Experimenting with different n-gram ranges or word-level embeddings
* Detailed error analysis for similar languages
* Deep learning approaches for comparison


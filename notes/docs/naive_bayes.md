---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

<!-- (about_py)= -->

# Naive Bayes Classification Problem:

Classification problem is a supervised learning procedure to be able
to label all data points as being part of a class. A bayesian
perspecitive to classification problem is that we can calculate
posterior probabilities of being from different classes given the data
and assuming a prior. 

The reason it is called Naive is because it considers naively the
assumption of conditional independence between features given the labels.


Naive Bayes is a kind of supervised learning procedure as we consider
a training dataset on which the model learns relation between features
and label using a probabilistic model and then assigns labels using
the model.


Often when the label or the class is given, we can find the
distribution of features, however, in Bayes method, we try to use the
Bayes theorem to find the reverse probabilities, i.e., probability of
label given the features.

**Bayes Theorem:**
$$P(Label|X_1, X_2,..X_n) = \frac{\prod{P(X_{i}|Label)}.
P(Label)}{P(X_1, X_2, ... X_n)}$$

We often term $P(X_1, X_2,..X_n|Label)$ as the likelihood, $P(Label)$ asthe
prior probability of class, $P(X_1, X_2,..X_n)$ as the predictor prior
probability and lastly, $P(Label|X_1, X_2,..X_n)$ as the posterior
probability.

If the probability of a given label is more
than the others, we choose that label for a given object.  Sometimes,
we can also look at the ratio of probabilities of various labels to
see if it exceeds 1 or similarly look at the proportionalities using
only the numerator. The reason to look at ratio or proportionalities is to avoid
calculations of $P(X_1, X_2,..X_n)$

**Kinds of Bayes Procedures:**

There are three types of Naive Bayes Classifiers in `Scikit-learn` based
on likelihood kernel for the features given label:

- `GaussianNB` - likelihood is assumed to be normal
- `MultinomialNB` - used for multinomial data, especially used in text
  classification and is characterized by $\theta_y = (\theta_{1y},
  \theta_{2y}, ... \theta_{ny})$
- `ComplementNB` - An alternate to `MultinomialNB` in imbalanced
  datasets. (Often outperforms `Multinomial` in text classification)
- `BernoulliNB` - Assumes each feature is binary-valued.
- `CategoricalNB` - Assumes each feature has its own categorical
  distribution

+++

## Toy Example

- We first load the dataset from sklearn.datasets. 
- Note the `dir(data)` suggests that it has data, target which attribute to X and y.
- This suggests the pattern for kind of dataset that is accepted for naive bayes fit in `Sklearn`

```{code-cell} ipython3
# !pip install sklearn
import sklearn

#Loading the Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs

data = load_breast_cancer()
print(type(data))
print(dir(data))
```

```{code-cell} ipython3
print(data.feature_names)
print(data.target_names)
```

### Train/Test Split
- Using test_size, and random_state, we split the data (X) and target (Y) into train and test.

```{code-cell} ipython3
## Divide into train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2, random_state=20)
```

### Fitting the Training Data to Gaussian NB

- We fit using `.fit` method to training data.
- We can access probabilities using `predict_proba` method, but according to Scikit learn page it is found: "although Naive Bayes is a decent classifier, it is a bad estimator and thus probabilities are not to be taken seriously"

```{code-cell} ipython3
from sklearn.naive_bayes import GaussianNB
 
#Calling the Class
model = GaussianNB()
 
#Fitting the data to the classifier
model.fit(X_train , y_train)
```

```{code-cell} ipython3
## If you want to access probabilities for X_train or X_test, use the following:
## Note that the following output is an ndarray and taking the first 6 rows. 
model.predict_proba(X_test)[1:6,:]
```

### Predicting based on the fit on X_test

```{code-cell} ipython3
model_pred = model.predict(X_test)   ## Predicted Probabilities.
```

### Metrics:

- For multiclass, there are more options especially the kind of averaging. If you want to know more use `help(metrics.f1_score)`
- I am doing precision, recall and f1_score amongst the many metrics that it has.
- More metrics can be found at: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

```{code-cell} ipython3
from sklearn import metrics

print(f"Precision: {metrics.precision_score(y_test, model_pred)}")
print(f"Recall: {metrics.recall_score(y_test, model_pred)}")
print(f"F1 Score: {metrics.f1_score(y_test, model_pred)}")
print(f"AUC: {metrics.roc_auc_score(y_test, model_pred)}")
```

## Credit Card Example 

- I assume that the dataset for credit card (150 MB) is stored under data locally as `creditcard.csv`

```{code-cell} ipython3
import pandas as pd
## Import Dataset
data = pd.read_csv ('../data/creditcard.csv')
data.head()
```

```{code-cell} ipython3
## Encodes categorical text into understandable labels for machine learning (not really useful in this example)
from sklearn.preprocessing import LabelEncoder
encoded_data = data.apply(LabelEncoder().fit_transform)
```

### Test/Train Split

- One can also split based on time variable to ensure that test dataset consists of recent data.
- However, here we're using the sklearn's function.

```{code-cell} ipython3
## Divide into train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(encoded_data.drop(['Class'], axis = 1), encoded_data['Class'], test_size = 0.2, random_state=20)
```

### Fitting on Training

```{code-cell} ipython3
## Calling the Class
model = GaussianNB()
 
#Fitting the data to the classifier
model.fit(X_train , y_train)
```

### Predicting class for test dataset

```{code-cell} ipython3
model_pred = model.predict(X_test)   ## Predicted Probabilities.
```

### Metrics

```{code-cell} ipython3
from sklearn import metrics

print(f"Precision: {metrics.precision_score(y_test, model_pred)}")
print(f"Recall: {metrics.recall_score(y_test, model_pred)}")
print(f"F1 Score: {metrics.f1_score(y_test, model_pred)}")
print(f"AUC: {metrics.roc_auc_score(y_test, model_pred)}")
```

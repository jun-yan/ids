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

## Text documents

We can also work on text documents especially using `MultinomialNB` to determine and classify text. This can be used in identifying SPAM emails etc. 

I am considering the example noted in: [Scikit Learn Website](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

```{code-cell} ipython3
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
```

We download the dataset called 20newsgroups and consider all text corresponding to four categories.

```{code-cell} ipython3
print(f"Target: {twenty_train.target_names}")
print(f"data: {len(twenty_train.data)}")
print(f"Type of training data: {type(twenty_train)}")
print(f"Type of data component: {type(twenty_train.data)}")
```

```{code-cell} ipython3
twenty_train.data[1]  ## just to give an idea of the data. (It is a list)
```

```{code-cell} ipython3
twenty_train.target[:10]
```

```{code-cell} ipython3
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])
```

## Using Bag of Words

- We use the words in each text in training dataset and construct a dictionary mapped to integer indices.
- For each text, we can count the instance of words and let that determine using NB method.

We use `CountVectorizer` and CountVectorizer supports counts of N-grams of words or consecutive characters. Once fitted, the vectorizer has built a dictionary of feature indices.

```{code-cell} ipython3
## Occurence count

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
```

```{code-cell} ipython3
X_train_counts[:10, :10]
```

```{code-cell} ipython3
## Frequency count

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
```

## Training a Classifier (NB method)

Now that we have the frequency data in a sparse data, and also our targets, we can train a classifier using NB method. 

```{code-cell} ipython3
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
```

```{code-cell} ipython3
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
X_new_tfidf.shape
```

```{code-cell} ipython3
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
     print('%r => %s' % (doc, twenty_train.target_names[category]))
```

## Building a Pipeline

In order to make the vectorizer => transformer => classifier easier to work with, scikit-learn provides a Pipeline class that behaves like a compound classifier:

```{code-cell} ipython3
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
```

```{code-cell} ipython3
text_clf.fit(twenty_train.data, twenty_train.target) 
```

## Predicting and checking metrics

```{code-cell} ipython3
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)
```

```{code-cell} ipython3
from sklearn import metrics
metrics.confusion_matrix(twenty_test.target, predicted)
```

---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Measures of classification accuracy and functions in Python

source: [Machine Learning Mastery](https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/)

Classification accuracy is the total number of correct predictions divided by the  
total number of predictions made for a dataset.  

As a performance measure, accuracy is inappropriate for imbalanced classification problems.  

The main reason is that the overwhelming number of examples from the majority class (or classes)  
will overwhelm the number of examples in the minority class, meaning that even unskillful models  
can achieve accuracy scores of 90 percent, or 99 percent, depending on how severe  
the class imbalance happens to be.  

An alternative to using classification accuracy is to use **precision** and **recall** metrics.

After completing this tutorial, you will know:

\* **Precision** quantifies the number of positive class predictions that actually  
belong to the positive class.   

\* **Recall** quantifies the number of positive class predictions made out of all   
positive examples in the dataset.  

\* **F-Measure** provides a single score that balances both the concerns of precision  
and recall in one number.

## Confusion Matrix for Imbalanced Classification

Before we dive into precision and recall, it is important to review the [confusion matrix](https://machinelearningmastery.com/confusion-matrix-machine-learning/).  

For classification problems, the majority class is typically referred to as the  
negative outcome (e.g. such as “no change” or “negative test result“), and the  
minority class is typically referred to as the positive outcome (e.g. “change”   
or “positive test result”).




The confusion matrix provides more insight into not only the performance  
of a predictive model, but also which classes are being predicted correctly,  
which incorrectly, and what type of errors are being made.  

The **simplest confusion matrix** is for a two-class classification problem, with  
**negative (class 0) and positive (class 1) classes**.  

In this type of confusion matrix, each cell in the table has a specific and   
well-understood name, summarized as follows:  


|               | Positive Prediction | Negative Prediction|
| ------------- | ------------------- | ------------------ |
|Positive Class | True Positive (TP)  | False Negative (FN)|
|Negative Class | False Positive (FP) | True Negative (TN) |

The **precision and recall metrics** are defined in terms of the cells in  
the confusion matrix, specifically terms like true positives and false negatives.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/525px-Precisionrecall.svg.png" alt="Precision and Recall" width="400" height="400" /> 
(Image by Wikipedia)  

## Precision metric

**Precision** is a metric that quantifies the number of correct positive predictions made.  

**Precision**, therefore, calculates the accuracy for the **minority** class.

It is calculated as the **ratio of correctly predicted positive examples divided by the  
total number of positive examples that were predicted**.

### Precision for Binary Classification

In an imbalanced classification problem with two classes,   
**precision** is calculated as **the number of true positives divided by  
the total number of true positives and false positives**.  

\* **Precision = TruePositives / (TruePositives + FalsePositives)**  

The result is a value between 0.0 for no precision and 1.0 for full or perfect precision.

A model makes predictions and predicts 120 examples as belonging  
to the minority class, 90 of which are correct, and 30 of which are incorrect.  

The precision for this model is calculated as:  

Precision = TruePositives / (TruePositives + FalsePositives)  
Precision = 90 / (90 + 30)  
Precision = 90 / 120  
Precision = 0.75  

The result is a precision of 0.75, which is a reasonable value but not outstanding.  

You can see that precision is simply the ratio of correct positive predictions out  
of all positive predictions made, or the accuracy of minority class predictions.  

Consider the same dataset, where a model predicts 50 examples belonging to the minority   
class, 45 of which are true positives and five of which are false positives. We can  
calculate the precision for this model as follows:  

Precision = TruePositives / (TruePositives + FalsePositives)  
Precision = 45 / (45 + 5)  
Precision = 45 / 50  
Precision = 0.90  
In this case, although the model predicted far fewer examples as belonging to the  
minority class, the ratio of correct positive examples is much better.  

This highlights that although precision is useful, it does not tell the whole story.   
It does not comment on how many real positive class examples were predicted as belonging  
to the negative class, so-called false negatives.

###  Calculate Precision With Scikit-Learn

The precision score can be calculated using the [precision_score() scikit-learn function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html).  

For example, we can use this function to calculate precision for the scenarios in the previous section.  

First, the case where there are 100 positive to 10,000 negative examples, and a model   
predicts 90 true positives and 30 false positives. The complete example is listed below.

```{code-cell} ipython3
# calculates precision for 1:100 dataset with 90 tp and 30 fp
# pip install -U scikit-learn
from sklearn.metrics import precision_score
# define actual
act_pos = [1 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# define predictions
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
pred_neg = [1 for _ in range(30)] + [0 for _ in range(9970)]
y_pred = pred_pos + pred_neg
# calculate prediction
precision = precision_score(y_true, y_pred, average='binary')
print('Precision: %.3f' % precision)
```

## Recall Metric

**Recall** is a metric that quantifies **the number of correct positive predictions   
made out of all positive predictions that could have been made**.  

Unlike precision that only comments on the correct positive predictions out of  
all positive predictions, recall provides an indication of missed positive predictions.  

In this way, **recall** provides some **notion of the coverage of the positive class**.

### Recall for Binary Classification

In an imbalanced classification problem with two classes, **recall**  
is calculated as **the number of true positives divided by the total  
number of true positives and false negatives**.

**Recall = TruePositives / (TruePositives + FalseNegatives)**  
The result is a value between 0.0 for no recall and 1.0 for full or perfect recall.  

Let’s make this calculation concrete with some examples.  

As in the previous section, consider a dataset with 1:100 minority to majority ratio,  
with 100 minority examples and 10,000 majority class examples.  

A model makes predictions and predicts 90 of the positive class predictions correctly  
and 10 incorrectly. We can calculate the recall for this model as follows:  

Recall = TruePositives / (TruePositives + FalseNegatives)  
Recall = 90 / (90 + 10)  
Recall = 90 / 100  
Recall = 0.9  
This model has a good recall.

### Calculate Recall With Scikit-Learn

The recall score can be calculated using the [recall_score() scikit-learn function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html).  

For example, we can use this function to calculate recall for the scenarios above.

First, we can consider the case of a 1:100 imbalance with 100 and 10,000 examples  
respectively, and a model predicts 90 true positives and 10 false negatives.  

The complete example is listed below.

```{code-cell} ipython3
# calculates recall for 1:100 dataset with 90 tp and 10 fn
from sklearn.metrics import recall_score
# define actual
act_pos = [1 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# define predictions
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
pred_neg = [0 for _ in range(10000)]
y_pred = pred_pos + pred_neg
# calculate recall
recall = recall_score(y_true, y_pred, average='binary')
print('Recall: %.3f' % recall)
```

## Precision vs. Recall for Imbalanced Classification

You may decide to use precision or recall on your imbalanced classification problem.

Maximizing precision will minimize the number false positives, whereas maximizing  
the recall will minimize the number of false negatives.  

**Precision: Appropriate when minimizing false positives is the focus.  
Recall: Appropriate when minimizing false negatives is the focus.**    
Sometimes, we want excellent predictions of the positive class. We want high precision and high recall.  

This can be challenging, as often increases in recall often come at the expense of decreases in precision.

## F-Measure for Imbalanced Classification

**Classification accuracy** is widely used because it is one single measure used  
to summarize model performance.

**F-Measure** provides a way to **combine both precision and recall** into a single measure  
that captures both properties.  

Alone, neither precision or recall tells the whole story. We can have excellent precision  
with terrible recall, or alternately, terrible precision with excellent recall.  
F-measure provides a way to express both concerns with a single score.  

Once precision and recall have been calculated for a classification problem,  
the two scores can be combined into the calculation of the F-Measure.  

The traditional F measure is calculated as follows:  
**F-Measure = (2 * Precision * Recall) / (Precision + Recall)**  

This is the [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of the two fractions. This is sometimes called the  
F-Score or the F1-Score and might be the most common metric used on imbalanced classification problems.

A more general F score, $F_{\beta }$,   
that uses a positive real factor β, **where β is chosen such that recall is  
considered β times as important as precision**, is:  
$F_\beta = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{(\beta^2 \cdot \mathrm{precision}) + \mathrm{recall}}$  
The more generic $F_{\beta }$ score applies additional weights,   
valuing one of precision or recall more than the other.

The highest possible value of an F-score is 1.0, indicating perfect precision and recall,  
and the lowest possible value is 0, if either the precision or the recall is zero.

Like precision and recall, a poor F-Measure score is 0.0 and a best or perfect F-Measure score is 1.0  

For example, a perfect precision and recall score would result in a perfect F-Measure score:  

F-Measure = (2 * Precision * Recall) / (Precision + Recall)  
F-Measure = (2 * 1.0 * 1.0) / (1.0 + 1.0)  
F-Measure = (2 * 1.0) / 2.0  
F-Measure = 1.0

Let’s make this calculation concrete with a worked example.  

Consider a binary classification dataset with 1:100 minority to majority ratio,  
with 100 minority examples and 10,000 majority class examples.  

Consider a model that predicts 150 examples for the positive class, 95 are correct  
(true positives), meaning five were missed (false negatives) and 55 are incorrect (false positives).  

We can calculate the precision as follows:  

Precision = TruePositives / (TruePositives + FalsePositives)  
Precision = 95 / (95 + 55)  
Precision = 0.633  
We can calculate the recall as follows:  

Recall = TruePositives / (TruePositives + FalseNegatives)  
Recall = 95 / (95 + 5)  
Recall = 0.95   
This shows that the model has poor precision, but excellent recall.  

Finally, we can calculate the F-Measure as follows:  

F-Measure = (2 * Precision * Recall) / (Precision + Recall)  
F-Measure = (2 * 0.633 * 0.95) / (0.633 + 0.95)  
F-Measure = (2 * 0.601) / 1.583  
F-Measure = 1.202 / 1.583  
F-Measure = 0.759  
We can see that the good recall levels-out the poor precision, giving an okay or reasonable F-measure score.

### Calculate F-Measure With Scikit-Learn

The F-measure score can be calculated using the f1_score() scikit-learn function.  

For example, we use this function to calculate F-Measure for the scenario above.  

This is the case of a 1:100 imbalance with 100 and 10,000 examples respectively,   
and a model predicts 95 true positives, five false negatives, and 55 false positives.  
  
The complete example is listed below.

```{code-cell} ipython3
# calculates f1 for 1:100 dataset with 95tp, 5fn, 55fp
from sklearn.metrics import f1_score
# define actual
act_pos = [1 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# define predictions
pred_pos = [0 for _ in range(5)] + [1 for _ in range(95)]
pred_neg = [1 for _ in range(55)] + [0 for _ in range(9945)]
y_pred = pred_pos + pred_neg
# calculate score
score = f1_score(y_true, y_pred, average='binary')
print('F-Measure: %.3f' % score)
```

## Extension

sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, __average ='binary'__ , sample_weight=None, zero_division='warn')

'binary':
Only report results for the class specified by pos_label.   
This is applicable only if targets ($y_{true,pred}$) are binary.  

'micro':
Calculate metrics globally by counting the total true positives,  
false negatives and false positives.  

'macro':
Calculate metrics for each label, and find their unweighted mean.  
This does not take label imbalance into account.  

'weighted':
Calculate metrics for each label, and find their average weighted by support   
(the number of true instances for each label).  
This alters ‘macro’ to account for label imbalance;  
it can result in an F-score that is not between precision and recall.

'samples':
Calculate metrics for each instance, and find their average (only meaningful  
for multilabel classification where this differs from accuracy_score).

```{code-cell} ipython3
from sklearn.metrics import precision_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
precision_score(y_true, y_pred, average='macro')
```

```{code-cell} ipython3
precision_score(y_true, y_pred, average='micro')
```

```{code-cell} ipython3
precision_score(y_true, y_pred, average='weighted')
```

More resource: [Multi-Class Metrics](https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1)

+++

## ROC Curves and ROC AUC

+++

An ROC curve (or receiver operating characteristic curve) is a plot that summarizes the performance of a      
 binary classification model on the positive class.

The x-axis indicates the False Positive Rate and the y-axis indicates the True Positive Rate.

+++

- ROC Curve: Plot of False Positive Rate (x) vs. True Positive Rate (y).  
- TruePositiveRate = TruePositives / (TruePositives + False Negatives)  
- FalsePositiveRate = FalsePositives / (FalsePositives + TrueNegatives)

+++

Ideally, we want the fraction of correct positive class predictions to be 1 (top of the plot) and the  
fraction of incorrect negative class predictions to be 0 (left of the plot). This highlights that the  
best possible classifier that achieves perfect skill is the top-left of the plot (coordinate 0,1).

+++

We can plot a ROC curve for a model in Python using the [roc_curve() scikit-learn function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html).

The function takes both the true outcomes (0,1) from the test set and the predicted probabilities for the 1  
class. The function returns the false positive rates for each threshold, true positive rates for each  
 threshold and thresholds.

+++

The complete example is listed below.

```{code-cell} ipython3
# example of a roc curve for a predictive model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(testy, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
```

The ROC Curve for the Logistic Regression model is shown (orange with dots). A no skill classifier as a  
 diagonal line (blue with dashes).

+++

### ROC Area Under Curve (AUC) Score

+++

Although the ROC Curve is a helpful diagnostic tool, it can be challenging to compare two or more   
classifiers based on their curves.  

Instead, the area under the curve can be calculated to give a single score for a classifier model across  
 all threshold values. This is called the ROC area under curve or ROC AUC or sometimes ROCAUC.     

The score is a value between 0.0 and 1.0 for a perfect classifier.

+++

The AUC for the ROC can be calculated in scikit-learn using the [roc_auc_score() function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).  

Like the roc_curve() function, the AUC function takes both the true outcomes (0,1) from the test set and the  
 predicted probabilities for the positive class.

+++

The complete example is listed below.

```{code-cell} ipython3
# example of a roc auc for a predictive model
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# no skill model, stratified random class predictions
model = DummyClassifier(strategy='stratified')
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
pos_probs = yhat[:, 1]
# calculate roc auc
roc_auc = roc_auc_score(testy, pos_probs)
print('No Skill ROC AUC %.3f' % roc_auc)
# skilled model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
pos_probs = yhat[:, 1]
# calculate roc auc
roc_auc = roc_auc_score(testy, pos_probs)
print('Logistic ROC AUC %.3f' % roc_auc)
```

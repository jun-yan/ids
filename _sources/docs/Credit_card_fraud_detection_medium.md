---
jupytext:
  formats: ipynb,md:myst
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

## Credit Card Fraud Detection

+++

Our goal is to identify fraudulent transactions using classification models to classify and distinguish them.

Steps involved:
- Importing the required packages
- Importing the data
- Exploratory Data Analysis
- Data Split
- Building various classification models
- Evaluating the classification models using the evaluation metrics

+++

### Importing packages

+++

We will be using the following packages: Pandas to manipulate data, NumPy to manipulate arrays, scikit-learn for spliting the data into train and test, building and evaluating the classification models, and xgboost to use the xgboost classifier model algorithm.

```{code-cell} ipython3
import warnings
warnings.filterwarnings('error')
```

```{code-cell} ipython3
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from termcolor import colored as cl # text customization
import itertools # advanced tools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
```

### Importing Data

+++

The variables V1 to V28 in the data are the principal components obtained by PCA. Since the variable time does not help our modeling we are going to remove it from our data. The variable ‘Amount’ contains the total amount of money being transacted and the variable ‘Class’ contains the information whether the transaction is a fraudulent one or not.

```{code-cell} ipython3
data = pd.read_csv('../data/creditcard.csv')
data.drop('Time', axis = 1, inplace = True)

print(data.head())
```

### Exploratory Data Analysis

+++

Next we look at the number of fraudulent and non-fraudulent cases that are present in our dataset and also compute the percentage of fraudulent cases.

```{code-cell} ipython3
cases = len(data)
nonfraud_count = len(data[data.Class == 0])
fraud_count = len(data[data.Class == 1])
fraud_percentage = round(fraud_count/nonfraud_count*100, 2)

print(cl('CASE COUNT', attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('Total number of cases are {}'.format(cases), attrs = ['bold']))
print(cl('Number of Non-fraud cases are {}'.format(nonfraud_count), attrs = ['bold']))
print(cl('Number of fraud cases are {}'.format(fraud_count), attrs = ['bold']))
print(cl('Percentage of fraud cases is {}'.format(fraud_percentage), attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))
```

```{code-cell} ipython3
## Descriptive Statistics

nonfraud_cases = data[data.Class == 0]
fraud_cases = data[data.Class == 1]

print(cl('CASE AMOUNT STATISTICS', attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('NON-FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(nonfraud_cases.Amount.describe())
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(fraud_cases.Amount.describe())
print(cl('--------------------------------------------', attrs = ['bold']))
```

We can see that the values in the variable ‘Amount’ has a high variability in comparison to the other variables. In order to tackle this issue we are going to standardize this variable using the ‘StandardScaler’ method.

```{code-cell} ipython3
sc = StandardScaler()
amount = data['Amount'].values

data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

print(cl(data['Amount'].head(10), attrs = ['bold']))
```

### Data Split

```{code-cell} ipython3
X = data.drop('Class', axis = 1).values
y = data['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(cl('X_train samples : ', attrs = ['bold']), X_train[:1])
print(cl('X_test samples : ', attrs = ['bold']), X_test[0:1])
print(cl('y_train samples : ', attrs = ['bold']), y_train[0:20])
print(cl('y_test samples : ', attrs = ['bold']), y_test[0:20])
```

### Modeling

+++

#### Decision Tree

```{code-cell} ipython3
tree_model = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
tree_model.fit(X_train, y_train)
tree_yhat = tree_model.predict(X_test)
```

The ‘DecisionTreeClassifier’ algorithm is used to build the model. And the ‘max_depth’ specification in the function refers to the number of splits in the tree and the ‘criterion’ specification determines when to stop splitting the tree.

+++

#### K-Nearest Neighbors

```{code-cell} ipython3
n = 5

knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train, y_train)
knn_yhat = knn.predict(X_test)
```

The ‘KNeighborsClassifier’ algorithm is used to build the model. The value of the ‘n_neighbors’ is chosen randomly but it can also be chosen optimistically by iterating through a range of values.

+++

#### Logistic Regression

```{code-cell} ipython3
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)
```

#### Support Vector Machine

```{code-cell} ipython3
svm = SVC()
svm.fit(X_train, y_train)
svm_yhat = svm.predict(X_test)
```

#### Random Forest Tree

```{code-cell} ipython3
rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)
```

The ‘RandomForestClassifier’ algorithm is used to build the model and the ‘max_depth’ specification in the function refers to the number of splits in the tree. The main difference between the decision tree and the random forest is that, the decision tree uses the entire dataset to construct a single model whereas, the random forest uses randomly selected features to construct multiple models.

+++

#### XGBoost

```{code-cell} ipython3
xgb = XGBClassifier(max_depth = 4, use_label_encoder=False,eval_metric='error')
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)
```

The ‘XGBClassifier’ algorithm provided is used to build the model.

+++

### Model Evaluation

+++

#### Accuracy Score

+++

The accuracy score is calculated by dividing the number of correct predictions made by the model by the total number of predictions made by the model. It can generally be expressed as:

Accuracy score = No.of correct predictions / Total no.of predictions

```{code-cell} ipython3
print(cl('ACCURACY SCORE', attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test, tree_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the KNN model is {}'.format(accuracy_score(y_test, knn_yhat)), attrs = ['bold'], color = 'green'))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)), attrs = ['bold'], color = 'red'))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the SVM model is {}'.format(accuracy_score(y_test, svm_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the Random Forest Tree model is {}'.format(accuracy_score(y_test, rf_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
```

According to the accuracy score evaluation metric, the KNN model seems to be the most accurate model and the Logistic regression model seems to be the least accurate model.

+++

#### F1 Score

+++

The F1 score or F-score can be defined as the harmonic mean of the model’s precision and recall. It is calculated by dividing the product of the model’s precision and recall by the value obtained on adding the model’s precision and recall and finally multiplying the result with 2 (We are giving equal weights to precision and recall). It can be expressed as:

F1 score = 2( (precision * recall) / (precision + recall) )

Precision = True Positive / (True Positive + False Positive)

Recall = True Positive / (True Positive + False Negative)

```{code-cell} ipython3
print(cl('F1 SCORE', attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, tree_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the KNN model is {}'.format(f1_score(y_test, knn_yhat)), attrs = ['bold'], color = 'green'))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)), attrs = ['bold'], color = 'red'))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the SVM model is {}'.format(f1_score(y_test, svm_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the Random Forest Tree model is {}'.format(f1_score(y_test, rf_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
```

On basis of the F1 score evaluation metric, the KNN model is the best performing model and the Logistic regression model seems to be the least accurate model.

+++

#### Confusion Matrix

+++

Typically, a confusion matrix is a visualization of a classification model that shows how well the model has predicted the outcomes when compared to the original ones. Usually, the predicted outcomes are stored in a variable that is then converted into a correlation table. Using the correlation table, the confusion matrix is plotted in the form of a heatmap.

```{code-cell} ipython3
# defining the plot function

def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix for the models

tree_matrix = confusion_matrix(y_test, tree_yhat, labels = [0, 1]) # Decision Tree
knn_matrix = confusion_matrix(y_test, knn_yhat, labels = [0, 1]) # K-Nearest Neighbors
lr_matrix = confusion_matrix(y_test, lr_yhat, labels = [0, 1]) # Logistic Regression
svm_matrix = confusion_matrix(y_test, svm_yhat, labels = [0, 1]) # Support Vector Machine
rf_matrix = confusion_matrix(y_test, rf_yhat, labels = [0, 1]) # Random Forest Tree
xgb_matrix = confusion_matrix(y_test, xgb_yhat, labels = [0, 1]) # XGBoost

# Plot the confusion matrix

plt.rcParams['figure.figsize'] = (6, 6)
```

#### Decision tree

```{code-cell} ipython3
tree_cm_plot = plot_confusion_matrix(tree_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'Decision Tree')
plt.show()
```

The first row represents transactions whose actual fraud value in the test dataset is 0. As we can see, the total number of non-fraud transactions is 56861. And out of these 56861 non-fraud transactions, the classifier correctly predicted 56849 of them as 0 and 12 of them as 1.

Let’s look at the second row. It looks like there were 101 fraudulent transactions. The classifier correctly predicted 77 of them as 1, and 24 of them wrongly as 0.

+++

#### K-Nearest Neighbors

```{code-cell} ipython3
knn_cm_plot = plot_confusion_matrix(knn_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'KNN')
plt.show()
```

Out of these 56861 non-fraud transactions, the classifier correctly predicted 56854 of them as 0 and 7 of them as 1. 

Out of 101 fraudulent cases the classifier correctly predicted 81 of them as 1, and 20 of them wrongly as 0.

+++

#### Logistic regression

```{code-cell} ipython3
lr_cm_plot = plot_confusion_matrix(lr_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'Logistic Regression')
plt.show()
```

Out of these 56861 non-fraud transactions, the classifier correctly predicted 56852 of them as 0 and 9 of them as 1. 

Out of 101 fraudulent cases the classifier correctly predicted 64 of them as 1, and 37 of them wrongly as 0.

+++

#### Support Vector Machine

```{code-cell} ipython3
svm_cm_plot = plot_confusion_matrix(svm_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'SVM')
plt.show()
```

Out of these 56861 non-fraud transactions, the classifier correctly predicted 56855 of them as 0 and 6 of them as 1. 

Out of 101 fraudulent cases the classifier correctly predicted 68 of them as 1, and 33 of them wrongly as 0.

+++

#### Random forest tree

```{code-cell} ipython3
rf_cm_plot = plot_confusion_matrix(rf_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'Random Forest Tree')
plt.show()
```

Out of these 56861 non-fraud transactions, the classifier correctly predicted 56854 of them as 0 and 7 of them as 1. 

Out of 101 fraudulent cases the classifier correctly predicted 69 of them as 1, and 32 of them wrongly as 0.

+++

#### XGBoost

```{code-cell} ipython3
xgb_cm_plot = plot_confusion_matrix(xgb_matrix, 
                                classes = ['Non-Default(0)','Default(1)'], 
                                normalize = False, title = 'XGBoost')
plt.show()
```

Out of these 56861 non-fraud transactions, the classifier correctly predicted 56854 of them as 0 and 7 of them as 1. 

Out of 101 fraudulent cases the classifier correctly predicted 79 of them as 1, and 22 of them wrongly as 0.

+++

Comparing the confusion matrix of all the models, it can be seen that the K-Nearest Neighbors model has performed really well in classifying the fraud transactions from the non-fraud transactions followed. So we can conclude that the most appropriate model which can be used for our case is the K-Nearest Neighbors model and the least appropraite model is the Logistic regression model.

+++

Reference:
- Credit Card Fraud Detection With Machine Learning in Python by Nikhil Adityan
  https://medium.com/codex/credit-card-fraud-detection-with-machine-learning-in-python-ac7281991d87

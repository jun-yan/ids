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

## XGBoost

+++

### What is XGBoost?

+++

XGBoost (Extreme Gradient Boosting) belongs to a family of boosting algorithms and uses the gradient boosting (GBM) framework at its core. It is an optimized distributed gradient boosting library.

+++

### What is Boosting?

+++

Boosting is a sequential technique which works on the principle of an ensemble. It combines a set of weak learners and delivers improved prediction accuracy. At any instant t, the model outcomes are weighed based on the outcomes of previous instant t-1. The outcomes predicted correctly are given a lower weight and the ones miss-classified are weighted higher. Note that a weak learner is one which is slightly better than random guessing. The basic idea behind boosting algorithms is building a weak model, making conclusions about the various feature importance and parameters, and then using those conclusions to build a new, stronger model and capitalize on the misclassification error of the previous model and try to reduce it. Now, let's come to XGBoost. To begin with, you should know about the default base learners of XGBoost: tree ensembles. The tree ensemble model is a set of classification and regression trees (CART). Trees are grown one after another ,and attempts to reduce the misclassification rate are made in subsequent iterations. Each tree gives a different prediction score depending on the data it sees and the scores of each individual tree are summed up to get the final score.

+++

### What makes XGBoost so popular?

+++

- Speed and performance : It is comparatively faster than other ensemble classifiers.

- Core algorithm is parallelizable : It can use the power of multi-core computers. It is also parallelizable onto GPU’s and across networks of computers making it feasible to train on very large datasets as well.

- Consistently outperforms other algorithm methods : It has shown better performance on a variety of machine learning benchmark datasets.

- Wide variety of tuning parameters : XGBoost internally has parameters for cross-validation, regularization, user-defined objective functions, missing values, tree parameters, scikit-learn compatible API etc.

+++

We will be using the credit card fraud data as an example to see how XGBoost works.

+++

### Importing Packages

+++

We will be using the following packages: Pandas to manipulate data, NumPy to manipulate arrays, scikit-learn for spliting the data into train and test, building and evaluating the classification models, and xgboost to use the xgboost classifier model algorithm.

```{code-cell} ipython3
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import itertools # advanced tools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
```

### Importing Dataset

+++

The variables V1 to V28 in the data are the principal components obtained by PCA. Since the variable time does not help our modeling we are going to remove it from our data. The variable ‘Amount’ contains the total amount of money being transacted and the variable ‘Class’ contains the information whether the transaction is a fraudulent one or not.

```{code-cell} ipython3
data = pd.read_csv('../data/creditcard.csv')
data.drop('Time', axis = 1, inplace = True)

print(data.head())
```

### Exploratory data analysis

+++

Next we look at the number of fraudulent and non-fraudulent cases that are present in our dataset and also compute the percentage of fraudulent cases.

```{code-cell} ipython3
cases = len(data)
nonfraud_count = len(data[data.Class == 0])
fraud_count = len(data[data.Class == 1])
fraud_percentage = round(fraud_count/nonfraud_count*100, 2)

print('CASE COUNT')
print('--------------------------------------------')
print('Total number of cases are {}'.format(cases))
print('Number of Non-fraud cases are {}'.format(nonfraud_count))
print('Number of fraud cases are {}'.format(fraud_count))
print('Percentage of fraud cases is {}'.format(fraud_percentage))
print('--------------------------------------------')
```

```{code-cell} ipython3
data.describe()
```

We can see that the values in the variable ‘Amount’ has a high variability in comparison to the other variables. In order to tackle this issue we are going to standardize this variable using the ‘StandardScaler’ method.

```{code-cell} ipython3
sc = StandardScaler()
amount = data['Amount'].values

data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

print(data['Amount'].head(10))
```

If your dataset has categorical features you may want to consider applying some encoding (like one-hot encoding) to such features before training the model using XGBoost. Also, XGBoost is capable of handling missing values internally so its not neccesary to address this issue before you fit the model. 

+++

### Data Manipulation 

```{code-cell} ipython3
X = data.drop('Class', axis = 1).values
y = data['Class'].values
```

Now we will convert the dataset into an optimized data structure called Dmatrix that XGBoost supports and gives it acclaimed performance and efficiency gains.

```{code-cell} ipython3
data1 = xgb.DMatrix(data=X,label=y)
```

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print('X_train samples : ', X_train[:1])
print('X_test samples : ', X_test[0:1])
print('y_train samples : ', y_train[0:20])
print('y_test samples : ', y_test[0:20])
```

### Model building

+++

#### XGBoost hyperparamaters

+++

The most common parameters in xgboost are:

- learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]

- max_depth: determines how deeply each tree is allowed to grow during any boosting round.

- subsample: percentage of samples used per tree. Low value can lead to underfitting.

- colsample_bytree: percentage of features used per tree. High value can lead to overfitting.

- n_estimators: number of trees you want to build.

- objective: determines the loss function to be used like reg:linear for regression problems, reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability.

XGBoost also supports regularization parameters to penalize models as they become more complex and reduce them to simple models.

- gamma: controls whether a given node will split based on the expected reduction in loss after the split. A higher value leads to fewer splits. Supported only for tree-based learners.

- alpha: L1 regularization on leaf weights. A large value leads to more regularization.

- lambda: L2 regularization on leaf weights and is smoother than L1 regularization.

It's also worth mentioning that though you are using trees as your base learners, you can also use XGBoost's relatively less popular linear base learners and one other tree learner known as dart. All you have to do is set the booster parameter to either gbtree (default),gblinear or dart.

+++

#### Model fitting

```{code-cell} ipython3
xg_reg = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10,use_label_encoder=False)
```

#### Model prediction

```{code-cell} ipython3
xg_reg.fit(X_train,y_train,eval_metric='error')

preds = xg_reg.predict(X_test)
```

#### Model validatoion

```{code-cell} ipython3
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
```

```{code-cell} ipython3
print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, preds)))
```

```{code-cell} ipython3
print('F1 score of the XGBoost model is {}'.format(f1_score(y_test, preds)))
```

Reference:
- Credit Card Fraud Detection With Machine Learning in Python by Nikhil Adityan
  https://medium.com/codex/credit-card-fraud-detection-with-machine-learning-in-python-ac7281991d87

- Using XGBoost in Python
  https://www.datacamp.com/community/tutorials/xgboost-in-python#what

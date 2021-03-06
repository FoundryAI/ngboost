![Python package](https://github.com/stanfordmlgroup/ngboost/workflows/Python%20package/badge.svg)

# NGBoost: Natural Gradient Boosting for Probabilistic Prediction

ngboost is a Python library that implements Natural Gradient Boosting, as described in ["NGBoost: Natural Gradient Boosting for Probabilistic Prediction"](https://stanfordmlgroup.github.io/projects/ngboost/). It is built on top of [Scikit-Learn](https://scikit-learn.org/stable/), and is designed to be scalable and modular with respect to choice of proper scoring rule, distribution, and base learners.

Installation:

```
pip install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git
```

Probabilistic regression example on the Boston housing dataset:


```python
from ngboost import NGBRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, Y = load_boston(True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test).mean()
print('Test NLL', test_NLL)
```

More information about available distributions, scoring rules, learners, how to tune NGBoost models, and model interpretation is available in our [vignette](https://github.com/stanfordmlgroup/ngboost/blob/master/examples/vignette.ipynb). Please see our [developer guide](https://github.com/stanfordmlgroup/ngboost/blob/master/examples/dev_vignette.ipynb) for more information on how to add new distributions or scores to NGBoost.

A [slide deck](https://drive.google.com/file/d/183BWFAdFms81MKy6hSku8qI97OwS_JH_/view?usp=sharing) with an accessible presentation of how NGBoost works is also available.

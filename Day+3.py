
# coding: utf-8

# In[5]:

import numpy as np
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

from plot_util import get_colormap

get_ipython().magic('matplotlib inline')


# In[3]:

iris = pd.read_csv('data/iris.data', header=None)


# In[4]:

iris[:5]


# In[12]:

X = iris[:100].loc[:, 0:3].values


# In[13]:

iris[:100][4].value_counts()


# In[26]:

y = iris[:100][4]


# In[27]:

y = np.where(y == 'Iris-setosa', 1, -1)


# In[28]:

np.unique(y)


# In[17]:

from sklearn.model_selection import train_test_split


# In[18]:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)


# In[19]:

from sklearn.linear_model import Perceptron


# In[20]:

model = Perceptron(n_iter=10, eta0=0.1)


# In[21]:

model.fit(X_train, y_train)


# In[24]:

model.coef_, model.intercept_


# In[29]:

y_pred = model.predict(X_test)


# In[34]:

score = (y_pred == y_test).sum() / len(y_test)


# In[35]:

score


# #### 연습
# 
# 아래 구성된 data 대해 퍼셉트론 모델을 훈련하고 평가합니다.

# In[38]:

data = iris[50:]


# In[39]:

data[4].value_counts()


# In[42]:

X = data.loc[:, 0:3].values


# In[45]:

y = np.where(data[4] == 'Iris-versicolor', 1, -1)


# In[46]:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)


# In[47]:

model = Perceptron(n_iter=10, eta0=0.1)


# In[48]:

model.fit(X_train, y_train)


# In[49]:

score = model.score(X_test, y_test)


# In[50]:

score


# In[54]:

c = get_colormap(y, colors='rb')
data.plot(kind='scatter', x=0, y=1, c=c)


# In[62]:

results = []
for n in range(10, 110, 10):
    model = Perceptron(n_iter=n, eta0=0.1)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    results.append((n, score))


# In[63]:

report = DataFrame(results)
report.columns = ['n_iter', 'score']
report = report.set_index('n_iter')
report


# In[64]:

report.plot(style='o--')


# In[65]:

from sklearn.linear_model import LogisticRegression


# In[66]:

model = LogisticRegression(C=1.0)


# In[67]:

model.fit(X_train, y_train)


# In[68]:

score = model.score(X_test, y_test)


# In[69]:

score


# In[70]:

model.predict_proba(X_test)


# In[73]:

X = iris.loc[:, 0:3].values


# In[76]:

y = iris[4].values


# In[77]:

np.unique(y)


# In[78]:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)


# In[79]:

model = LogisticRegression(C=1.0)


# In[80]:

model.fit(X_train, y_train)


# In[81]:

score = model.score(X_test, y_test)


# In[82]:

score


# In[83]:

model.coef_


# In[84]:

c = get_colormap(y, colors='rgb')
iris.plot(kind='scatter', x=0, y=1, c=c)


# In[85]:

model.score(X_train, y_train)


# In[87]:

wine = pd.read_csv('data/wine.data')


# In[88]:

wine.shape


# In[91]:

wine['Class label'].value_counts()


# In[92]:

y = wine['Class label'].values


# In[97]:

X = wine.loc[:, 'Alcohol':].values


# In[98]:

X.shape


# #### 연습
# 
# 구성한 X, y에 대해 로지스틱 회귀(LR; Logistic Regression) 모델을 훈련하고, 평가합니다.

# In[ ]:




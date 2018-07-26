
# coding: utf-8

# # 2. Problem Statement¶
# Build the linear regression model using scikit learn in boston data to predict 'Price' based on other dependent variable. Here is the code to load the data 
# import numpy as np
# import pandas as pd
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# import sklearn import scipy.stats from sklearn.linear_model
# import LinearRegression from sklearn.model_selection
# import train_test_split from sklearn.datasets
# import load_boston boston = load_boston() bos = pd.DataFrame(boston.data)

# In[117]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import load_boston
boston = load_boston()


# ## Basic details about data

# In[118]:


boston.keys()


# In[119]:


print(boston.DESCR)


# In[120]:


bos= pd.DataFrame(boston.data)


# In[121]:


bos.head()


# In[122]:


bos.columns = boston.feature_names


# boston.target contains housing price

# In[123]:


boston.target[:5]


# In[124]:


bos['PRICE'] = boston.target


# In[125]:


bos.head()


# In[126]:


bos.info()


# In[127]:


bos.describe()


# ## Visual Analysis

# In[128]:


bos.hist(bins=50, figsize=(20,15))
plt.show()


# In[129]:


sns.pairplot(bos.iloc[:,:-1])


# In[130]:


bos.corr()


# Note: RM has high correlation with PRICE. LStata and PTRATIO has negative correlation with PRICE

# In[131]:


plt.figure(figsize=(10,8))
sns.heatmap(bos.corr())


# Histogram
# 
# Histograms are a useful way to visually summarize the statistical properties of numeric variables. They can give you an idea of the mean and the spread of the variables as well as outliers

# In[132]:


plt.hist(bos.CRIM)
plt.title("CRIME")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequency")
plt.show()


# In[133]:


plt.hist(bos.RM)
plt.title("RM")
plt.xlabel("Avg Number of Rooms")
plt.ylabel("Frequency")
plt.show()


# ## Statistical Analysis

# ## Linear regression with Boston housing data

# 
# Y = boston housing prices (also called "target" data in python)
# and
# X = all the other features (or independent variables)
# which we will use to fit a linear regression model and predict Boston housing prices. We will use the least squares method as the way to estimate the coefficients.
# We'll use two ways of fitting a linear regression.

# ## Train and Test Split

# In[134]:


# Import regression modules
# ols - stands for Ordinary least squares, we'll use this
import statsmodels.api as sm
from statsmodels.formula.api import ols


# ### We will be using 80:20 split for train and test datasets.

# In[135]:


X = bos.drop('PRICE',axis=1)
Y = bos['PRICE']


# In[136]:


train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state= 10)


# In[137]:


print(train_x.shape)
print(train_y.shape)


# In[138]:


print(test_x.shape)
print(test_y.shape)


# ## Fitting Regression Model

# In[139]:


import statsmodels.api as sm


# In[140]:


train_x = sm.add_constant(train_x)
test_x = sm.add_constant(test_x)
model = sm.OLS(train_y, train_x).fit()


# In[141]:


model.summary()


# In[142]:


lm = LinearRegression()


# In[143]:


model = lm.fit(train_x.values, train_y.values)


# In[144]:


model.coef_


# In[145]:


model.intercept_


# In[146]:


model.score(train_x, train_y)


# In[147]:


model.score(test_x, test_y)


# ## Evaluate Model Performance

# In[148]:


from sklearn.metrics import mean_squared_error


# In[149]:


import math


# In[150]:


train_pred_y = model.predict(train_x)


# In[151]:


test_pred_y = model.predict(test_x)


# In[152]:


print('RMSE Train', math.sqrt(mean_squared_error(train_pred_y, train_y)))
print('RMSE Test' ,math.sqrt(mean_squared_error(test_pred_y, test_y)))


# In[153]:


train_pred_y.shape


# In[154]:


train_y.shape


# In[155]:


bos.PRICE.mean()


# In[156]:


plt.scatter(train_pred_y, train_y)


# In[157]:


plt.scatter(test_pred_y, test_y)


# In[158]:


plt.hist(train_pred_y - train_y, bins=20)


# In[159]:


plt.hist(test_pred_y - test_y, bins=20)


# In[160]:


plt.scatter(train_y, train_y - train_pred_y)


# In[161]:


plt.scatter(test_y, test_y - test_pred_y)


# Residual Plot

# In[162]:


plt.xlabel('Price')
plt.ylabel('Predicted Price Residuals')
plt.title('Training - Residuals Plot')
plt.scatter(train_y,train_y - train_pred_y, c= "green")
plt.hlines(y=0,xmin=0,xmax=50)
plt.show()


# In[163]:


plt.xlabel('Price')
plt.ylabel('Predicted Price Residuals')
plt.title('Test - Residuals Plot')
plt.scatter(test_y, test_y - test_pred_y, c= "blue")
plt.hlines(y=0,xmin=0,xmax=50)
plt.show()


# ## Cross Validation Score

# In[164]:


from sklearn.model_selection import cross_val_score
from sklearn import metrics

# Perform 6-fold cross validation
scores = cross_val_score(model, X, Y, cv=10, scoring='neg_mean_squared_error')

print("Cross-validated scores:", scores)


# In[165]:


scores.std()


# In[166]:


scores.mean()


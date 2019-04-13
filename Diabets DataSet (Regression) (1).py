#!/usr/bin/env python
# coding: utf-8

# In[17]:


print(__doc__)

#kutuphaneleri importlanmasi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# diabetes datasetin yüklenmesi
diabetes = datasets.load_diabetes()


# Sadece 1 feature kullanıyorum 
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Datayı training/testing sets olmak üzere ikiye ayırıyorum
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-10:]

# Outputu training/testing sets olarak ayırdım
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-10:]

# Linear Regression modeli oluşturdum
regr = linear_model.LinearRegression()

# Modelimi train ettim
regr.fit(diabetes_X_train, diabetes_y_train)

# Test setimi kullanarak predit ediyorum
diabetes_y_pred = regr.predict(diabetes_X_test)

#Coefficient değerlerim
print('Coefficients: \n', regr.coef_)
#Mean squred error saptandı
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Predicition doğruluk oranım
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Değerler grafikle gösterildi
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[10]:


print(diabetes.feature_names)


# In[11]:


print(diabetes.data.shape)


# In[18]:


dib = pd.DataFrame(diabetes.data)
print(dib.describe())


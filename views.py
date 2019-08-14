import calendar
import pickle
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.svm import SVR
from statsmodels import robust

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# Fetching Dataset
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)

# Data Exloration and summary
#bike_data.head()
#bike_data.describe()
#bike_data.dtypes
#bike_data.shape

features = bike_data.columns[:-3]
target = bike_data.columns[-1]

print ("Feature column(s):\n{}\n".format(features))
print ("Target column:\n{}".format(target))

# Number of Bikes rented per month visualisation
plt.style.use('ggplot')

bike_data.boxplot(column='cnt', by=['yr','mnth'])

plt.title('Number of bikes rented per month')
plt.xlabel('')
plt.xticks((np.arange(0,len(bike_data)/30,len(bike_data)/731)), calendar.month_name[1:13]*2, rotation=45)
plt.ylabel('Number of bikes')
plt.show()

# Data Pre-processing
X = bike_data[features.drop(['dteday'],['instant'])] # feature values 
y = bike_data[target]  # corresponding targets

# test size is set to 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y)

svr = SVR(gamma = 'auto')
svr.fit(X_train, y_train)

svr_pred = svr.predict(X_test)

score_svr = r2_score(y_test, svr_pred)
rmse_svr = sqrt(mean_squared_error(y_test, svr_pred))

print("Score SVR: %f" % score_svr)
print("RMSE SVR: %f" % rmse_svr)


# Tuning SVR with GridSearch
tuned_parameters = [{'C': [1000, 3000, 10000], 
                     'kernel': ['linear', 'rbf']}
                   ]

#svr_tuned = GridSearchCV(SVR (C=1), param_grid = tuned_parameters, scoring = 'mean_squared_error') #default 3-fold cross-validation, score method of the estimator
svr_tuned_GS = GridSearchCV(SVR (C=1), param_grid = tuned_parameters, scoring = 'r2', n_jobs=-1, cv = 3) #default 3-fold cross-validation, score method of the estimator

svr_tuned_GS.fit(X_train, y_train)

print (svr_tuned_GS)
print ('\n' "Best parameter from grid search: " + str(svr_tuned_GS.best_params_) +'\n')


# Validation - SVR tuned 
svr_tuned_pred_GS = svr_tuned_GS.predict(X_test)

score_svr_tuned_GS = r2_score(y_test, svr_tuned_pred_GS)
rmse_svr_tuned_GS = sqrt(mean_squared_error(y_test, svr_tuned_pred_GS))

print("SVR Results\n")

print("Score SVR: %f" % score_svr)
print("Score SVR tuned GS: %f" % score_svr_tuned_GS)

print("\nRMSE SVR: %f" % rmse_svr)
print("RMSE SVR tuned GS: %f" % rmse_svr_tuned_GS)

# SVR tuned with RandomizesSearch
# may take a while!

# Parameters
param_dist = {  'C': sp_uniform (1000, 10000), 
                'kernel': ['linear']
             }

n_iter_search = 1

# MSE optimized
#SVR_tuned_RS = RandomizedSearchCV(SVR (C=1), param_distributions = param_dist, scoring = 'mean_squared_error', n_iter=n_iter_search)

# R^2 optimized
SVR_tuned_RS = RandomizedSearchCV(SVR (C=1), param_distributions = param_dist, scoring = 'r2', n_iter=n_iter_search)

# Fit
SVR_tuned_RS.fit(X_train, y_train)

# Best score and corresponding parameters.
print('best CV score from grid search: {0:f}'.format(SVR_tuned_RS.best_score_))
print('corresponding parameters: {}'.format(SVR_tuned_RS.best_params_))

# Predict and score
predict = SVR_tuned_RS.predict(X_test)

score_svr_tuned_RS = r2_score(y_test, predict)
rmse_svr_tuned_RS = sqrt(mean_squared_error(y_test, predict))

print('SVR Results\n')

print("Score SVR: %f" % score_svr)
print("Score SVR tuned GS: %f" % score_svr_tuned_GS)
print("Score SVR tuned RS: %f" % score_svr_tuned_RS)

print("\nRMSE SVR: %f" % rmse_svr)
print("RMSE SVR tuned GS: %f" % rmse_svr_tuned_GS)
print("RMSE SVR tuned RS: %f" % rmse_svr_tuned_RS)

print('Results\n')

print("Score SVR: %f" % score_svr)
print("Score SVR tuned GS: %f" % score_svr_tuned_GS)
print("Score SVR tuned RS: %f" % score_svr_tuned_RS)

print('\n')
print("RMSE SVR: %f" % rmse_svr)
print("RMSE SVR tuned GS: %f" % rmse_svr_tuned_GS)
print("RMSE SVR tuned RS: %f" % rmse_svr_tuned_RS)

#SVR with GridSearch - for casual users

# Extracting
feature_cols_cas = bike_data.columns[:-3]  # all columns but last are features
target_col_cas = bike_data.columns[-3]  # last column is the target
print ("Feature columns:\n{}\n".format(feature_cols_cas))
print ("Target column:\n{}\n".format(target_col_cas))

# Pre-processing
X_cas = bike_data[feature_cols_cas.drop(['dteday'],['casual'])]  # feature values 
y_cas = bike_data[target_col_cas]  # corresponding targets

# Split
X_train_cas, X_test_cas, y_train_cas, y_test_cas = train_test_split(X_cas, y_cas)# test size is set to 0.25


# Tuning SVR
param_grid = [
             {'C': [1, 3, 10, 30, 100, 300, 1000, 3000],
              'kernel': ['linear', 'rbf']}
             ]

# MSR optimized
#svr_tuned_cas = GridSearchCV(SVR (C=1), param_grid = param_grid, scoring = 'mean_squared_error')

# R^2 optimized
svr_tuned_cas_GS = GridSearchCV(SVR (C=1), param_grid = param_grid, scoring = 'r2', n_jobs=-1)

# Fitting
svr_tuned_cas_GS.fit(X_train_cas, y_train_cas)

print (svr_tuned_cas_GS)
print ('\n' "Best parameter from grid search: {}".format(svr_tuned_cas_GS.best_params_))

# SVR with RandomizesSearch - for casual users
# may take a while!

# Parameters
param_dist = {  'C': sp_uniform (300, 3000), 
                'kernel': ['linear']
             }

n_iter_search = 1

svr_tuned_cas_RS = RandomizedSearchCV(SVR (C=1), param_distributions = param_dist, scoring = 'r2', n_iter=n_iter_search)

# Fit
svr_tuned_cas_RS.fit(X_train_cas, y_train_cas)

# Best score and corresponding parameters.
print('best CV score from random search: {0:f}'.format(svr_tuned_cas_RS.best_score_))
print('corresponding parameters: {}'.format(svr_tuned_cas_RS.best_params_))

# Predict and score
predict = svr_tuned_cas_RS.predict(X_test)

score_SVR_tuned_RS = r2_score(y_test, predict)
rmse_SVR_tuned_RS = sqrt(mean_squared_error(y_test, predict))

#SVR with GridSearch - for registered users

# Extracting
feature_cols_reg = bike_data.columns[:-3]  # all columns but last are features
target_col_reg = bike_data.columns[-2]  # last column is the target
print ("Feature column(s):\n{}\n".format(feature_cols_reg))
print ("Target column:\n{}\n".format(target_col_reg))

# Pre-processing
X_reg = bike_data[feature_cols_reg.drop(['dteday'],['casual'])]  # feature values 
y_reg = bike_data[target_col_reg]  # corresponding targets

# Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg)# test size is set to 0.25

# Tuning SVR
param_grid = [
             {'C': [1000, 3000, 10000],
              'kernel': ['linear', 'rbf']}
             ]

#svr_tuned_reg = GridSearchCV(SVR (C=1), param_grid = param_grid, scoring = 'mean_squared_error')
svr_tuned_reg_GS = GridSearchCV(SVR (C=1), param_grid = param_grid, scoring = 'r2', n_jobs=-1)


# Fitting 
svr_tuned_reg_GS.fit(X_train_reg, y_train_reg)

print (svr_tuned_reg_GS)
print ('\n' "Best parameter from grid search:{}".format(svr_tuned_reg_GS.best_params_))

#SVR with RandomizesSearch - for registered users
# may take a while!

# Parameters
param_dist = {  'C': sp_uniform (300, 3000), 
                'kernel': ['linear']
             }

n_iter_search = 1

svr_tuned_reg_RS = RandomizedSearchCV(SVR (C=1), param_distributions = param_dist, scoring = 'r2', n_iter=n_iter_search)

# Fit
svr_tuned_reg_RS.fit(X_train_reg, y_train_reg)

# Best score and corresponding parameters.
print('best CV score from random search: {0:f}'.format(svr_tuned_reg_RS.best_score_))
print('corresponding parameters: {}'.format(svr_tuned_reg_RS.best_params_))

# Predict and score
predict = svr_tuned_reg_RS.predict(X_test)

score_SVR_tuned_reg_RS = r2_score(y_test, predict)
rmse_SVR_tuned_reg_RS = sqrt(mean_squared_error(y_test, predict))

# Prediction

predict_sum_test = svr_tuned_cas_RS.predict(X_test) + svr_tuned_reg_RS.predict(X_test)

score = r2_score(y_test, predict_sum_test)
rmse = sqrt(mean_squared_error(y_test, predict_sum_test))

print ('Score cas: {0:f}'.format(r2_score(y_test_cas,svr_tuned_cas_RS.predict(X_test_cas))))
print ('Score reg: {0:f}'.format(r2_score(y_test_reg,svr_tuned_reg_RS.predict(X_test_reg))))
print ('Score sum: {0:f}'.format(score))
print ('\n')

print ('RMSE cas: {0:f}'.format(sqrt(mean_squared_error(y_test_cas,svr_tuned_cas_RS.predict(X_test_cas)))))
print ('RMSE reg: {0:f}'.format(sqrt(mean_squared_error(y_test_reg,svr_tuned_reg_RS.predict(X_test_reg)))))
print ('RMSE sum: {0:f}'.format(rmse))

# Results
print("Results as RMSE")
print('\n')
print("SVR: %f" % rmse_svr)
print("SVR tuned GS: %f" % rmse_svr_tuned_GS)
print("SVR tuned RS: %f" % rmse_svr_tuned_RS)
print('\n')

# Visualization

predict_sum_all = svr_tuned_cas_RS.predict(X_test) + svr_tuned_reg_RS.predict(X_test)
predictions = pd.Series(predict_sum_all, index = y_test.index.values)

plt.style.use('ggplot')
plt.figure(1)

plt.plot(y_test,'go', label='truth')
plt.plot(predictions,'bx', label='prediction')

plt.title('Number of bikes rented per day')
plt.xlabel('Days')
plt.xticks((np.arange(0,len(bike_data),len(bike_data)/24)), calendar.month_name[1:13]*2, rotation=45)

plt.ylabel('Number of bikes')

plt.legend(loc='best')

plt.show()


filename = 'models_new_cas.pk'
with open('../models/'+filename, 'wb') as file:
    pickle.dump(svr_tuned_cas_RS, file)
filename = 'models_new_reg.pk'
with open('../models/'+filename, 'wb') as file:
    pickle.dump(svr_tuned_reg_RS, file)

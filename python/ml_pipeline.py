"""
General pipeline approach to training models with a grid search and multiple estimators.

R environment setup:
    library(tidyverse)
    library(reticulate)
    use_condaenv('<env_name>')
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

# Get data, prep:
df = <somedata>

X = df[['predictor_var1', 'predictor_var2', 
        'categorical_var1', 'categorical_var2']]
y = np.ravel(df[['response_var']]) # Pipeline complains if y is a column.

# Set up grid search:
# !!NOTE!! For Pipeline, prefix parameter names with the name of the 
# estimator. E.g., "RF" + "__" + "max_depth"
cv = 3
random_seed = 42
n_estimators = [50, 100, 200, 300, 500]
min_samples_split = [2, 3, 5, 10, 20]
param_grid = []
param_grid.append(
    # For baseline comparison
    ('DUM', DummyRegressor(),
    dict()))
param_grid.append(
    # For baseline comparison
    ('LREG', LinearRegression(),
    dict()))
param_grid.append(
    # For maximum overfitting
    ('REGTREE', DecisionTreeRegressor(),
    dict()))
param_grid.append(
    ('GBM', GradientBoostingRegressor(),
    dict(
        GBM__n_estimators = n_estimators,
        GBM__min_samples_split = min_samples_split,
        # GBM__learning_rate = [0.05, 0.1]
        )))

# Set up pipeline:
num_features = ['predictor_var1', 'predictor_var2']
num_transformer = Pipeline(steps = [
    ('standard_scaler', StandardScaler())])
cat_features = ['categorical_var1', 'categorical_var2']
cat_transformer = Pipeline(steps = [
    ('one_hot_encoder', OneHotEncoder())])
pre_processor = ColumnTransformer(
    transformers = [
        ('scale_cols', num_transformer, num_features),
        ('one_hot_cols', cat_transformer, cat_features)])

# Grid search
estimator_search = {} # Store competing estimator grid results.
for grid in param_grid:
    pipeline = Pipeline(
            steps = [
                ('preprocessor', pre_processor),
                (grid[0], grid[1])],
            memory = '../ml_cache/',
            verbose = True)
    search = GridSearchCV(
        estimator = pipeline,
        scoring = 'neg_mean_absolute_error',
        param_grid = grid[2], 
        cv = cv, 
        return_train_score = True, 
        n_jobs=-1)
    search.fit(X, y)
    estimator_search.update({grid[0]: search})


# Dump all models, grids
with open('../ml_output/estimator_search.pickle', 'wb') as f:
    pickle.dump(estimator_search, f)

# Diagnostics: 

# with open('../ml_output/estimator_search.pickle', 'rb') as f:
#     estimator_search = pickle.load(f)

# Plot Actuals vs Predicteds
preds_dum = estimator_search['DUM'].best_estimator_.predict(X)
preds_regtree = estimator_search['REGTREE'].best_estimator_.predict(X)
preds_lreg = estimator_search['LREG'].best_estimator_.predict(X)
preds_gbm  = estimator_search['GBM'].best_estimator_.predict(X)
fig, ax = plt.subplots()
ax.plot(y, preds_dum, 'o', alpha = 0.5, mfc = 'none')
ax.plot(y, preds_regtree, 'o', alpha = 0.5, mfc = 'none')
ax.plot(y, preds_lreg, 'o', alpha = 0.5, mfc = 'none')
ax.plot(y, preds_gbm, 'o', alpha = 0.5, mfc = 'none')
plt.title("Estimator performance: actuals vs predicteds")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
plt.close()

# Plot grid search results
plot_grid_search(estimator_search['GBM'])
plt.show()

# Print entire grid search
grid_search_2_df(estimator_search['GBM'])

# # Check pipeline:    
# # 1 . Pull results from pipeline
# debug_df = pd.DataFrame.sparse.from_spmatrix(
#     estimator_search['GBM']
#     best_estimator_.named_steps['preprocessor'].transform(X))
# 
# # 2. Check StandardScaler
# s_scaler = StandardScaler()
# sum(s_scaler.fit_transform(np.array(X.Distance_km).reshape(-1,1))) == sum(debug_df[0])
# 
# # 3. Check OneHotEncoder
# one_hoter = OneHotEncoder()
# raw_month = X['Month'].value_counts().sort_index()
# one_hot_month = pd.DataFrame.sparse.from_spmatrix(
#     one_hoter.fit_transform(
#         np.array(X.Month).reshape(-1,1))
#     ).apply(sum, axis = 0).sort_index()
# pipe_month = debug_df.iloc[:, 1:13].apply(sum, axis = 0).sort_index()
# pd.DataFrame(dict(
#     raw_month = raw_month, 
#     one_hot_month = one_hot_month, 
#     pipe_month = pipe_month))
# 
# raw_freq = X['Frequency'].value_counts().sort_index()
# one_hot_freq = pd.DataFrame.sparse.from_spmatrix(
#     one_hoter.fit_transform(
#         np.array(X.Frequency).reshape(-1,1))
#     ).apply(sum, axis = 0).sort_index()
# pipe_freq = debug_df.iloc[:, 13:16].apply(sum, axis = 0).sort_index()
# pd.DataFrame(dict(
#     raw_freq = raw_freq, 
#     one_hot_freq = one_hot_freq, 
#     pipe_freq = pipe_freq))

# Results from my last `run_experiment.py` run

Auto-saved so I can paste tables into my report without re-copying numbers.

- Rows after loading/cleaning: **515131**
- Run: **full dataset**

## Regression (my four models + splits)
```
               split                  model        mse      rmse       mae  median_ae  max_error         r2  explained_var                                                                                                       note
              random      linear_regression  89.720910  9.472112  6.763457   4.983433  70.178685   0.235681       0.235761                                                                                                        NaN
              random       ridge_regression  89.720910  9.472112  6.763457   4.983433  70.178685   0.235681       0.235761                                                                                                alpha=0.001
              random          random_forest  78.877521  8.881302  6.290074   4.465148  68.456546   0.328055       0.329088                         {"max_depth": 28, "max_features": 0.5, "min_samples_leaf": 4, "n_estimators": 200}
              random hist_gradient_boosting  78.124829  8.838825  6.208950   4.327201  66.187074   0.334467       0.334539 {"l2_regularization": 0.1, "learning_rate": 0.06, "max_depth": 6, "max_iter": 350, "min_samples_leaf": 30}
     blocked_holdout      linear_regression  73.357758  8.564914  7.150527   6.437211  52.669305  -3.389052      -1.009943                                                                                                        NaN
     blocked_holdout       ridge_regression  73.357758  8.564914  7.150527   6.437211  52.669305  -3.389052      -1.009943                                                                                                alpha=0.001
     blocked_holdout          random_forest  76.474509  8.744971  7.326457   6.731786  37.301595  -3.575530      -1.490005                       {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 2, "n_estimators": 100}
     blocked_holdout hist_gradient_boosting  75.521186  8.690293  7.196032   6.507209  36.404603  -3.518492      -1.972139  {"l2_regularization": 0.0, "learning_rate": 0.1, "max_depth": 6, "max_iter": 250, "min_samples_leaf": 30}
future_extrapolation      linear_regression 193.363362 13.905516 13.139996  13.144581  57.027803 -29.107929      -2.484876                                                                                                        NaN
future_extrapolation       ridge_regression 193.363363 13.905516 13.139996  13.144582  57.027802 -29.107930      -2.484876                                                                                                alpha=0.001
future_extrapolation          random_forest 201.683334 14.201526 13.545511  13.271121  42.212845 -30.403403      -1.834242                         {"max_depth": 20, "max_features": 0.5, "min_samples_leaf": 4, "n_estimators": 100}
future_extrapolation hist_gradient_boosting 174.870076 13.223845 12.346529  12.008405  43.533948 -26.228405      -2.496518  {"l2_regularization": 0.0, "learning_rate": 0.1, "max_depth": 6, "max_iter": 250, "min_samples_leaf": 30}
```

## Baselines (mean / median predictors — for comparison)
```
               split                   model        mse      rmse       mae  median_ae  max_error         r2  explained_var              note
              random     baseline_train_mean 117.392342 10.834775  8.100304   6.614869  74.385131  -0.000047   0.000000e+00 constant=1998.385
              random   baseline_train_median 129.921069 11.398292  7.575036   5.000000  78.000000  -0.106778   0.000000e+00 constant=2002.000
     blocked_holdout     baseline_train_mean 124.017393 11.136310 10.358745   9.407683  19.407683  -6.420058   0.000000e+00 constant=2000.408
     blocked_holdout   baseline_train_median 211.345927 14.537741 13.951062  13.000000  23.000000 -11.644993   0.000000e+00 constant=2004.000
future_extrapolation     baseline_train_mean 269.799818 16.425584 16.228909  16.708003  21.708003 -41.009581   0.000000e+00 constant=1989.292
future_extrapolation   baseline_train_median 163.195425 12.774796 12.520906  13.000000  18.000000 -24.410586  -2.220446e-16 constant=1993.000
future_extrapolation baseline_last_seen_year  36.902742  6.074763  5.520906   6.000000  11.000000  -4.745996  -2.220446e-16 constant=2000.000
```
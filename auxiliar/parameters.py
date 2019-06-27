"""

Clase de configuración para la configuración de los entrenamientos

"""
import numpy as np
from scipy.stats  import randint as sp_randint
# Autor Adrián Insua Yañez

vect_params = {
    'vect__max_df': [0.5],
    'vect__min_df': [10],
    'vect__max_features': [500],
    'vect__ngram_range': [(1, 1)],
    }

# Aqui definimos el espacio de parámetros a explorar
parameters_ls = {
    'ls__estimator__base_estimator__C': [1.0, 0.5, 1.5],
    'ls__estimator__base_estimator__loss': ['hinge', 'squared_hinge'],
    'ls__estimator__base_estimator__max_iter': [1, 100]
}

parameters_lr = {
    'lr__estimator__C': [1.0, 0.5, 1.5],
    'lr__estimator__fit_intercept': [True, False],
    'lr__estimator__solver': ['liblinear', 'lbfgs'],
    'lr__estimator__max_iter': [100, 200]
}

parameters_mb = {
    'mb__estimator__alpha': [1.0, 0.5, 1.5],
    'mb__estimator__fit_prior': [True, False]
}

parameters_rf = {
    'rf__estimator__criterion': ['gini', 'entropy'],
    'rf__estimator__max_depth': [10, None],
    'rf__estimator__max_features': ['auto', 10],
    'rf__estimator__min_samples_split': [2, 10]
}

parameters_bin_ls = {
    'ls__base_estimator__C': [1.0, 0.5, 1.5],
    'ls__base_estimator__max_iter': [1, 100]
}

parameters_bin_lr = {
    'lr__C': [1.0, 0.5, 1.5],
    'lr__fit_intercept': [True, False],
    'lr__solver':  ['liblinear', 'lbfgs'],
    'lr__max_iter': [100, 200]
}

parameters_bin_mb = {
    'mb__alpha': [1.0, 0.5, 1.5],
    'mb__fit_prior': [True, False],
}

parameters_bin_rf = {
    'rf__criterion': ['gini', 'entropy'],
    'rf__max_depth': [10, None],
    'rf__max_features': ['auto', 10],
    'rf__min_samples_split': [2, 10]
}

parameters_bin = {
    'ls': parameters_bin_ls,
    'lr': parameters_bin_lr,
    'mb': parameters_bin_mb,
    'rf': parameters_bin_rf
}

parameters = {
    'ls': parameters_ls,
    'lr': parameters_lr,
    'mb': parameters_mb,
    'rf': parameters_rf
}
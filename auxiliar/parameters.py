"""

Clase de configuración para la configuración de los entrenamientos

"""
import numpy as np
from auxiliar.VectorizerHelper import tokenize, stemize
from scipy.stats  import randint as sp_randint
# Autor Adrián Insua Yañez

prepro_params = {
    'prep__negation': [ 0 , 1 ],
    'prep__repeated_letters': [ True ]
}

vect_params = {
    'vect__tokenizer': [stemize, tokenize],
    }

# Aqui definimos el espacio de parámetros a explorar
parameters_ls = {
    'ls__estimator__base_estimator__max_iter': [1000, 1200, 1500]
}

parameters_lr = {
    'lr__estimator__max_iter': [100, 200, 500]
}

parameters_mb = {
    'mb__estimator__alpha': [1.0, 0.5, 1.5]
}

parameters_rf = {
    'rf__estimator__n_estimators': [10, 100]
}

parameters_bin_ls = {
    'ls__base_estimator__max_iter': [1000, 1200, 1500]
}

parameters_bin_lr = {
    'lr__max_iter': [100, 200, 500]
}

parameters_bin_mb = {
    'mb__alpha': [1.0, 0.5, 1.5]
}

parameters_bin_rf = {
    'rf__n_estimators': [10, 100]
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
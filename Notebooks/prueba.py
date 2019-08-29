# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Machine learning Grid Search
#%% [markdown]
# ## Imports

#%%
import sys
import cufflinks
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import pickle

warnings.filterwarnings('ignore')

sys.path.append('./..')
cufflinks.go_offline()


#%%
from Corpus.Corpus import get_corpus, filter_binary_pn, filter_corpus_small
from auxiliar.VectorizerHelper import vectorizer, vectorizerIdf, preprocessor
from auxiliar import parameters
from auxiliar.HtmlParser import HtmlParser


#%%
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
import copy

#%% [markdown]
# ## Config

#%%
polarity_dim = 2
clasificadores=['lr', 'ls', 'mb', 'rf']
idf = False
target_names=['Neg', 'Pos']
kfolds = 10
base_dir = '2-clases' if polarity_dim == 2 else ('3-clases' if polarity_dim == 3 else '5-clases')
name = 'machine_learning/tweeter/grid_search'

#%% [markdown]
# ## Get data

#%%
# cine = HtmlParser(200, "http://www.muchocine.net/criticas_ultimas.php", 1)
data_corpus = get_corpus('general-corpus', 'general-corpus', 1, None)

if polarity_dim == 2:
    data_corpus = filter_binary_pn(data_corpus)
#     cine = filter_binary_pn(cine.get_corpus())
elif polarity_dim == 3:
    data_corpus = filter_corpus_small(data_corpus)
#     cine = filter_corpus_small(cine.get_corpus())
elif polarity_dim == 5:
    cine = cine.get_corpus()
# used_data = cine[:5000]
used_data = pd.DataFrame(data_corpus)

#%% [markdown]
# ## Split data

#%%
split = used_data.shape[0] * 0.7
train_corpus = used_data.loc[:split - 1 , :]
test_corpus = used_data.loc[split:, :]

#%% [markdown]
# ## Initialize ML

#%%
vect = vectorizerIdf if idf else vectorizer
ls = CalibratedClassifierCV(LinearSVC()) if polarity_dim == 2 else OneVsRestClassifier(CalibratedClassifierCV(LinearSVC()))
lr = LogisticRegression(solver='lbfgs') if polarity_dim == 2 else OneVsRestClassifier(LogisticRegression())
mb = MultinomialNB() if polarity_dim == 2 else OneVsRestClassifier(MultinomialNB())
rf = RandomForestClassifier() if polarity_dim == 2 else OneVsRestClassifier(RandomForestClassifier())


#%%
pipeline_ls = Pipeline([
    ('prep', copy.deepcopy(preprocessor)),
    ('vect', copy.deepcopy(vect)),
    ('ls', ls)
])
pipeline_lr = Pipeline([
    ('prep', copy.deepcopy(preprocessor)),
    ('vect', copy.deepcopy(vect)),
    ('lr', lr)
])
pipeline_mb = Pipeline([
    ('prep', copy.deepcopy(preprocessor)),
    ('vect', copy.deepcopy(vect)),
    ('mb', mb)
])
pipeline_rf = Pipeline([
    ('prep', copy.deepcopy(preprocessor)),
    ('vect', copy.deepcopy(vect)),
    ('rf', rf)
])


#%%
pipelines = {
    'ls': pipeline_ls,
    'lr': pipeline_lr,
    'mb': pipeline_mb,
    'rf': pipeline_rf
}
pipelines_train = {
    'ls': ls,
    'lr': lr,
    'mb': mb,
    'rf': rf
}


#%%
params = parameters.parameters_bin if polarity_dim == 2 else parameters.parameters


#%%
params

#%% [markdown]
# ## Train

#%%
folds = pd.read_pickle('./../data/pkls/folds.pkl') # k-folds precargados
folds = folds.values


#%%
results = {}
grids = {}
with tqdm(total=len(clasificadores) * 10) as pbar:
    for c in clasificadores:
        results[c] = { 'real': {}, 'predicted': {} }
        i = 0
        params[c].update(parameters.vect_params)
        params[c].update(parameters.prepro_params)
        param_grid = params[c]
        grid_search = GridSearchCV(pipelines[c], param_grid, verbose=2, scoring='accuracy', refit=True, cv=3)
        grid = grid_search.fit(train_corpus.content, train_corpus.polarity)
        grids[c] = grid
        best_parameters = grid.best_params_
        train_params = {}
        for param_name in sorted(parameters.vect_params.keys()):
            train_params.update({param_name[6:]: best_parameters[param_name]})
        vect.set_params(**train_params)
        preprocessor.set_params(**train_params)
        x_prepro = preprocessor.fit_transform(train_corpus.content)
        x_vect = vect.fit_transform(x_prepro, train_corpus.polarity).toarray()
        for train_index, test_index in folds:
            train_x = x_vect[train_index]
            train_y = train_corpus.polarity[train_index]
            test_x = x_vect[test_index]
            test_y = train_corpus.polarity[test_index]

            pipelines_train[c].fit(train_x, train_y)

            predicted = pipelines_train[c].predict(test_x)
            
            results[c]['real'][i] = test_y.values.tolist()
            results[c]['predicted'][i] = predicted.tolist()
            i = i + 1

            pbar.update(1)

    


#%% 
results


#%%
pd.DataFrame(results).to_pickle('../results/'+name+'/'+base_dir+'/results.pkl')


#%%
with open('../results/'+name+'/'+base_dir+'/grid_results.pkl', 'wb') as fp:
    pickle.dump(grid, fp)


#%%
test_results = {}
with tqdm(total=len(clasificadores)) as pbar:
    for c in clasificadores:
        test_results[c] = { 'real': {}, 'predicted': {} }
        i = 0
        grid = grids[c]
        best_parameters = grid.best_params_
        train_params = {}
        for param_name in sorted(parameters.vect_params.keys()):
            train_params.update({param_name[6:]: best_parameters[param_name]})
        vect.set_params(**train_params)
        vect.fit(data_corpus.content, data_corpus.polarity)
        x_vect = vect.transform(train_corpus.content).toarray()
        x_vect_test = vect.transform(test_corpus.content).toarray()
        train_x = x_vect
        train_y = train_corpus.polarity
        test_x = x_vect_test
        test_y = test_corpus.polarity

        pipelines_train[c].fit(train_x, train_y)

        predicted = pipelines_train[c].predict(test_x)

        test_results[c]['real'][i] = test_y.values.tolist()
        test_results[c]['predicted'][i] = predicted.tolist()
        i = i + 1

        pbar.update(1)


#%%
pd.DataFrame(test_results).to_pickle('../results/'+name+'/'+base_dir+'/test_results.pkl')


#%%




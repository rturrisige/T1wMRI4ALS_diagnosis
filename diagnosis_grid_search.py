"""
# rosanna.turrisi@irib.cnr.it

This code trains different ML classifiers looking for optimal parameters.
Best parameters are saved in a .csv file in saver_dir direcotry.

Feature data (radiomics, L7 or L8) and labels are supposed to be saved as numpy arrays.

"""
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse


def grid_search(X, y, saver_path, input_type='radiomics', nsplits=10):
    if not os.path.exists(saver_path):
        os.makedirs(saver_path)

    cv = StratifiedShuffleSplit(n_splits=nsplits, test_size=0.2, random_state=42)

    # SVM Parameters
    C_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
    gamma_range = [0.001, 0.01, 0.1, 1, 10]

    # Linear SVM
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    svm_parameters = {'model__C': C_range}
    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', LinearSVC())])

    svm_grid = GridSearchCV(pipe, param_grid=svm_parameters, cv=cv)
    svm_grid.fit(X, y)
    print(svm_grid.best_params_)
    linear_best = svm_grid.best_params_

    # SVM (P)

    from sklearn.svm import SVC
    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', SVC(kernel='poly'))])
    svm_parameters = {'model__C': C_range, 'model__degree': (2, 3)}
    svm_grid = GridSearchCV(pipe, param_grid=svm_parameters, cv=cv)
    svm_grid.fit(X, y)
    print(svm_grid.best_params_)
    poly_best = svm_grid.best_params_

    # SVM (RBF)
    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', SVC(kernel='rbf'))])
    svm_parameters = {'model__C': C_range, 'model__gamma': gamma_range}
    svm_grid = GridSearchCV(pipe, param_grid=svm_parameters, cv=cv)
    svm_grid.fit(X, y)
    print(svm_grid.best_params_)
    rbf_best = svm_grid.best_params_

    # SVM (S)
    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', SVC(kernel='sigmoid'))])
    svm_parameters = {'model__C': C_range}
    svm_grid = GridSearchCV(pipe, param_grid=svm_parameters, cv=cv)
    svm_grid.fit(X, y)
    print(svm_grid.best_params_)
    sigmoid_best = svm_grid.best_params_

    # RF

    from sklearn.ensemble import RandomForestClassifier
    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', RandomForestClassifier(random_state=0))])

    rf_parameters = {'model__n_estimators': [100, 200, 500]}
    rf_grid = GridSearchCV(pipe, param_grid=rf_parameters, cv=cv)
    rf_grid.fit(X, y)
    print(rf_grid.best_params_)
    rf_best = rf_grid.best_params_
    
    # KNN

    from sklearn.neighbors import KNeighborsClassifier as KN
    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', KN())])

    knn_parameters = {'model__n_neighbors': [3, 5, 7]}
    knn_grid = GridSearchCV(pipe, param_grid=knn_parameters, cv=cv)
    knn_grid.fit(X, y)
    print(knn_grid.best_params_)
    knn_best = knn_grid.best_params_

    df = pd.DataFrame({'SVM (L)': linear_best, 'SVM (P)': poly_best,
                        'SVM (RBF)': rbf_best, 'SVM (S)': sigmoid_best,
                       'KNN': knn_best, 'RF:': rf_best})
    df.to_csv(saver_path + '/' + input_type + '_best_parameters_' + str(nsplits) + 'CV.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Load mimicking and als numpy data
    from <data_dir>. Grid search for SVM, RF and KNN is performed. Results on the 
    best parameters are saved in a DataFrame file in <saver_dir>. """)
    parser.add_argument('--data_X', required=True, type=str,
                        help='Path/to/file containing input feature array.')
    parser.add_argument('--data_Y', required=True, type=str,
                        help='Path/to/file containing output label array.')
    parser.add_argument('--saver_dir', required=True, type=str,
                        help='The directory where to save the grid search results')
    parser.add_argument('--input_type', default='radiomics', type=str,
                        help='It must be radiomics, L7 or L8.')
    parser.add_argument('--n_splits', default=10, type=int,
                        help='Number of folders for cross validation grid search.')
    args = parser.parse_args()
    X = np.load(args.data_X)
    y = np.load(args.data_Y)
    grid_search(X, y, args.aver_path, args.input_type)


"""
# rosanna.turrisi@irib.cnr.it

This code trains different ML classifiers with optimal parameters from .csv file. 
Cross-validation is performed. Confusion matrix, and multiple-metrics evaluation is 
saved in saver_dir direcotry.

Feature data (radiomics, L7 or L8) and labels are supposed to be saved as numpy arrays.

Example of .csv file containing best parameters:
            Unnamed: 0  SVM (L)  SVM (P)  SVM (RBF)  KNN    RF:
             model__C    0.001      0.1      1.0      -      -
        model__degree      -        3.0       -  -    -
         model__gamma      -         -       0.1      -      -
   model__n_neighbors      -         -        -      7.0     -
  model__n_estimators      -         -        -       -    200.0

"""

from plot_functions import *
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from utilities import apply_svm
from sklearn.preprocessing import StandardScaler
from utilities import apply_classifier
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.ensemble import RandomForestClassifier
from models import DNN_2HL, DNN_3HL
from utilities import train_nobatch
from utilities import compute_CV_confusion_matrix, complete_evaluation
import argparse
import pandas as pd


def train(X, y, input_type, best_param, saver_path, cv=3):

    knn = KN(n_neighbors=int(best_param['KNN'][3]))
    rf = RandomForestClassifier(n_estimators=int(best_param['RF'][-1]), random_state=0)
    dnn_3l = DNN_3HL(X.shape[1])
    dnn_2l = DNN_2HL(X.shape[1])

    skf = StratifiedKFold(n_splits=cv)

    L_train_score, L_test_score, L_test_results, L_y_prob_0, L_y_prob_1 = [], [], [], [], []
    P_train_score, P_test_score, P_test_results, P_y_prob_0, P_y_prob_1 = [], [], [], [], []
    G_train_score, G_test_score, G_test_results, G_y_prob_0, G_y_prob_1 = [], [], [], [], []
    S_train_score, S_test_score, S_test_results, S_y_prob_0, S_y_prob_1 = [], [], [], [], []
    knn_train_score, knn_test_score, knn_test_results, knn_y_prob_0, knn_y_prob_1 = [], [], [], [], []
    rf_train_score, rf_test_score, rf_test_results, rf_y_prob_0, rf_y_prob_1 = [], [], [], [], []

    nn3_train_score, nn3_test_score, nn3_test_results, nn3_y_prob_0, nn3_y_prob_1 = [], [], [], [], []
    nn2_train_score, nn2_test_score, nn2_test_results, nn2_y_prob_0, nn2_y_prob_1 = [], [], [], [], []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        oversample = SMOTE(random_state=0)
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        X_test, y_test = X[test_index], y[test_index]

        # LINEAR
        L_train_score, L_test_score, L_test_results, L_y_prob_0, L_y_prob_1 = \
            apply_svm(X_train, y_train, X_test, y_test, 'linear', L_train_score, L_test_score, L_test_results,
                      L_y_prob_0, L_y_prob_1, saver_path + 'Linear_SVC_fold' + str(i) + '.sav', 
                      C=best_param['SVM (L)'][0])
        # POLINOMIAL
        P_train_score, P_test_score, P_test_results, P_y_prob_0, P_y_prob_1 = \
            apply_svm(X_train, y_train, X_test, y_test, 'poly',
                      P_train_score, P_test_score, P_test_results, P_y_prob_0, P_y_prob_1,
                      saver_path + 'Poly_SVC_fold' + str(i) + '.sav',
                      C=best_param['SVM (P)'][0], degree=best_param['SVM (P)'][1])
        # GAUSSIAN
        G_train_score, G_test_score, G_test_results, G_y_prob_0, G_y_prob_1 = \
            apply_svm(X_train, y_train, X_test, y_test, 'rbf',
                      G_train_score, G_test_score, G_test_results, G_y_prob_0, G_y_prob_1,
                      saver_path + 'RBF_SVC_fold' + str(i) + '.sav', 
                      C=best_param['SVM (RBF)'][0], gamma=best_param['SVM (L)'][2])

        # SIGMOID
        S_train_score, S_test_score, S_test_results, S_y_prob_0, S_y_prob_1 = \
            apply_svm(X_train, y_train, X_test, y_test, 'sigmoid',
                      S_train_score, S_test_score, S_test_results, S_y_prob_0, S_y_prob_1,
                      saver_path + 'Sigmoid_SVC_fold' + str(i) + '.sav',
                      C=best_param['SVM (S)'][0])
        # KNN (K=7)
        knn_train_score, knn_test_score, knn_test_results, knn_y_prob_0, knn_y_prob_1 =\
            apply_classifier(knn, X_train, y_train, X_test, y_test, knn_train_score,
                             knn_test_score, knn_test_results, knn_y_prob_0, knn_y_prob_1,
                             saver_path + 'KNN_k_fold' + str(i) + '.sav')
        
        # Random Forest
        rf_train_score, rf_test_score, rf_test_results, rf_y_prob_0, rf_y_prob_1 =\
            apply_classifier(rf, X_train, y_train, X_test, y_test,
                             rf_train_score, rf_test_score, rf_test_results, rf_y_prob_0, rf_y_prob_1,
                             saver_path + 'RandomForest_fold' + str(i) + '.sav')

        normalize = StandardScaler()
        normalize.fit(X_train, y_train)
        X_train = normalize.transform(X_train)
        X_test = normalize.transform(X_test)

        # 2HL Neural Network
        nn2_train_score, nn2_test_score, nn2_test_results, nn2_y_prob_0, nn2_y_prob_1 = \
            train_nobatch(dnn_2l, X_train, y_train, X_test, y_test,
                             nn2_train_score, nn2_test_score, nn2_test_results, nn2_y_prob_0, nn2_y_prob_1)


        # 3HL Neural Network
        nn3_train_score, nn3_test_score, nn3_test_results, nn3_y_prob_0, nn3_y_prob_1 = \
            train_nobatch(dnn_3l, X_train, y_train, X_test, y_test,
                             nn3_train_score, nn3_test_score, nn3_test_results, nn3_y_prob_0, nn3_y_prob_1)

    # Compute confusion matrix   

    L_df_cm = compute_CV_confusion_matrix(L_test_results)
    P_df_cm = compute_CV_confusion_matrix(P_test_results)
    G_df_cm = compute_CV_confusion_matrix(G_test_results)
    S_df_cm = compute_CV_confusion_matrix(S_test_results)
    knn_df_cm = compute_CV_confusion_matrix(knn_test_results)
    rf_df_cm = compute_CV_confusion_matrix(rf_test_results)

    nn2_df_cm = compute_CV_confusion_matrix(nn2_test_results)
    nn3_df_cm = compute_CV_confusion_matrix(nn3_test_results)

    cm_sum(L_df_cm, ['Mimicking', 'ALS'], saver_path, 'Linear_SVC')
    cm_sum(P_df_cm, ['Mimicking', 'ALS'], saver_path, 'Poly_SVC')
    cm_sum(G_df_cm, ['Mimicking', 'ALS'], saver_path, 'RBF_SVC')
    cm_sum(S_df_cm, ['Mimicking', 'ALS'], saver_path, 'Sigmoid_SVC')
    cm_sum(knn_df_cm, ['Mimicking', 'ALS'], saver_path, 'KNN')
    cm_sum(rf_df_cm, ['Mimicking', 'ALS'], saver_path, 'RandomForest')
    cm_sum(nn2_df_cm, ['Mimicking', 'ALS'], saver_path, '2L-DNN')
    cm_sum(nn3_df_cm, ['Mimicking', 'ALS'], saver_path, '3L-DNN')

    # Complete evaluation of the models
    L_df = complete_evaluation(L_test_results, L_y_prob_0, L_y_prob_1)
    P_df = complete_evaluation(P_test_results, P_y_prob_0, P_y_prob_1)
    G_df = complete_evaluation(G_test_results, G_y_prob_0, G_y_prob_1)
    S_df = complete_evaluation(S_test_results, S_y_prob_0, S_y_prob_1)
    knn_df = complete_evaluation(knn_test_results, knn_y_prob_0, knn_y_prob_1)
    rf_df = complete_evaluation(rf_test_results, rf_y_prob_0, rf_y_prob_1)

    nn2_df = complete_evaluation(nn2_test_results, nn2_y_prob_0, nn2_y_prob_1)
    nn3_df = complete_evaluation(nn3_test_results, nn3_y_prob_0, nn3_y_prob_1)

    cl_mean(L_df, ['Mimicking', 'ALS'], saver_path, 'Linear_SVC')
    cl_mean(P_df, ['Mimicking', 'ALS'], saver_path, 'Poly_SVC')
    cl_mean(G_df, ['Mimicking', 'ALS'], saver_path, 'RBF_SVC')
    cl_mean(S_df, ['Mimicking', 'ALS'], saver_path, 'Sigmoid_SVC')
    cl_mean(knn_df, ['Mimicking', 'ALS'], saver_path, 'KNN')
    cl_mean(rf_df, ['Mimicking', 'ALS'], saver_path, 'RandomForest')
    cl_mean(nn2_df, ['Mimicking', 'ALS'], saver_path, '2L-DNN')
    cl_mean(nn3_df, ['Mimicking', 'ALS'], saver_path, '3L-DNN')

    logfile = open(saver_path + 'all_results.txt', 'a')
    logfile.write('Results with optimal parameters\n\n')
    logfile.write('SVM results\n\n')

    for i in range(cv):
        logfile.write('Fold ' + str(i) + '\n')
        logfile.write('Linear SVM: Train={:.4f}, Test={:.4f}\n'.format(L_train_score[i], L_test_score[i]))
        logfile.write('Polynomial SVM: Train={:.4f}, Test={:.4f}\n'.format(P_train_score[i], P_test_score[i]))
        logfile.write('RBF SVM: Train={:.4f}, Test={:.4f}\n'.format(G_train_score[i], G_test_score[i]))
        logfile.write('Sigmoid SVM: Train={:.4f}, Test={:.4f}\n'.format(S_train_score[i], S_test_score[i]))
        logfile.write('KNN: Train={:.4f}, Test={:.4f}\n'.format(knn_train_score[i], knn_test_score[i]))
        logfile.write('RandomForest: Train={:.4f}, Test={:.4f}\n\n'.format(rf_train_score[i], rf_test_score[i]))
        logfile.write('2L-DNN: Train={:.4f}, Test={:.4f}\n'.format(nn2_train_score[i], nn2_test_score[i]))
        logfile.write('3L-DNN: Train={:.4f}, Test={:.4f}\n'.format(nn3_train_score[i], nn3_test_score[i]))

    logfile.flush()

    logfile.write('Average (F1 score)\n')
    logfile.write('Linear SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}\n'
                  .format(np.mean(L_train_score), np.std(L_train_score), np.mean(L_test_score), np.std(L_test_score)))
    logfile.write('Polynomial SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}\n'
                  .format(np.mean(P_train_score), np.std(P_train_score), np.mean(P_test_score), np.std(P_test_score)))
    logfile.write('RBF SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}\n'
                  .format(np.mean(G_train_score), np.std(G_train_score), np.mean(G_test_score), np.std(G_test_score)))
    logfile.write('Sigmoid SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}\n'
                  .format(np.mean(S_train_score), np.std(S_train_score), np.mean(S_test_score), np.std(S_test_score)))

    logfile.write('KNN: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}\n'
                  .format(np.mean(knn_train_score), np.std(knn_train_score), np.mean(knn_test_score), np.std(knn_test_score)))
    logfile.write('RandomForest: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}\n\n'
                  .format(np.mean(rf_train_score), np.std(rf_train_score), np.mean(rf_test_score), np.std(rf_test_score)))
    logfile.write('2L-DNN: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}\n'
                  .format(np.mean(nn2_train_score), np.std(nn2_train_score), np.mean(nn2_test_score), np.std(nn2_test_score)))
    logfile.write('3L-DNN: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}\n'
                  .format(np.mean(nn3_train_score), np.std(nn3_train_score), np.mean(nn3_test_score), np.std(nn3_test_score)))

    logfile.flush()
    logfile.close()

    # Save results 
    np.save(saver_path + 'LinearSVM_classification' + input_type + 'f1_train.npy', L_train_score)
    np.save(saver_path + 'LinearSVM_classification' + input_type + 'f1_test.npy', L_test_score)
    np.save(saver_path + 'PolySVM_classification' + input_type + 'f1_train.npy', P_train_score)
    np.save(saver_path + 'PolySVM_classification' + input_type + 'f1_test.npy', P_test_score)
    np.save(saver_path + 'RbfSVM_classification' + input_type + 'f1_train.npy', G_train_score)
    np.save(saver_path + 'RbfSVM_classification' + input_type + 'f1_test.npy', G_test_score)
    np.save(saver_path + 'SigmoidSVM_classification' + input_type + 'f1_train.npy', S_train_score)
    np.save(saver_path + 'SigmoidSVM_classification' + input_type + 'f1_test.npy', S_test_score)
    np.save(saver_path + 'KNN_classification' + input_type + 'f1_train.npy', knn_train_score)
    np.save(saver_path + 'KNN_classification' + input_type + 'f1_test.npy', knn_test_score)
    np.save(saver_path + 'RandomForest_classification' + input_type + 'f1_train.npy', rf_train_score)
    np.save(saver_path + 'RandomForest_classification' + input_type + 'f1_test.npy', rf_test_score)
    np.save(saver_path + '2L-DNN_classification' + input_type + 'f1_train.npy', nn2_train_score)
    np.save(saver_path + '2L-DNN_classification' + input_type + 'f1_test.npy', nn2_test_score)
    np.save(saver_path + '3L-DNN_classification' + input_type + 'f1_train.npy', nn3_train_score)
    np.save(saver_path + '3L-DNN_classification' + input_type + 'f1_test.npy', nn3_test_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Load mimicking and als numpy data
    from <data_dir>. SVM, RF and KNN are trained. Results on the testing set are saved in 
    <saver_dir>. """)
    parser.add_argument('--data_X', required=True, type=str,
                        help='Path/to/file containing input feature array.')
    parser.add_argument('--data_Y', required=True, type=str,
                        help='Path/to/file containing output label array.')
    parser.add_argument('--saver_dir', required=True, type=str,
                        help='The directory where to save the grid search results')
    parser.add_argument('--best_parameters', required=True, type=str,
                        help='Path/to/file containing the best parameters from grid search')
    parser.add_argument('--input_type', default='radiomics', type=str,
                        help='It must be radiomics, L7 or L8.')
    parser.add_argument('--n_splits', default=2, type=int,
                        help='Number of folders for cross validation.')
    args = parser.parse_args()
    best_param = pd.read_csv(args.best_param)
    X = np.load(args.data_X)
    y = np.load(args.data_Y)
    train(X, y, args.input_type, best_param, args.saver_path, args.n_splits)




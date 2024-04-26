import torch
from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy as dcopy
import os
from torchmetrics.classification import BinaryF1Score
f1torch = BinaryF1Score()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pickle
tonpy = lambda x: x.detach().cpu().numpy()
totorch = lambda x: torch.from_numpy(x).to(device)
torchtensor = lambda x: torch.from_numpy(x)
import sys
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, roc_auc_score, auc, precision_recall_curve, confusion_matrix
import pandas as pd

##
# LOADER


class AE_loader(Dataset):
    def __init__(self, dataset, norm=None):
        """
        dataset : list of all filenames
        transform (callable) : a function/transform that acts on the images (e.g., normalization).
        """
        self.dataset = dataset
        self.norm = norm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x_batch = self.dataset[index]
        if self.norm:
            xm, xs = np.mean(x_batch, 0), np.std(x_batch, 0)
            x_batch = (x_batch - xm)/xs
        return totorch(x_batch)

##


def AE_train(net, config, train_loader, val_loader, logfile=None):

    # PARAMETER DEFINITION:
    epochs_loss, epochs_val_results, epochs_train_acc = [], [], []
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.l2_reg)
    updated_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)
    nerrors, best_epoch = 0, 0
    total_step = len(train_loader)
    print('Number of total step per epoch:', total_step)

    # STARTING TRAINING:
    for epoch_counter in range(config.num_epochs):

        # EPOCH TRAINING:
        epoch_loss = 0.0
        for step, batch_x in enumerate(train_loader):
            net = net.train()
            reconstruction = net(batch_x)
            loss = config.criterion(reconstruction, batch_x)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            if step == 0 and epoch_counter != 0:
                if current_lr > 0.00001:
                    updated_lr.step()
            optimizer.step()
            if step % 10 == 0:
                print('Epoch {} step {} - Loss {:.4f}'.format(epoch_counter, step, loss.item()))
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # VALIDATION
        val_results = 0.0
        for _, batch_x in enumerate(val_loader):
            net = net.eval()
            reconstruction = net(batch_x)
            val_mse = config.criterion(reconstruction, batch_x)
            val_results += val_mse.item()
        val_results /= len(val_loader)
        if epoch_counter == 0:
            best_val_result = val_results
            net_weights = dcopy(net.state_dict())

        # PRINT RESULTS
        print('Epoch {} LR {:.8f} - Loss {:.4f} - Val MSE {:.4f}'.format(epoch_counter, current_lr,
                                                                         epoch_loss / total_step,
                                                                         val_results))
        if logfile:
            logfile.write('Epoch {} LR {:.8f} - Loss {} - Val MSE {}\n'.format(epoch_counter, config.lr,
                                                                               epoch_loss / total_step,
                                                                               val_results))
            logfile.flush()
        print('')
        epochs_loss.append(epoch_loss / total_step)
        epochs_val_results.append(val_results)

        # EARLY STOPPING
        if (val_results > best_val_result) or (val_results == best_val_result):
            nerrors += 1
            if nerrors > config.patience:
                print('Early stopping applied. Best epoch {}. Val. MSE {:.4f}'.format(best_epoch, best_val_result))
                if logfile:
                    logfile.write('Early stopping applied. Best epoch {}. Val. MSE {:.4f}'.format(best_epoch, best_val_result))
                    logfile.flush()
                net.load_state_dict(net_weights)
                break
        else:
            print('Saved weights at epoch', epoch_counter)
            torch.save(dcopy(net.state_dict()), config.saver_path + 'AE_Exp' + str(config.nexp) + '_model_best_weights.pt')
            net_weights = dcopy(net.state_dict())
            best_val_result = val_results
            best_epoch = epoch_counter
            print('best epoch', best_epoch)
            nerrors = 0

    net.load_state_dict(net_weights)
    print('best epoch', best_epoch)
    return epochs_loss, epochs_val_results, best_epoch


def train_classification(net, config, X_train, y_train, X_val, y_val, param_to_train, logfile=None, metric='acc'):

    # PARAMETER DEFINITION:
    epochs_loss, epochs_val_results, epochs_train_acc = [], [], []
    optimizer = torch.optim.Adam(param_to_train, lr=config.lr, weight_decay=config.l2_reg)
    updated_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)
    nerrors, best_epoch = 0, 0
    net_weights = dcopy(net.state_dict())
    best_val_result = 0.0

    # STARTING TRAINING:
    for epoch_counter in range(config.num_epochs):

        # EPOCH TRAINING:
        epoch_loss = 0.0
        net = net.train()
        output = net(X_train)
        epoch_loss = config.criterion(output, y_train.long())
        optimizer.zero_grad()
        epoch_loss.backward()
        if epoch_counter != 0:
            if current_lr > 0.00001:
                updated_lr.step()
        optimizer.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # VALIDATION
        net = net.eval()
        pred_val = torch.max(net(X_val), 1)[1]

        if metric == 'accuracy':
            # Compute accuracy
            val_results = torch.sum(pred_val == y_val.long()).to(dtype=torch.float).item()
            val_results /= X_val.shape[0]
        elif metric == 'f1 score':
            # Compute f1_score
            val_results = f1torch(pred_val, y_val.long()).item()

        epoch_loss = epoch_loss.item()
        # PRINT RESULTS
        print('Epoch {} LR {:.8f} - Loss {:.4f} - Val {} {:.4f}'.format(epoch_counter, current_lr,
                                                                         epoch_loss, metric,
                                                                         val_results))
        if logfile:
            logfile.write('Epoch {} LR {:.8f} - Loss {} - Val {} {}\n'.format(epoch_counter, config.lr,
                                                                               epoch_loss, metric,
                                                                               val_results))
            logfile.flush()
        print('')
        epochs_loss.append(epoch_loss)
        epochs_val_results.append(val_results)

        # EARLY STOPPING
        if (val_results < best_val_result) or (val_results == best_val_result):
            nerrors += 1
            if nerrors > config.patience:
                print('Early stopping applied. Best epoch {}. Val. {} {:.4f}'.format(best_epoch, metric, best_val_result))
                if logfile:
                    logfile.write('Early stopping applied. Best epoch {}. Val. {} {:.4f}'.format(best_epoch, metric, best_val_result))
                    logfile.flush()
                net.load_state_dict(net_weights)
                break
        else:
            print('Saved weights at epoch', epoch_counter)
            torch.save(dcopy(net.state_dict()), config.saver_path + 'Exp' + str(config.nexp) + '_best_weights.pt')
            net_weights = dcopy(net.state_dict())
            best_val_result = val_results
            best_epoch = epoch_counter
            print('best epoch', best_epoch)
            nerrors = 0

    net.load_state_dict(net_weights)
    print('best epoch', best_epoch)
    return epochs_loss, epochs_val_results, best_epoch


def test_model(net, data_loader, metric='accuracy'):
    net = net.eval()
    tot = 0.0
    N = 0
    for _, (x, y) in enumerate(data_loader):
        pred = torch.max(net(x), 1)[1]
        if metric == 'accuracy':
            results = torch.sum(pred == y.long()).to(dtype=torch.float).item()
        elif metric == 'f1 score':
            results = f1torch(pred, y.long()).item()
        else:
            print('Metric not found. Allowed choices: accuracy, f1 score.')
            sys.exit()
        tot += results
        N += x.shape[0]
    return tot / N


import torch
from torch.utils.data import Dataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tonpy = lambda x: x.detach().cpu().numpy()
totorch = lambda x: torch.from_numpy(x).to(device)
torchtensor = lambda x: torch.from_numpy(x)


class AE_loader(Dataset):
    def __init__(self, dataset, norm=None):
        """
        dataset : list of all filenames
        transform (callable) : a function/transform that acts on the images (e.g., normalization).
        """
        self.dataset = dataset
        self.norm = norm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x_batch = self.dataset[index]
        if self.norm:
            xm, xs = np.mean(x_batch, 0), np.std(x_batch, 0)
            x_batch = (x_batch - xm)/xs
        return totorch(x_batch)


from sklearn.metrics import f1_score


def train_no_batch(net, config, X_train, y_train, X_val, y_val, logfile=False):
    # PARAMETER DEFINITION:
    train_acc = config.return_train_acc
    return_prob = config.return_train_prob
    epochs_loss, epochs_dev_acc, epochs_train_acc = [], [], []
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.l2_reg)
    # updated_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)
    nerrors, best_f1, all_probs = 0, 0.0, []
    net_weights = dcopy(net.state_dict())
    # STARTING TRAINING:
    for epoch_counter in range(config.num_epochs):

        # EPOCH TRAINING:
        net = net.train()
        prob = net(X_train)
        loss = config.criterion(prob, y_train.long())
        epoch_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if return_prob:
            prob = tonpy(prob)
            all_probs.append(prob)
        net = net.eval()
        val_out = tonpy(torch.max(net(X_val), 1)[1])
        f1_val = f1_score(val_out, y_val)
        # COMPUTE TRAINING ACCURACY:

        print('Epoch {}: Loss {:.4f} - Val F1 score {:.4f}'.format(epoch_counter,epoch_loss, f1_val))
        if logfile:
            logfile.write('Epoch {}  - Loss {} - Val F1 score {}\n'.format(epoch_counter,
                                                                                   epoch_loss,
                                                                                   f1_val))
        print('')
        epochs_loss.append(epoch_loss)
        epochs_dev_acc.append(f1_val)
        # COMPUTE VAL ACCURACY AND APPLY EARLY STOPPING
        if (f1_val < best_f1) or (f1_val == best_f1):
            nerrors += 1
            if nerrors > config.maxerror:
                print('Early stopping applied at iteration', epoch_counter, '. Dev F1=', best_f1)
                if logfile:
                    logfile.write('Early stopping applied at epoch {}. Val. F1 score {:.4f}'.format(epoch_counter,
                                                                                                    best_f1))
                    logfile.flush()
                net.load_state_dict(net_weights)
                break
        else:
            print('Saved weights at epoch', epoch_counter)
            torch.save(dcopy(net.state_dict()), config.saver_path + 'Exp' + str(config.nexp) + '_nn_best_weights.pt')
            print('At ', config.saver_path + 'nn_best_weights.pt')
            net_weights = dcopy(net.state_dict())
            best_epoch = dcopy(epoch_counter)
            best_f1 = f1_val
            nerrors = 0
    net.load_state_dict(net_weights)
    to_return = [epochs_loss, epochs_dev_acc]
    if train_acc:
        to_return.append(epochs_train_acc)
    if return_prob:
        to_return.append(all_probs)
    to_return.append(best_f1)
    return to_return


def apply_classifier(algorithm, X_train, y_train, X_test, y_test, train_score, test_score, test_results, y_prob_0, y_prob_1, saver_name=None):
    clf = make_pipeline(StandardScaler(), algorithm)
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    y_prob_0.append(clf.predict_proba(X_test)[:, 0])
    y_prob_1.append(clf.predict_proba(X_test)[:, 1])
    train_f1 = f1_score(y_train, train_pred)
    test_results.append([y_test, test_pred])
    test_f1 = f1_score(y_test, test_pred)
    print('F1 score. Train={:.4f}. Test={:.4f}.'.format(train_f1, test_f1))
    print('')
    train_score.append(train_f1)
    test_score.append(test_f1)
    if saver_name:
        pickle.dump(clf, open(saver_name, 'wb'))
    return train_score, test_score, test_results, y_prob_0, y_prob_1


def apply_svm(X_train, y_train, X_test, y_test, kernel, train_score, test_score, test_results, y_prob_0, y_prob_1,
              saver_name=None, gamma=None, degree=None, C=None):
    if kernel == 'linear' or 'sigmoid' and C:
        algorithm = SVC(kernel=kernel, C=C, probability=True)
    elif kernel == 'rbf' and C and gamma:
        algorithm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    elif kernel == 'poly' and C and degree:
        algorithm = SVC(kernel=kernel, C=C, degree=degree, probability=True)
    else:
        algorithm = SVC(kernel=kernel, probability=True)
    print('SVM with kernel:', kernel)
    train_score, test_score, test_results, y_prob_0, y_prob_1 = \
        apply_classifier(algorithm, X_train, y_train, X_test, y_test, train_score, test_score, test_results,
                         y_prob_0, y_prob_1, saver_name=saver_name)
    return train_score, test_score, test_results, y_prob_0, y_prob_1


def complete_evaluation(test_results, y_prob_0, y_prob_1):
    precision_0, recall_0, precision_1, recall_1 = [], [], [], []
    f1_0, f1_1, roc_auc_0, roc_auc_1, auprc_0, auprc_1 = [], [], [], [], [], []
    for i in range(len(test_results)):
        y_test, y_pred = test_results[i]
        yp0 = y_prob_0[i]
        yp1 = y_prob_1[i]
        precision_0.append(precision_score(y_test, y_pred, pos_label=0))
        recall_0.append(recall_score(y_test, y_pred, pos_label=0))
        precision_1.append(precision_score(y_test, y_pred, pos_label=1))
        recall_1.append(recall_score(y_test, y_pred, pos_label=1))
        f1_0.append(f1_score(y_test, y_pred, pos_label=0))
        f1_1.append(f1_score(y_test, y_pred, pos_label=1))
        roc_auc_0.append(roc_auc_score(y_test, yp0))
        roc_auc_1.append(roc_auc_score(y_test, yp1))
        prec_0, rec_0, _ = precision_recall_curve(y_test, yp0, pos_label=0)
        prec_1, rec_1, _ = precision_recall_curve(y_test, yp1)
        auprc_0.append(auc(rec_0, prec_0))
        auprc_1.append(auc(rec_1, prec_1))

    df = pd.DataFrame({'Precision Positive Class': precision_1, 'Precision Negative Class': precision_0,
                       'Recall Positive Class': recall_1, 'Recall Negative Class': recall_0,
                       'F1-score Negative Class': f1_0,'F1-score Positive Class': f1_1,
                       'AUC Positive Class': roc_auc_1, 'AUC Negative Class': roc_auc_0,
                       'AUPRC Positive Class': auprc_1, 'AUPRC Negative Class': auprc_0})
    return df


def compute_CV_confusion_matrix(test_results):
    all_tn, all_fp, all_fn, all_tp = [], [], [], []
    for y_test, y_pred in test_results:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        all_tn.append(tn)
        all_fp.append(fp)
        all_tp.append(tp)
        all_fn.append(fn)
    df = pd.DataFrame({'True Positive': all_tp, 'True Negative': all_tn, 'False Positive': all_fp, 'False Negative': all_fn})
    return df


##

import torch.nn as nn
def to_onehot(y):
    onehot = np.zeros([y.shape[0], 2])
    class0 = np.where(y==0)[0]
    class1= np.where(y==1)[0]
    onehot[class0, 0] = 1
    onehot[class1, 1] = 1
    return onehot


def train_nobatch(net, X_train, y_train, X_test, y_test, train_score, test_score,
                  test_results, y_prob_0, y_prob_1, max_epochs=200):
    # PARAMETER DEFINITION:
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.1)
    X_train = totorch(X_train).float()
    y_train = totorch(y_train)
    X_test = totorch(X_test).float()
    # STARTING TRAINING:
    for epoch_counter in range(max_epochs):
        # EPOCH TRAINING:
        net = net.train()
        output = net(X_train)
        epoch_loss = loss(output, y_train.long())
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss.item()
        # PRINT RESULTS
        print('Epoch {:.4f} - Loss {:.4f}'.format(epoch_counter, epoch_loss))
        print('')

    # TESTING
    net = net.eval()
    test_prob = net(X_test)
    y_prob_0.append(tonpy(test_prob[:, 0]))
    y_prob_1.append(tonpy(test_prob[:, 1]))
    test_output = tonpy(torch.max(test_prob, 1)[1])
    test_score.append(f1_score(y_test, test_output))
    test_results.append([y_test, test_output])

    train_output = tonpy(torch.max(net(X_train), 1)[1])
    train_score.append(f1_score(tonpy(y_train), train_output))
    return train_score, test_score, test_results, y_prob_0, y_prob_1

##


def check_labels(all_subjects, code_to_check, feature):
    code_with_labels = []
    labels = []
    for name in code_to_check:
        code = name.split('_')[0]
        correspondence = np.where(np.array(all_subjects) == code)[0]
        label = feature[correspondence[0]]
        if isinstance(label, str):
            code_with_labels.append(name)
            labels.append(label)
        elif isinstance(label, float) or isinstance(label, int):
            if not np.isnan(label):
                code_with_labels.append(name)
                labels.append(label)
    return code_with_labels, np.array(labels)
"""
Utilities for models, e.g. training them and evaluating performance
"""

import sys
import os
import copy
import itertools
import warnings
import multiprocessing
import time
import shutil
import logging
import collections
import functools

import tqdm

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import utils
import data_loader

if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
else:
    DEVICE = torch.device("cpu")
    logging.warn("Non-gpu device: {}".format(DEVICE))

# Default data loader parameters
DATA_LOADER_PARAMS = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": 6,
}

OPTIMIZER_PARAMS = {
    "lr": 1e-3,
}

ModelPerf = collections.namedtuple('ModelPerf', ['auroc', 'auroc_curve', 'auprc', 'auprc_curve', 'accuracy', 'recall', 'precision', 'f1', 'ce_loss'])

class RandomClassifier(object):
    """
    Predict randomly according to random uniform distribution
    """
    def __init__(self, n_classes:int=8, seed:int=7737):
        self.n_classes = n_classes
        self.seed = seed

    def predict_proba(self, _x):
        np.random.seed(self.seed)  # Calling predict several times will return same result
        return np.random.uniform(low=0.0, high=1.0, size=(_x.shape[0], self.n_classes))

class PadSequence(object):
    """
    Helper function for performing collation of a batch for RNN/LSTM batching. Namely, it sorst the batch by
    sequence length and pads with 0s. Based on the following link:
    https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/

    Example usage:
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, collate_fn=PadSequence())
    """
    def __call__(self, batch):
        """Batch comes as a list of examples where each example is a tuple of (data, label)"""
        # Assume that each element in "batch" is a tuple (data, label).
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)  # Sort the batch in descending order
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0)
        # Also need to store the length of each sequence, later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = torch.FloatTensor(np.vstack([b[1] for b in sorted_batch]))
        assert lengths.shape[0] == sequences_padded.shape[1] == labels.shape[0]
        return sequences_padded, lengths, labels

class EarlyStopper(object):
    """
    Helper class that monitors a given metric and determines if training should continue
    Namely, stops training after we have gone <patience> epochs without an improvement
    """
    def __init__(self, metric='auroc', patience=3, mode='max', verbose=False):
        assert patience > 0
        self.metric = metric
        self.mode = mode
        assert self.mode in ['max', 'min']
        self.patience = patience
        self.verbose = verbose

        self.metric_values = []

    def record_epoch(self, model_perf):
        """Return True if training should stop"""
        self.metric_values.append(model_perf._asdict()[self.metric])
        best_epoch = np.argmax(self.metric_values) if self.mode == 'max' else np.argmin(self.metric_values)
        # If best epoch was first epoch, and we are on second epoch, this evaluates to
        # 2 - 0 - 1 = 1 epochs without improvement
        # If best epocvh was second epoch, and we are on 5th epoch, this evaluates to
        # 5 - 1 - 1 = 3 epochs without improvement
        epochs_without_improvement = len(self.metric_values) - best_epoch - 1
        if epochs_without_improvement >= self.patience:
            if self.verbose:
                print(f"{epochs_without_improvement} epochs without improvement - sending signal to stop training")
            return True
        return False

def single_train(model, train_data, valid_data, verbose=True):
    """
    Train the sklearn model on training data and report results on validation
    train_data and valid_data should be tuples of (x, y)
    """
    if verbose:
        print(type(model))
    model.fit(*train_data)

    validation_probs = model.predict_proba(valid_data[0])[:, 1]
    retval = generate_model_perf(valid_data[1].flatten(), validation_probs)
    if verbose:
        print("AUC:\t", retval.auroc)
        print("AUPRC:\t", retval.auprc)
        print("Acc:\t", retval.accuracy)
        print("Rec:\t", retval.recall)
        print("Prec:\t", retval.precision)
        print("F1:\t", retval.f1)
    return model, retval

def multi_train(model, train_data, valid_data, verbose=True):
    """
    Train the sklearn model on training data and report results on validation
    train_data and valid_data should be tuples of (x, y). This function is designed to
    handle multi-label classification
    """
    if verbose:
        print(type(model))
    model.fit(*train_data)

    validation_probs = model.predict_proba(valid_data[0])
    num_classes = train_data[1].shape[1]
    assert valid_data[1].shape[1] == num_classes, "Valid data has unexpected shape for {} classes: {}".format(num_classes, valid_data[1].shape)

    # Create per-class performance
    per_class_performance = []
    per_class_pos_probs = []
    for class_index in range(num_classes):
        if isinstance(validation_probs, list):
            preds = validation_probs[class_index][:, 1]
        else:
            preds = validation_probs[:, class_index]  # Handles OneVsRestClassifier output of (n_samples, n_classes)
        per_class_pos_probs.append(np.atleast_2d(preds).T)
        perf = generate_model_perf(valid_data[1][:, class_index].flatten(), preds)
        per_class_performance.append(perf)
        if verbose:
            print("Class: {}".format(class_index))
            print("AUC:\t", perf.auroc)
            print("AUPRC:\t", perf.auprc)
            print("Acc:\t", perf.accuracy)
            print("Rec:\t", perf.recall)
            print("Prec:\t", perf.precision)
            print("F1:\t", perf.f1)

    overall_model_probs = np.hstack(per_class_pos_probs)
    overall_performance = generate_model_perf(
            valid_data[1],
            overall_model_probs,
            multiclass=True,
    )
    return model, per_class_performance, overall_performance

def cross_validate(model, folds, model_kwargs=None):
    """
    Perform cross validation, returning a two lists, overall perf and class perf, for each fold
    If model_kwargs is not given, hyperparams are fitted in each round
    """
    retval_overall, retval_class, retval_clustering = [], [], []
    param_combos = list(itertools.product(*model_kwargs.values()))
    for train_data, valid_data, test_data in folds:
        per_fold_overall_perfs = []
        per_fold_models = []
        for combo in param_combos:
            kwargs = {k:v for k, v in zip(model_kwargs.keys(), combo)}
            trained_model, per_class_perf, overall_perf = multi_train(
                model(**kwargs),
                train_data,
                valid_data,
                verbose=False,
            )
            per_fold_overall_perfs.append(overall_perf)
            per_fold_models.append(trained_model)
        per_fold_aurocs = [p.auroc for p in per_fold_overall_perfs]  # AUROCs for each hyperparam combo
        best_idx = np.argmax(per_fold_aurocs)
        best_model = per_fold_models[best_idx]  # Select best best_model based on validation AUROC
        fold_test_preds = best_model.predict_proba(test_data[0])
        if isinstance(fold_test_preds, (tuple, list)):
            fold_test_preds = list_preds_to_array_preds(fold_test_preds)
        fold_test_perf = generate_model_perf(test_data[1], fold_test_preds, multiclass=True)
        fold_test_per_class_perf = generate_multiclass_perf(test_data[1], fold_test_preds, 8)
        fold_test_clustering = sns.clustermap(
            pd.DataFrame(fold_test_preds, columns=data_loader.LOCALIZATIONS),
            row_cluster=False,
            col_cluster=True,
            cmap='Spectral',
            metric='cosine',
        )
        retval_overall.append(fold_test_perf)
        retval_class.append(fold_test_per_class_perf)
        retval_clustering.append(fold_test_clustering)
    return retval_overall, retval_class, retval_clustering

def generate_model_perf(truths, pred_probs, multiclass=False):
    """Given truths, and predicted probabilities, generate ModelPerf object"""
    pred_classes = np.round(pred_probs).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        retval = ModelPerf(
            auroc=metrics.roc_auc_score(truths, pred_probs),
            auroc_curve=metrics.roc_curve(truths, pred_probs) if not multiclass else None,
            auprc=metrics.average_precision_score(truths, pred_probs),
            auprc_curve=metrics.precision_recall_curve(truths, pred_probs) if not multiclass else None,
            accuracy=metrics.accuracy_score(truths, pred_classes) if not multiclass else None,
            recall=metrics.recall_score(truths, pred_classes) if not multiclass else None,
            precision=metrics.precision_score(truths, pred_classes) if not multiclass else None,
            f1=metrics.f1_score(truths, pred_classes) if not multiclass else None,
            ce_loss=metrics.log_loss(truths, pred_probs, normalize=False) / np.prod(truths.shape),
        )
    return retval

def generate_multiclass_perf(truths, pred_probs, num_classes):
    """Given truths, return a list of ModelPerf objects, one for each class"""
    per_class_perfs = []
    for class_index in range(num_classes):
        if isinstance(pred_probs, list):
            preds = pred_probs[class_index][:, 1]
        else:
            preds = pred_probs[:, class_index]  # Handles OneVsRestClassifier output of (n_samples, n_classes)
        perf = generate_model_perf(truths[:, class_index].flatten(), preds)
        per_class_perfs.append(perf)
    return per_class_perfs

def _eval_preds_performance(preds, y, metric):
    """Return per-class metric on the given preds/y. Helper function for below"""
    assert preds.shape == y.shape
    retval = []
    for j in range(y.shape[1]):
        val = metric(y[:, j].flatten(), preds[:, j].flatten())
        retval.append(val)
    return retval

def permutation_feature_importance(model, x, y, metric=metrics.roc_auc_score, seed=36261, threads=multiprocessing.cpu_count()):
    """Return a matrix of performance in each class when each feature is permuted"""
    n_features = x.shape[1]
    xs = []
    np.random.seed(seed)
    for i in range(n_features):  # For each feature
        x_copy = np.copy(x)  # Permutation happens in-place later
        np.random.shuffle(x_copy[:, i])  # Shuffle that feature
        xs.append(x_copy)
    x_all = np.vstack(xs)
    preds_all = model.predict_proba(x_all)  # parallel because model is parallel
    # Reformat into a matrix instead of a list of preds
    preds_all_mat = np.hstack([np.atleast_2d(preds_all[j][:, 1]).T for j in range(len(preds_all))])
    # Split into per-feature chunks
    preds_by_ft = np.split(preds_all_mat, n_features, axis=0)
    # Evaluate metric in parallel
    eval_pfunc = functools.partial(_eval_preds_performance, y=y, metric=metric)
    pool = multiprocessing.Pool(threads)
    results = pool.map(eval_pfunc, preds_by_ft)
    pool.close()
    pool.join()
    return np.vstack(results)

def pytorch_train(net, train_data, valid_data, max_epochs=30, early_stop_patience=3, loss=nn.BCELoss, optim=torch.optim.Adam, data_loader_kwargs=DATA_LOADER_PARAMS, optim_kwargs=OPTIMIZER_PARAMS, verbose=True, progressbar=False):
    """
    Train the given pytorch model.
    Return the performance at the end of every epoch, as well as the path to each epoch checkpoint
    """
    assert isinstance(train_data, torch.utils.data.Dataset)
    assert isinstance(valid_data, torch.utils.data.Dataset)
    if data_loader_kwargs is not None:  # If this is none, then we don't use the DataLoader wrapper
        train_loader = torch.utils.data.DataLoader(train_data, **data_loader_kwargs)

    net.to(DEVICE)
    criterion = loss()
    optimizer = optim(net.parameters(), **optim_kwargs)
    early_stopper = EarlyStopper(patience=early_stop_patience)

    epoch_dicts = []
    np.random.seed(48381)
    pbar = tqdm.tqdm_notebook if utils.isnotebook() else tqdm.tqdm
    for epoch in pbar(range(max_epochs), disable=not progressbar, desc=f"Epochs"):
        running_loss = 0.0  # Tracks the average loss at each epoch
        num_examples = 0
        if data_loader_kwargs is None:
            indices = np.arange(len(train_data))
            np.random.shuffle(indices)
            train_loader = (train_data[i] for i in indices)  # Manual generator if not using DataLoader
        for local_batch, local_labels in train_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(DEVICE), local_labels.to(DEVICE)
            # Zero out gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            local_outputs = net(local_batch)
            if isinstance(local_outputs, tuple):
                local_outputs = local_outputs[0]  # If there are multiple return values, assume first output is target for loss func
            loss = criterion(torch.squeeze(local_outputs), torch.squeeze(local_labels))
            loss.backward()
            optimizer.step()

            num_examples += local_outputs.shape[0]
            running_loss += loss.item()
        # Evaluate validation loss per epoch
        valid_overall_perf, valid_per_class_perf, _valid_truths, _valid_preds = pytorch_eval(net, valid_data, data_loader_kwargs=data_loader_kwargs)
        amortized_loss = running_loss / float(num_examples)
        if verbose:
            print(f"Train loss:          {amortized_loss}")
            print(f"Valid loss:          {valid_overall_perf.ce_loss}")
            print("Validation accuracy: {}".format(valid_overall_perf.accuracy))
            print("Validation F1:       {}".format(valid_overall_perf.f1))
            print("Validation AUPRC:    {}".format(valid_overall_perf.auprc))
            print("Validation AUROC:    {}".format(valid_overall_perf.auroc))
        epoch_dict = {
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(state_dict_to_cpu(net.state_dict())),
            # 'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'train_loss': amortized_loss,
            'valid_loss': valid_overall_perf.ce_loss,
            'auroc': valid_overall_perf.auroc,
            'auprc': valid_overall_perf.auprc,
        }
        epoch_dicts.append(epoch_dict)

        if early_stopper.record_epoch(valid_overall_perf):
            break  # Stop if we've gone several epochs without improvement on validation set

    # Find epoch with the best auroc
    auroc_vals = [p['auroc'] for p in epoch_dicts]
    best_auroc_index = np.argmax(auroc_vals)
    print("Epoch (1-index) with best AUROC: {}".format(best_auroc_index + 1))
    print("AUROC: {}".format(epoch_dicts[best_auroc_index]['auroc']))
    print("AUPRC: {}".format(epoch_dicts[best_auroc_index]['auprc']))

    return epoch_dicts

def pytorch_eval(net, eval_data, preds_index=0, data_loader_kwargs=DATA_LOADER_PARAMS, device=DEVICE):
    """Takes a model and a Dataset and evaluates"""
    assert isinstance(eval_data, torch.utils.data.Dataset)
    if data_loader_kwargs is not None:
        data_loader = torch.utils.data.DataLoader(eval_data, **data_loader_kwargs)
    else:
        data_loader = eval_data

    net.to(device)
    net.eval()
    truths, preds = [], []
    for batch, labels in data_loader:
        truths.extend(labels.numpy())
        batch, labels = batch.to(device), labels.to(device)
        outputs= net(batch)
        if isinstance(outputs, tuple):
            outputs = outputs[preds_index]  # If mulitiple outputs assume the first is the target for loss
        preds.extend(outputs.detach().cpu().numpy())
    truths = np.round(np.vstack(truths)).astype(int)
    preds = np.array(preds).astype(float)

    assert np.alltrue(preds <= 1.0) and np.alltrue(preds >= 0.0), "Found predicted probs outside of [0, 1]"
    assert truths.shape == preds.shape, f"Got mismatched shapes: {truths.shape} {preds.shape}"
    is_multitask = len(truths.shape) > 1 and truths.shape[1] > 1
    overall_perf = generate_model_perf(truths, preds, multiclass=is_multitask)
    per_class_perf = generate_multiclass_perf(truths, preds, num_classes=truths.shape[1]) if is_multitask else None

    net.train()
    return overall_perf, per_class_perf, truths, preds

def state_dict_to_cpu(d):
    """Transfer the state dict to CPU. Avoids lingering GPU memory buildup."""
    retval = collections.OrderedDict()
    for k, v in d.items():
        retval[k] = v.cpu()
    return retval

def list_preds_to_array_preds(list_preds):
    """
    Given a list of preds, with each item in the list corresponding for predictions for a class return
    as a matrix of preds. This assumes that the preds are for a binary classification task at each class.
    """
    assert isinstance(list_preds, list), f"Unrecognized input type: {type(list_preds)}"
    pos_preds = []
    for i in range(len(list_preds)):
        pos_preds.append(np.atleast_2d(list_preds[i][:, 1]).T)
    retval = np.hstack(pos_preds)
    assert retval.shape == (list_preds[0].shape[0], len(list_preds))  # num_examples x num_classes
    return retval

def youden_threshold(preds, truth):
    """Use Youden's J-score to set thresholds for a SINGLE output"""
    # https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    preds = np.array(preds)
    truth = np.array(truth)
    assert len(preds.shape) == len(truth.shape) == 1
    assert preds.shape == truth.shape
    fpr, tpr, thresh = metrics.roc_curve(
        truth.flatten(),
        preds.flatten(),
        drop_intermediate=False,
    )
    optimal_idx = np.argmax(tpr - fpr)
    cutoff = thresh[optimal_idx]
    assert 0.0 < cutoff < 1.0, f"Unexpected cutoff value: {cutoff}"
    return cutoff

if __name__ == "__main__":
    print(generate_model_perf([1, 0], [0.8, 0.2]))


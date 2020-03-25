"""
Utilities for common plotting operations
"""

import sys
import os
import collections

import numpy as np
import pandas as pd
import sklearn.metrics as metrics

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def plot_auroc(auroc_dict, title="Receiver operating characteristic", lw=2, cmap="Set2", bg_color='white', grid=True, fname=None, dpi=300, linestyle_dict={}):
    """Plots the auroc curves given in the dictionary, which maps from label to ModelPerf object"""
    fig, ax = plt.subplots(figsize=(8*0.9, 6*0.9), dpi=dpi)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    # Set the axes to be black
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (label, perf) in enumerate(auroc_dict.items()):
        fpr, tpr, _ = perf.auroc_curve
        color = cm.get_cmap(cmap)(i)
        label_expanded = label
        while len(label_expanded) < max([len(x) for x in auroc_dict.keys()]):
            label_expanded += ' '  # Ensure all labels have same length for a cleaner plot
        lstyle = linestyle_dict[label] if label in linestyle_dict else "-"
        ax.plot(fpr, tpr, lw=lw, label=label_expanded + f" AUC={perf.auroc:.2f}", c=color, alpha=0.8, linestyle=lstyle)
    l = ax.legend(loc="lower right")
    plt.setp(l.texts, family='monospace')
    if bg_color is not None:
        ax.set_facecolor(bg_color)
        frame = l.get_frame()
        frame.set_color(bg_color)
    if grid:
        ax.grid(True, color='black' if bg_color is not None else 'grey', alpha=0.1)
    if fname:
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    else:
        fig.show()

def plot_auprc(auprc_dict, title="Precision-recall curve", lw=2, cmap="Set2", bg_color='white', grid=True, fname=None, dpi=300):
    """Plot the precision recall or AUPRC curve"""
    fig, ax = plt.subplots(figsize=(8*0.9, 6*0.9), dpi=dpi)
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="Recall",
        ylabel="Precision",
        title=title,
    )
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (label, perf) in enumerate(auprc_dict.items()):
        precision, recall, _ = perf.auprc_curve
        color = cm.get_cmap(cmap)(i)
        label_expanded = label
        while len(label_expanded) < max([len(x) for x in auprc_dict.keys()]):
            label_expanded += " "  # Ensure all labels have the same length
        ax.step(recall, precision, color=color, label=label_expanded + f" AUC={perf.auprc:.2f}")
    l = ax.legend(loc='upper right')
    plt.setp(l.texts, family='monospace')
    if bg_color is not None:
        ax.set_facecolor(bg_color)
        frame = l.get_frame()
        frame.set_color(bg_color)
    if grid:
        ax.grid(True, color='black' if bg_color is None else 'grey', alpha=0.1)
    if fname:
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    else:
        fig.show()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def triangular_heatmap(mat, ax, title="", logscale=True):
    """Plot a triangular heatmap on the given axes"""
    def check_symmetric(a, rtol=1e-5, atol=1e-8):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)
    assert check_symmetric(mat), "Matrix must be symmetric for a triangular heatmap"

    mask_matrix = np.zeros_like(mat)
    mask_matrix[np.triu_indices_from(mat, k=1)] = True
    if logscale:
        mat = np.log1p(mat)
    sns.heatmap(
        mat,
        ax=ax,
        cbar=False,
        annot=True,
        mask=mask_matrix,
    )
    ax.set(
        facecolor='white',
        title=title,
    )


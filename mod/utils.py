import numpy as np
import nibabel as nib
import sys
import os


def dice(vol1, vol2, labels=None, nargout=1):
    """
     Code comes from:
     Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MICCAI 2018.
    """

    '''
    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)

def load_nii_by_name(vol_name):
    X = nib.load(vol_name).get_data()
    X = np.reshape(X, X.shape + (1,))
    return X

def load_nii_by_name_seg(vol_name, seg_name):
    X = nib.load(vol_name).get_data()
    X = np.reshape(X, X.shape + (1,))
    return_vals = [X]

    X_seg = nib.load(seg_name).get_data()
    return_vals.append(X_seg)

    return tuple(return_vals)

def show_progress(epoch, batch, batch_total, **kwargs):
    message = f'\r{epoch} epoch: [{batch}/{batch_total}'
    for key, item in kwargs.items():
        message += f', {key}: {item}'
    sys.stdout.write(message+']')
    print(message+']')
    if not os.path.exists('model/pro-model'):
        os.mkdir('model/pro-model')
    with open('model/pro-model/loss.txt', 'a') as f:
        f.write(message+']'+ '\n')
    sys.stdout.flush()

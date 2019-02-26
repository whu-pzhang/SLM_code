import numpy as np
import pandas as pd


def calc_entropy(x):
    """Compute entropy of dataset x

    Parameters
    ----------
    x : pandas.DataFrame
        training dataset with "label"

    Returns
    -------
    out: float
        Entropy of the input dataset
    """
    prob = x["label"].value_counts(normalize=True).values
    return np.sum(-prob*np.log2(prob))


def cond_entropy(x):
    """Compute condition entropy of dataset x

    Parameters
    ----------
    x : pandas.DataFrame

    Returns
    -------
    out : dict

    """
    feature_names = x.drop("label", axis=1).columns
    cond_entropy = {}

    for name in feature_names:
        # the classes under this feature
        sub_features = set(x[name])
        prob = x[name].value_counts(normalize=True)
        temp = 0
        for f in sub_features:
            temp += prob[f] * calc_entropy(x[x[name] == f])
        cond_entropy[name] = temp

    return cond_entropy


def info_gain(x):
    """Compute the information gain of dataset x

    Parameters
    ----------
    x : pandas.DataFrame

    Returns:
    --------
    out : dict

    """
    res = {}
    ent = calc_entropy(x)
    for key, value in cond_entropy(x).items():
        res[key] = ent - value
    return res

import pandas as pd
import numpy as np


def ccdf(s):
    """
    Parameters:
        `s`, series, the values of s should be variable to be handled
    Return:
        a new series `s`, index of s will be X axis (number), value of s
        will be Y axis (probability)
    """
    s = s.copy()
    s = s.sort_values(ascending=True, inplace=False)
    s.reset_index(drop=True, inplace=True)
    n = len(s)
    s.drop_duplicates(keep='first', inplace=True)
    X = s.values
    Y = [n - i for i in s.index]

    return pd.Series(data=Y, index=X) / n


def sum_cdf(s):
    s = s.copy()
    s = s.value_counts()
    s = s.sort_index(ascending=True)
    cumulative = []
    for i in range(len(s)):
        s0 = s.iloc[:i + 1]
        cumulative.append(np.inner(s0.index, s0.values))
    s = pd.Series(cumulative, index=s.index)
    return s / s.max()


def sum_ccdf(s):
    """
    Parameters:
        `s`, series, the values of s should be variable to be handled
    Return:
        a news series `s`, index of s will be X axis (number), values
        will be Y axis (sum(X>=x))
    """
    s = s.copy()
    s = s.value_counts()
    s = s.sort_index(ascending=True)
    cumulative = []
    for i in range(len(s)):
        s1 = s.iloc[i:]
        cumulative.append(np.inner(s1.index, s1.values))
    return pd.Series(cumulative, index=s.index)

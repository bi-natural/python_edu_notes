import numpy as np
from pandas import Series

def get_colormap(y, colors):
    if not hasattr(y, 'replace'):
        y = Series(y)
    return y.replace(dict(zip(np.unique(y), colors)))
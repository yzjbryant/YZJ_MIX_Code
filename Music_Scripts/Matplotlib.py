import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib as mpl
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

large=22; med=16; small=12
params={'axes.titlesize': large,
        'legend.fontsize': med,
        'figure.figsize': (16,10),
        'axes.lablesize': med,
        'axes.titlesize': med,
        'xtick.lablesize': med,
        'ytick.lablesize': med,
        'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

print(mpl.__version__)
print(sns.__version__)

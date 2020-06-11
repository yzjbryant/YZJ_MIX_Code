import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib3 import urlopen

pop2014=pd.read_csv(urlopen('http://www.census.gov/popest/data/countries/totals/2014/files/CO-EST2014-alldata.csv'))

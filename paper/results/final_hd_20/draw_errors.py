import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pandas import read_csv
import numpy as np

## https://matplotlib.org/api/markers_api.html


## Paras

x_label = "Epoch"
#y_label = "CPU Usage"
y_label = "MAE"
title = 'Traffic EU'
#title = 'Multivariate Neural Network'

read_filepath = "WC-Error-sl_5.csv"
write_filepath = "WC-Error-sl_5.pdf"

# CPU:  290, 750, 850, 1000, 1300  (best images)
# RAM:  290, 780, 850, 1000, 1300  (best images)
point_number = 100
point_start = 0

colnames = ['GA', "PSO", "TWO", "ABFO", "DE"]
results_df = read_csv(read_filepath, header=None,names=colnames, index_col=False, engine='python')

ga = results_df['GA'].values
pso = results_df['PSO'].values
two = results_df['TWO'].values
abfo = results_df['ABFO'].values
de = results_df['DE'].values

x = np.arange(point_number)

plt.plot(x, ga[point_start:point_start + point_number],  label='GA')
plt.plot(x, pso[point_start:point_start + point_number],    label='PSO')
plt.plot(x, two[point_start:point_start + point_number],   label='TWO')
plt.plot(x, abfo[point_start:point_start + point_number],  label='ABFO')
plt.plot(x, de[point_start:point_start + point_number],  label='DE')

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.legend()
plt.savefig(write_filepath)
plt.show()

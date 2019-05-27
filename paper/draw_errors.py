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

read_filepath = "images/error_eu_k2.csv"
write_filepath = "images/error_eu_k2.pdf"

# CPU:  290, 750, 850, 1000, 1300  (best images)
# RAM:  290, 780, 850, 1000, 1300  (best images)
point_number = 600
point_start = 0

colnames = ['MLNN', "GA-MLNN", "CRO-MLNN", "RNN", "LSTM", "OCRO-MLNN"]
results_df = read_csv(read_filepath, header=None,names=colnames, index_col=False, engine='python')

mlnn = results_df['MLNN'].values
ga_mlnn = results_df['GA-MLNN'].values
cro_mlnn = results_df['CRO-MLNN'].values
rnn = results_df['RNN'].values
lstm = results_df['LSTM'].values
ocro_mlnn = results_df['OCRO-MLNN'].values

x = np.arange(point_number)

plt.plot(x, mlnn[point_start:point_start + point_number],  marker='o', linestyle=':',label='MLNN')
plt.plot(x, ga_mlnn[point_start:point_start + point_number],  marker='s', linestyle=':', label='GA-MLNN')
plt.plot(x, cro_mlnn[point_start:point_start + point_number],  marker='*', linestyle=':', label='CRO-MLNN')
plt.plot(x, rnn[point_start:point_start + point_number],  marker=lines.CARETDOWN, linestyle=':', label='RNN')
plt.plot(x, lstm[point_start:point_start + point_number],  marker='x', linestyle=':', label='LSTM')
plt.plot(x, ocro_mlnn[point_start:point_start + point_number],  marker=4, label='OCRO-MLNN')

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.legend()
plt.savefig(write_filepath)
plt.show()

import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pandas import read_csv
import numpy as np

## https://matplotlib.org/api/markers_api.html


## Paras

x_label = "Epoch"
#y_label = "CPU Usage"
y_label = "MSE"
title = 'Internet Traffic UK (k=5)'
#title = 'Multivariate Neural Network'

read_filepath = "error_uk_SL5.csv"
write_filepath = "img/error_uk_SL5.pdf"

# CPU:  290, 750, 850, 1000, 1300  (best images)
# RAM:  290, 780, 850, 1000, 1300  (best images)
point_number = 100
point_start = 0

t1 = "MLNN"  # GA
t2 = "PSO-ELM"     # PSO
t3 = "TWO-ELM"          # TWO
t4 = "OTWO-ELM"     # OTWO
t5 = "GA-ELM"   # DE
t6 = "ABFO"
colnames = [t1, t2, t3, t4, t5, t6]
results_df = read_csv(read_filepath, header=None,names=colnames, index_col=False, engine='python')

mlnn = results_df[t1].values
ga_mlnn = results_df[t2].values
cro_mlnn = results_df[t3].values
rnn = results_df[t4].values
lstm = results_df[t5].values
ocro_mlnn = results_df[t6].values

x = np.arange(point_number)

# plt.plot(x, mlnn[point_start:point_start + point_number],  marker='o', linestyle=':',label=t1)
# plt.plot(x, ga_mlnn[point_start:point_start + point_number],  marker='s', linestyle=':', label=t2)
# plt.plot(x, cro_mlnn[point_start:point_start + point_number],  marker='*', linestyle=':', label=t3)
# plt.plot(x, rnn[point_start:point_start + point_number],  marker=lines.CARETDOWN, linestyle=':', label=t4)
# plt.plot(x, lstm[point_start:point_start + point_number],  marker='x', linestyle=':', label=t5)
# plt.plot(x, ocro_mlnn[point_start:point_start + point_number],  marker=4, label=t6)


plt.plot(x, mlnn[point_start:point_start + point_number], label=t1)
plt.plot(x, lstm[point_start:point_start + point_number], label=t5)
plt.plot(x, ga_mlnn[point_start:point_start + point_number], label=t2)
plt.plot(x, cro_mlnn[point_start:point_start + point_number], label=t3)
plt.plot(x, rnn[point_start:point_start + point_number], label=t4)


plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.legend()
plt.savefig(write_filepath, bbox_inches="tight")
plt.show()

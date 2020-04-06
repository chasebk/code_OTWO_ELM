from utils.IOUtil import read_dataset_file
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pandas import read_csv
import numpy as np


def plot_file_with_scale_label(filename, xlabel, ylabel, title, pathsave, scaler):
    """
    :param scaler: [ text_scale, math_scale, coordinate_text ]
        Eg: [ "%1.2fK", 1e-3, "x" ], [ "%1.2fM", 1e-6, "y" ], [ "%1.2fB", 1e-9, "both" ]
    :return:
    """
    def scaler_function(x, pos):
        'The two args are the value and tick position' # millions, thousands, billions, ...
        return scaler[0] % (x * scaler[1])
    formatter = FuncFormatter(scaler_function)

    point_number = 100
    point_start = 0

    t1 = "MLNN"  # GA
    t2 = "GA-ELM"  # PSO
    t3 = "PSO-ELM"  # TWO
    t4 = "TWO-ELM"  # OTWO
    t5 = "OTWO-ELM"  # DE
    t6 = "ABFO"

    colnames = [t1, t2, t3, t4, t5, t6]
    results_df = read_csv(filename, header=None, names=colnames, index_col=False, engine='python')

    mlnn = results_df[t1].values
    lstm = results_df[t2].values
    ga_mlnn = results_df[t5].values
    cro_mlnn = results_df[t4].values
    rnn = results_df[t3].values

    ocro_mlnn = results_df[t6].values

    ax = plt.subplot()
    if scaler[2] == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif scaler[2] == "y":
        ax.yaxis.set_major_formatter(formatter)
    elif scaler[2] == "both":
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    else:
        "====== Don't wanna scale anything ====="

    x = np.arange(point_number)
    plt.plot(x, mlnn[point_start:point_start + point_number], color="#1f77b4", label=t1)
    plt.plot(x, lstm[point_start:point_start + point_number], color="#ff7f0e", label=t2)
    plt.plot(x, ga_mlnn[point_start:point_start + point_number], color="#2ca02c", label=t3)
    plt.plot(x, cro_mlnn[point_start:point_start + point_number], color="#9467bd", label=t4)
    plt.plot(x, rnn[point_start:point_start + point_number], color="#d62728", label=t5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    ax.set_title(title)
    plt.savefig(pathsave, bbox_inches="tight")
    plt.show()


read_filepath = "error_uk_SL5.csv"
write_filepath = "img/error_uk_SL5.pdf"
x_label = "Epoch"
y_label = "MSE"
title = 'Internet Traffic UK (k=5)'

scaler = ["%1.2fE-5", 1e+5, "y"]
plot_file_with_scale_label(read_filepath, x_label, y_label, title, write_filepath, scaler)



### Default Color plotly
# D3: #1f77b4
# Plotly: #1f77b4 ; rgb(31, 119, 180)
# D3: #ff7f0e
# Plotly: #ff7f0e ; rgb(255, 127, 14)
# D3: #2ca02c
# Plotly: #2ca02c ; rgb(44, 160, 44)
# D3: #d62728
# Plotly: #d62728 ; rgb(214, 39, 40)
# D3: #9467bd
# Plotly: #9467bd ; rgb(148, 103, 189)
# D3: #8c564b
# Plotly: #8c564b ; rgb(140, 86, 75)
# D3: #e377c2
# Plotly: #e377c2 ; rgb(227, 119, 194)
# D3: #7f7f7f
# Plotly: #7f7f7f ; rgb(127, 127, 127)
# D3: #bcbd22
# Plotly: #bcbd22 ; rgb(188, 189, 34)
# D3: #17becf
# Plotly: #17becf ; rgb(23, 190, 207)


# filenames = ["internet_traffic_eu_5m", "internet_traffic_uk_5m","worldcup98_5m", "google_5m", "google_5m"]
# pathsaves = ["internet_traffic_eu_5m", "internet_traffic_uk_5m","worldcup98_5m", "google_cpu_5m", "google_ram_5m"]
# col_indexs = [ [4], [1], [1], [1], [2] ]
# xlabels = ["Time (5 minutes)", "Time (5 minutes)", "Time (5 minutes)", "Time (5 minutes)", "Time (5 minutes)"]
# ylabels = ["Megabyte", "Bit", "Request", "CPU usage", "Memory usage"]
# titles = ["Internet traffic data from EU cities", "Internet traffic data from UK cities",
#           "Request to server in worldcup season in 1998",
#          "CPU usage from Google trace in 2011", "Memory usage from Google trace in 2011"]
# colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
#
# plot_all_files(filenames, col_indexs, xlabels, ylabels, titles, colours, pathsaves)



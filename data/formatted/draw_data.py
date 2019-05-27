from utils.IOUtil import read_dataset_file
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

def plot_all_files(filenames, col_indexs, xlabels, ylabels, titles, colours, pathsaves):
    for i in range(0, len(filenames)):
        filename = filenames[i] + ".csv"
        pathsave = pathsaves[i] + ".pdf"
        col_index = col_indexs[i]
        color = colours[i]
        xlabel = xlabels[i]
        ylabel = ylabels[i]
        title = titles[i]

        dataset = read_dataset_file(filename, usecols=col_index, header=0)
        ax = plt.subplot()
        plt.plot(dataset, color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_title(title)

        plt.savefig(pathsave, bbox_inches = "tight")
        plt.show()


def plot_file_with_scale_label(filename, col_index, xlabel, ylabel, title, color, pathsave, scaler):
    """
    :param scaler: [ text_scale, math_scale, coordinate_text ]
        Eg: [ "%1.2fK", 1e-3, "x" ], [ "%1.2fM", 1e-6, "y" ], [ "%1.2fB", 1e-9, "both" ]
    :return:
    """
    def scaler_function(x, pos):
        'The two args are the value and tick position' # millions, thousands, billions, ...
        return scaler[0] % (x * scaler[1])
    formatter = FuncFormatter(scaler_function)

    dataset = read_dataset_file(filename, usecols=col_index, header=0)
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
    plt.plot(dataset, color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_title(title)

    plt.savefig(pathsave, bbox_inches="tight")
    plt.show()


# filename = "worldcup98_5m.csv"
# pathsave = "results/worldcup98_5m_test.pdf"
# col_index = [1]
# color = '#ff7f0e'
# xlabel = "Time (5 minutes)"
# ylabel = "Number of requests"
# title = "Request to server in worldcup season in 1998"
# scaler = ["%1.1fK", 1e-3, "both"]
#
# plot_file_with_scale_label(filename, col_index, xlabel, ylabel, title, col_index, pathsave, scaler)




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




def draw_2d(data=None, labels=None, title=None, pathsave=None):
    ax = plt.subplot()
    plt.figure(1)
    plt.plot(data[:, 0:1], data[:, 1:2], 'co', markersize=1.0)
    plt.ylabel(labels[1])
    plt.xlabel(labels[0])
    ax.set_title(title)
    plt.savefig(pathsave, bbox_inches="tight")
    plt.close()
    return None

filename = "google_5m.csv"
pathsave = "results/cpu_ram.pdf"
col_idx = [1, 2]
labels = ["CPU usage", "Memory usage"]
title = "CPU usage and Memory usage from Google trace in 2011"
dataset = read_dataset_file(filename, usecols=col_idx, header=0)
draw_2d(dataset, labels, title, pathsave)



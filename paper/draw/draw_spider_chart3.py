import pandas as pd
from math import pi
import matplotlib.pyplot as plt


colors = ["teal", "red", "green", "blue", "orange", "yellow"]
titles = ["MLNN","ELM", "GA-ELM", "PSO-ELM", "TWO-ELM", "OTWO-ELM"]

#Create a data frame from Messi and Ronaldo's 6 Ultimate Team data points from FIFA 18
MLNN = {'RMSE':16.716,'MAPE':3.0125,'$R^2$':0.9915}
ELM = {'RMSE':16.292,'MAPE':2.9834,'$R^2$':0.9948}
GA_ELM = {'RMSE':15.378,'MAPE':2.9203,'$R^2$':0.9956}
PSO_ELM = {'RMSE':15.394,'MAPE':2.9255,'$R^2$':0.9955}
TWO_ELM = {'RMSE':15.400,'MAPE':2.9171,'$R^2$':0.9955}
OTWO_ELM = {'RMSE':15.363,'MAPE':2.9249,'$R^2$':0.9958}

data = pd.DataFrame([MLNN, ELM, GA_ELM, PSO_ELM, TWO_ELM, OTWO_ELM], index = titles)
print(data)

Attributes =list(data)
AttNo = len(Attributes)

print(Attributes)
print(AttNo)

values = []
angles = []
for i in range(len(colors)):
    value = data.iloc[i].tolist()
    value += values [:1]

    angle = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
    angle += angle[:1]

    values.append(value)
    angles.append(angle)


#Create the chart as before, but with both Ronaldo's and Messi's angles/values
ax = plt.subplot(111, polar=True)

#Add the attribute labels to our axes
plt.xticks(angles[0][:-1], Attributes)

#Plot the line around the outside of the filled area, using the angles and values calculated before
#Fill in the area plotted in the last line
for i in range(len(colors)):
    ax.plot(angles[i], values[i])
    ax.fill(angles[i], values[i], colors[i], 'teal', alpha=0.1)

    #Rather than use a title, individual text points are added
    plt.figtext(0.1, 0.9 - 0.05*i, titles[i], color=colors[i])

#Give the plot a title and show it
ax.set_title("Sliding window k = 2")

plt.savefig("hehe.png", bbox_inches="tight")
plt.show()



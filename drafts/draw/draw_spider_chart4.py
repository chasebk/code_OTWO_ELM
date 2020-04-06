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


values = data.iloc[0].tolist()
values += values [:1]
angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles += angles [:1]

values1 = data.iloc[1].tolist()
values1 += values1 [:1]
angles1 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles1 += angles1 [:1]

values2 = data.iloc[2].tolist()
values2 += values2 [:1]
angles2 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles2 += angles2 [:1]

values3 = data.iloc[3].tolist()
values3 += values3[:1]
angles3 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles3 += angles3[:1]

values4 = data.iloc[4].tolist()
values4 += values4[:1]
angles4 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles4 += angles4[:1]

values5 = data.iloc[5].tolist()
values5 += values5[:1]
angles5 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles5 += angles5[:1]


#Create the chart as before, but with both Ronaldo's and Messi's angles/values
ax = plt.subplot(111, polar=True)

#Add the attribute labels to our axes
plt.xticks(angles[:-1], Attributes)



#Plot the line around the outside of the filled area, using the angles and values calculated before
ax.plot(angles,values)
ax.fill(angles, values, 'teal', alpha=0.1)

ax.plot(angles1,values1)
ax.fill(angles1, values1, 'red', alpha=0.1)

ax.plot(angles2,values2)
ax.fill(angles2, values2, 'green', alpha=0.1)

ax.plot(angles3,values3)
ax.fill(angles3, values3, 'blue', alpha=0.1)

ax.plot(angles4,values4)
ax.fill(angles4, values4, 'orange', alpha=0.1)

ax.plot(angles5,values5)
ax.fill(angles5, values5, 'yellow', alpha=0.1)




#Give the plot a title and show it
ax.set_title("Sliding window k = 2")

plt.savefig("hehe.png", bbox_inches="tight")
plt.show()



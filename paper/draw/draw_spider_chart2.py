import pandas as pd
from math import pi
import matplotlib.pyplot as plt

#Create a data frame from Messi and Ronaldo's 6 Ultimate Team data points from FIFA 18
RMSE = {'MLNN':16.716,'ELM':16.292,'GA-ELM':15.378,'PSO-ELM':15.394,'TWO-ELM':15.400,'OTWO-ELM':15.363}
MAPE = {'MLNN':3.0125,'ELM':2.9834,'GA-ELM':2.9203,'PSO-ELM':2.9255, 'TWO-ELM':2.9171,'OTWO-ELM':2.9249}
R2 = {'MLNN':0.9915,'ELM':0.9948,'GA-ELM':0.9956,'PSO-ELM':0.9955,'TWO-ELM':0.9955,'OTWO-ELM':0.9958}

data = pd.DataFrame([RMSE, MAPE, R2], index = ["RMSE","MAPE", "$R^2$"])
print(data)

Attributes =list(data)
AttNo = len(Attributes)



values = data.iloc[0].tolist()
values += values [:1]

angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles += angles [:1]



#Find the values and angles for Messi - from the table at the top of the page
values2 = data.iloc[1].tolist()
values2 += values2 [:1]

angles2 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles2 += angles2 [:1]


values3 = data.iloc[2].tolist()
values3 += values3 [:1]

angles3 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles3 += angles3 [:1]



#Create the chart as before, but with both Ronaldo's and Messi's angles/values
ax = plt.subplot(111, polar=True)

#Add the attribute labels to our axes
plt.xticks(angles[:-1],Attributes)

#Plot the line around the outside of the filled area, using the angles and values calculated before
ax.plot(angles,values)
#Fill in the area plotted in the last line
ax.fill(angles, values, 'teal', alpha=0.1)

ax.plot(angles2,values2)
ax.fill(angles2, values2, 'red', alpha=0.1)

ax.plot(angles3,values3)
ax.fill(angles3, values3, 'green', alpha=0.1)

#Rather than use a title, individual text points are added
plt.figtext(0.1,0.9,"RMSE",color="teal")
plt.figtext(0.1,0.85,"MAPE", color="red")
plt.figtext(0.1,0.8,"$R^2$",color="green")

#Give the plot a title and show it
ax.set_title("Sliding window k = 2")

plt.savefig("hehe.png", bbox_inches="tight")
plt.show()



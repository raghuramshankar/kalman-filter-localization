import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("cfs_data_fsn17.csv")

fig = plt.figure()
f = fig.add_subplot(111)
# print(data.XX[2:])
x = data['XX'][2:].astype(float)
y = data['YY'][2:].astype(float)
f.plot(x, y)
plt.show()
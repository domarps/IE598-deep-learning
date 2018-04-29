import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
# df=pd.read_csv(sys.argv[1], sep=',',header=None)
# print df
# fig, ax = plt.subplots()
# heatmap = ax.pcolor(df, cmap=plt.cm.Blues, alpha=0.8)
# fig.savefig("confusion.png")
reader=csv.reader(open("part3_results/increaseEpochs_50.csv","rb"),delimiter=',')
conf_matrix = list(reader)
conf_matrix = [[int(j) for j in i] for i in conf_matrix]

conf_matrix = np.asarray(conf_matrix)

class_file = open('classInd.txt','r')
lines = class_file.readlines()
lines = [line.split(' ')[1].strip() for line in lines]

id = 0
# for line in conf_matrix:
# 	np.resize(line,(1,101))
# 	line = np.asarray(line)
# 	print lines[id], lines[np.argmax(line)]
# 	id = id + 1
# class_file.close()
class_names = np.asarray(lines)

plt.matshow(conf_matrix, fignum=100)
plt.matshow(conf_matrix, vmin=0, vmax=50) 
plt.gca().set_aspect('auto')

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix, interpolation='nearest')
fig.colorbar(cax)

#ax.set_xticklabels(['']+lines)
#ax.set_yticklabels(['']+lines)

plt.show()


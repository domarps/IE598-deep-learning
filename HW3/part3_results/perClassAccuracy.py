import csv
import numpy as np
import operator
M = np.genfromtxt('increaseEpochs_50.csv',delimiter=',')
print M
class_file = open('classInd.txt','r')
lines = class_file.readlines()
lines = [line.split(' ')[1].strip() for line in lines]
column_sums = [sum([row[i] for row in M]) for i in range(0,len(M[0]))]
row_sums = [sum(row) for row in M]

accuracy = {}
for i in xrange(len(row_sums)):
	accuracy[lines[i]] = column_sums[i]/row_sums[i]

sorted_accuracy = sorted(accuracy.items(), key=operator.itemgetter(1))

for key, value in sorted_accuracy:
	if(value == 1.0):
		print key
import numpy as np

mat = np.genfromtxt("../games.csv", delimiter=',', dtype='int')
number_of_teams = 107

scores = np.zeros((number_of_teams,number_of_teams))
counts = np.zeros((number_of_teams,number_of_teams))


for row in mat:
	scores[row[1],row[2]] = scores[row[1],row[2]] + row[3]
	scores[row[2],row[1]] = scores[row[2],row[1]] - row[3]
	counts[row[2],row[1]] = counts[row[2],row[1]] + 1
	counts[row[2],row[1]] = counts[row[2],row[1]] + 1
	
for i in range(number_of_teams):
	for j in range(number_of_teams):
		if(counts[i,j] != 0):
			scores[i,j] = scores[i,j] / counts[i,j]
	
np.savetxt("scores.csv", scores, delimiter=",")


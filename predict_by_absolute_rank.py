import numpy as np

number_of_teams = 107

number_of_test_case = 1000

mat = np.genfromtxt("games.csv", delimiter=',', dtype='int')

ranks = np.genfromtxt("./Absolute_Rank/rank.csv", delimiter=',', dtype='int')

np.random.shuffle(mat)

test_y = mat[:number_of_test_case,3]
pred_y = np.zeros(number_of_test_case)

for i in range(number_of_test_case):	
	if(ranks[mat[i,2]] >= 10 * ranks[mat[i,1]]):
		pred_y[i] = -1
	else:
		pred_y[i] = 1
  
  
#Check the number of currect predictions
  
counter = 0

for i in range(number_of_test_case):
    if(test_y[i] * pred_y[i] > 0):
        counter += 1
        
print("The prediction rate is: " + str(counter / number_of_test_case))
	
from scipy.optimize import linear_sum_assignment
import numpy as np


cost_matrix = np.array([[1,4,7],[2,5,6],[6,7,1]])
row_ind, col_ind = linear_sum_assignment(cost_matrix)


print([(x,y) for x,y in zip(row_ind,col_ind)])

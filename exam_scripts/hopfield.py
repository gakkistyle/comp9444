weights = [[0,0,-1,0,0],
            [0,0,0,0,-1],
            [-1,0,0,1,0],
            [0,0,1,0,1],
            [0,-1,0,1,0]]

input_matrix = [1,1 ,-1 ,1 ,1]

output = [] 

for i in range(len(input_matrix)):
    temp = 0
    for j in range(len(input_matrix)):
        temp += weights[j][i] * input_matrix[j]
    if (temp <0):
        output.append(-1)
    elif(temp>0):
        output.append(1)
    else:
        output.append(input_matrix[i])
print(output)

w0 = 1
w1 = 0 
w2 = 0
w3 = 0
lr = 1

targets = [-1,1,1,-1]
data = [[4,3,6],[2,-2,3],[1,0,-3],[4,2,3]]
count = 0

def check_convergence(w0,w1,w2,w3,lr,count):
    for i in range(3):
        temp = w1* data[i][0] + w2 * data[i][1] + w3 * data[i][2]+ w0
        # print("temp is", temp)
        if (temp * targets[i]<0): 
            if (targets[i]<0):
                w0 = w0 - lr 
                w1 = w1 - lr * data[i][0] 
                w2 = w2 - lr * data[i][1] 
                w3 = w3 - lr * data[i][2]
                # print("Target is positive, ",w0,w1,w2)
            else:
                w0 = w0 + lr 
                w1 = w1 + lr * data[i][0] 
                w2 = w2 + lr * data[i][1]
                w3 = w3 + lr * data[i][2]
                # print("Target is negative, ",w0,w1,w2)
            count = 0
        else:
            count += 1
        if (count==4):
            print(w0,w1,w2,w3)
            break
    return (w0,w1,w2,w3,count)

while True:
    w0,w1,w2,w3,count = check_convergence(w0,w1,w2,w3,lr,count)
    # print(w0,w1,w2,w3)
    # print("Now count is ",count)
    if (count==4):
        break


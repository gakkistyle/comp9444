w0 = 0.5
w1 = 1
w2 = -2
lr = 1

targets = [-1,-1,1]
data = [[-1,-1],[2,1],[-2,2]]
count = 0 

def check_convergence(w0,w1,w2,lr,count):
    for i in range(3):
        temp = w1* data[i][0] + w2 * data[i][1] + w0
        print("temp is", temp,"target is ",targets[i])
        
        if (temp * targets[i]<0): 
            print("Old weights are: ",w0,w1,w2)
            if (targets[i]<0):
                w0 = w0 - lr 
                w1 = w1 - lr * data[i][0] 
                w2 = w2 - lr * data[i][1] 
                # print("Target is positive, ",w0,w1,w2)
                
            else:
                w0 = w0 + lr 
                w1 = w1 + lr * data[i][0] 
                w2 = w2 + lr * data[i][1]
                # print("Target is negative, ",w0,w1,w2)
            count = 0
            print("New weights are: ",w0,w1,w2)
        else:
            count += 1
        if (count==3):
            print("The end",w0,w1,w2)
            break
    return (w0,w1,w2,count)

while True:
    # count = 0
    w0,w1,w2,count = check_convergence(w0,w1,w2,lr,count)
    # print(w0,w1,w2)
    # print("Now count is ",count)
    if (count==3):
        break


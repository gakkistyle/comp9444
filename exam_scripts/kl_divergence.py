from math import log
p = [1/2,1/4,1/8,1/16,1/16]
q = [1/8,1/16,1/4,1/2,1/16]

result = 0 

for i in range(len(p)):
    result += p[i] * (log(p[i],2)-log(q[i],2))

print(result)

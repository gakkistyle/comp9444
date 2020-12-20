lst = [1/2,1/4,1/8,1/16,1/16]
import math
result = 0
for i in lst:
    result += -i * math.log(i,2)
print(result)

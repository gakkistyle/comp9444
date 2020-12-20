s1_s1 = 4
s1_s2 = 8
s2_s1 = -1
s2_s2 = 1

import numpy as np


def find_optimal(r):
    v_11_s1 = s1_s1/(1-r)
    v_11_s2 = s2_s1 + r*v_11_s1 

    v_12_s1 = v_11_s1 
    v_12_s2 = s2_s2/(1-r)

    v_21_s1 = (s1_s2+ r*s2_s1)/(1-r**2)
    v_21_s2 = (s2_s1 + r*s1_s2)/(1-r**2)

    v_22_s1 = s1_s2 + (r*s2_s2)/(1-r)
    v_22_s2 = s2_s2/(1-r)

    
    if(v_11_s1 >= v_21_s1):
        if(v_11_s2 >= v_12_s2):
            print("The optimal is V11")

    if (v_12_s2 >= v_11_s2):
        if (v_12_s1 >= v_22_s1):
            print("The optimal is V12")
    
    if (v_21_s1 >= v_11_s1):
        if(v_21_s2 >= v_22_s2):
            print("The optimal is V21")
    
    if (v_22_s2>= v_21_s2):
        if (v_22_s1 >= v_12_s1):
            print("The optimal is V22")
    
# find_optimal(0.5)
for r in np.arange(0.2,0.3,0.01):
    print("Now r is ",r,end=" ")
    find_optimal(r)
    


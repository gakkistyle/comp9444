import math
out_z=[1.3,2.1,3.7]
z_exp_list=[]
for z in out_z:
    z_exp_list.append(math.exp(z))

exp_sum=sum(z_exp_list)

def log_par_der(marked):
    for index in range(len(z_exp_list)):
        if index==marked:
            log_pro=1-z_exp_list[index]/exp_sum
            print('d(log Prob('+str(marked+1)+'))/dz'+str(index)+"="+str(log_pro))
        else:
            log_pro=-z_exp_list[index]/exp_sum
            print('d(log Prob(' + str(marked+1) + '))/dz' + str(index) + "=" + str(log_pro))



for index in range(len(z_exp_list)):
    print("Prob("+str(index)+")="+str(z_exp_list[index]/exp_sum))
#z1===>0
#z2===>1
#z3===>2
log_par_der(2)


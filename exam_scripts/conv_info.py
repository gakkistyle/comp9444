J=42
K=54
P=0
L=3  #the number of channels
M=6 #kernel of filters
N=6
s=4 #stride
F=16 #the number of filters
weights=1 + M *N *L
width=(1+(J+2*P-M)//s)
heigh=(1+(K+2*P-N)//s)
print("weights per neuron: "+str(weights))
print("The width of layer "+str(width))
print("The heigh of layer "+str(heigh))
neurons=width*heigh*F
print("neurons in layer: "+str(neurons))
connections=neurons*weights
print("connections: "+str(connections))
parameters=F*weights
print('independent parameters: '+str(parameters))


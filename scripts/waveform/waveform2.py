import corr
import sys
import pylab as py
import numpy as np

sample_rate=800000000.
ts=1./sample_rate
freq=float(sys.argv[1])
#period = float(sys.argv[2])
period = 1./freq
skip=1
stall=1
#freq=float(sys.argv[1])
#length=512
rand_max=0
tmp=0

x=np.array([])
y=np.array([])

length = np.round(1/(freq*ts))

print length

for i in np.arange(length):
    tmp = tmp + ts 
    a = tmp
    y=np.append(y,np.sin(2*np.pi*a*freq))
 

'''
f=open('sin.txt')

arr=np.array([])

for b in f.readlines():
    arr=np.append(arr,float(b))
'''

y=np.append(y,y)
py.plot(y[:])
py.show()

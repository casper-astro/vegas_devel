import corr
import sys
import pylab as py
import numpy as np

period = 2*np.pi*float(sys.argv[2])
skip=1
stall=1
freq=float(sys.argv[1])
length=512
rand_max=0
tmp=0

x=np.array([])
y=np.array([])

if freq < 1:
    while 512./(freq*stall) > 512 :
	stall = stall + 1

    length = np.round(512./(freq*stall))

else:
    while 512.*(skip+1)/freq <= 512 :
	skip = skip + 1

    length = np.round(512.*skip/freq)

print length
for i in np.arange(length):
    tmp = tmp + period/length
    a = tmp
    y=np.append(y,np.sin(a))
 

'''
f=open('sin.txt')

arr=np.array([])

for b in f.readlines():
    arr=np.append(arr,float(b))
'''

y=np.append(y,y)
py.plot(y[:])
py.show()


#!/opt/vegas/bin/python2.6

import socket
from matplotlib import pyplot as plt
import struct
import numpy as np

udp_ip='10.0.0.145'
udp_port=60000
size=8208 #packet size
f_lo = 200.0
bw=2*1500./128.

# grab spead UDP packet
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((udp_ip, udp_port))
data, addr = sock.recvfrom(size)
sock.close()

a = np.array(struct.unpack('>8208b', data), dtype=np.int8)
a = a[16:]  # strip 16-byte header

X=range(256)
Y=range(256)

rows='4'
columns='2'
figures='8'
f_lo=[100,200,300,400,500,600,700,800]

for i in range(int(figures)):

  realX = a[4*i+0::32]
  imagX = a[4*i+1::32]
  realY = a[4*i+2::32]
  imagY = a[4*i+3::32]    

  f = np.linspace(f_lo[i] - bw/2., f_lo[i] + bw/2., 256)  
  X = np.zeros(256, dtype=np.complex64)
  Y = np.zeros(256, dtype=np.complex64)

  X.real = realX.astype(np.float)
  X.imag = imagX.astype(np.float)
  Y.real = realY.astype(np.float)
  Y.imag = imagY.astype(np.float)

  plt.figure(i+1)

  #first column (realX, imagX, realY, imagY)
  plt.subplot(rows+columns+'1') 
  plt.plot(realX,'o-')
  plt.subplot(rows+columns+'3') 
  plt.plot(realY,'o-')
  plt.subplot(rows+columns+'5') 
  plt.plot(imagX,'o-')
  plt.subplot(rows+columns+'7') 
  plt.plot(imagY,'o-')

  #second column FFT(X), FFT(Y), FFT(X)*FFT(Y), FFT(X)*conj(FFT(Y))
  plt.subplot(rows+columns+'2') 
  plt.plot(f, 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(X, 256)))))
  plt.subplot(rows+columns+'4') 
  plt.plot(f, 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(Y, 256)))))
  plt.subplot(rows+columns+'6') 
  plt.plot(f, 10 * np.log10(np.fft.fftshift(np.fft.fft(X, 256) * np.fft.fft(Y, 256).conjugate()).real))
  plt.subplot(rows+columns+'8') 
  plt.plot(f, 10 * np.log10(np.fft.fftshift(np.fft.fft(X, 256) * np.fft.fft(Y, 256).conjugate()).imag))
  plt.show()


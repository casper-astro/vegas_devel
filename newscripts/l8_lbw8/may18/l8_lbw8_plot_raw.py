#!/opt/vegas/bin/python2.6

import socket
from matplotlib import pyplot as plt
import struct
import numpy as np

udp_ip='10.0.0.145'
udp_port=60000
size=8208 #packet size
f_lo = 104.0
bw=2*1500./128.

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((udp_ip, udp_port))
data, addr = sock.recvfrom(size)
sock.close()

a = np.array(struct.unpack('>8208b', data), dtype=np.int8)
a = a[16:]  # skip 16-byte header
#{0, 2, 1, 3}   # 2 tones
#{0, 2, 3, 1}
#{1, 3, 0, 2}
#{1, 3, 2, 0}
#{2, 0, 1, 3}
#{2, 0, 3, 1}
#{3, 1, 0, 2}
#{3, 1, 2, 0}

realX = a[0::32]
imagX = a[1::32]
realY = a[2::32]
imagY = a[3::32]

realX1 = a[9::32]
imagX1 = a[9::32]
realY1 = a[10::32]
imagY1 = a[11::32]

plt.subplot(421)
plt.plot(realX, '-o')
plt.subplot(423)
plt.plot(imagX, '-o')
plt.subplot(425)
plt.plot(realY, '-o')
plt.subplot(427)
plt.plot(imagY, '-o')

f = np.linspace(f_lo - bw/2., f_lo + bw/2., 256)

X = np.zeros(256, dtype=np.complex64)
X.real = realX.astype(np.float)
X.imag = imagX.astype(np.float)

Y = np.zeros(256, dtype=np.complex64)
Y.real = realY.astype(np.float)
Y.imag = imagY.astype(np.float)

plt.subplot(422)
plt.plot(f, 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(X, 256)))))

plt.subplot(424)
plt.plot(f, 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(Y, 256)))))

plt.subplot(426)
plt.plot(f, 10 * np.log10(np.fft.fftshift(np.fft.fft(X, 256) * np.fft.fft(Y, 256).conjugate()).real))

plt.subplot(428)
plt.plot(f, 10 * np.log10(np.fft.fftshift(np.fft.fft(X, 256) * np.fft.fft(Y, 256).conjugate()).imag))

plt.show()


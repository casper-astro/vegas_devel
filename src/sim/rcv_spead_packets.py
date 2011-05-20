import socket, struct


# IP address and port for GUPPI receiver
UDP_IP="127.0.0.1"
UDP_PORT=50000

	
# Send the packet
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind( (UDP_IP, UDP_PORT) )


while True:
	(data, addr) = sock.recvfrom(2048)
	print '# ', struct.unpack('> Q xHxL xHxL xHxL xHxL xHxL xHxL xHxL xHxL xHxL xHxL 16I', data)


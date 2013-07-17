import struct, socket, time


#### Define global constants ####

NSAMPS                  = 2048

# IP address and port for GUPPI receiver
UDP_IP			        = "127.0.0.1"
UDP_PORT		        = 60000

#### Function definitions ####

# Transmit a SPEAD heap (may require many packets to transmit)
def send_spead_heap(time_cntr_id, time_cntr_id_p1, payload):

    packet = ""

    # Write the packet header always
    packet += struct.pack('!xxxxL', time_cntr_id)
    packet += struct.pack('!xxxxL', time_cntr_id_p1)

    # Now write the spectrum to the packet
    packet += struct.pack('> %dB' % (NSAMPS*4), *(payload))

    # Send the packet
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(packet, (UDP_IP, UDP_PORT))


#### Main function ###


# Generate the sampled waveform to transmit
waveform_list = [[w]*4 for w in range(256 if NSAMPS>256 else NSAMPS)]
waveform_list2 = []

for w in waveform_list:
    waveform_list2.extend(w)
    
waveform = []

if NSAMPS > 256:
    for w in [waveform_list2]*(NSAMPS/256):
      waveform.extend(w)
else:
    waveform = waveform_list2;
    

# Send heaps
for i in range(1, 200000):
    time_cntr_id = (i * 2048) + 9
    time_cntr_id_p1 = time_cntr_id + 1

    # Send the heap
    send_spead_heap(time_cntr_id, time_cntr_id_p1, waveform)

    time.sleep(0.001)  # 0.002 works


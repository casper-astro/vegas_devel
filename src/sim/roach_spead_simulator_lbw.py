import struct, socket, time


#### Define global constants ####

NSAMPS                  = 2048

# IP address and port for GUPPI receiver
UDP_IP			        = "127.0.0.1"
UDP_PORT		        = 60000

# Item Identifiers for the SPEAD packet
# Note: these are constants and should not be modified later
IMMEDIATE		        = 0x800000
POINTER			        = 0x000000

heap_cntr_id 		    = 0x000001 + IMMEDIATE
heap_size_id 		    = 0x000002 + IMMEDIATE
heap_offset_id 		    = 0x000003 + IMMEDIATE
packet_payload_len_id	= 0x000004 + IMMEDIATE

time_cntr_id 		    = 0x000020 + IMMEDIATE
mode_id			        = 0x000021 + IMMEDIATE
status_bits_id		    = 0x000022 + IMMEDIATE
payload_data_off_id	    = 0x000023 + POINTER


#### Function definitions ####

# Transmit a SPEAD heap (may require many packets to transmit)
def send_spead_heap(header, heap):

    packet = ""

    # Compute SPEAD header fields
    header[heap_offset_id] = 0
    header[packet_payload_len_id] = NSAMPS * 4

    # Write the packet header always
    packet += struct.pack('> 2I xHxL xHxL xHxL xHxL',
	            header['header_upr'],
	            header['header_lwr'],
	            heap_cntr_id, header[heap_cntr_id],
	            heap_size_id, header[heap_size_id], 
	            heap_offset_id, header[heap_offset_id], 
	            packet_payload_len_id, header[packet_payload_len_id]) 

    packet += struct.pack('> xHxL xHxL xHxL xHxL',
	        time_cntr_id , heap[time_cntr_id],
	        mode_id, heap[mode_id], 
	        status_bits_id, heap[status_bits_id],
	        payload_data_off_id, heap[payload_data_off_id])

    # Now write the spectrum to the packet
    packet += struct.pack('> %dB' % (NSAMPS*4), *(heap['payload']))

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
    

# Create the SPEAD packet
# The header is 8 bytes long
# The Item Identifiers (keys) are 3 bytes long
# The values and addresses are 5 bytes long
spead_header = {
	'header_upr'	        : 0x53040305,
    'header_lwr'            : 0x00000008,
	heap_cntr_id		    : 0,
	heap_size_id		    : 4*8 + NSAMPS*4,	    # size of total heap
	heap_offset_id		    : 0,
	packet_payload_len_id	: NSAMPS*4              # length of this pkt's payload
}

spead_heap = {
	time_cntr_id		    : 0,
	mode_id			        : 1,
	status_bits_id		    : 0,
	payload_data_off_id	    : 0,
	'payload'		        : waveform
}

# Send 10 heaps
for heap_cntr in range(1500, 12000):

    if heap_cntr < 2000:	
        spead_header[heap_cntr_id] = heap_cntr
    else:	
        spead_header[heap_cntr_id] = heap_cntr - 2000
 	
    spead_heap[time_cntr_id]        = (heap_cntr - 1500) * 10

    # Send the heap
    send_spead_heap(spead_header, spead_heap)

    time.sleep(0.0001)  # 0.002 works

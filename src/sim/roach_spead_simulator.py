import struct, socket, time


#### Define global constants ####


# IP address and port for GUPPI receiver
UDP_IP			        = "127.0.0.1"
UDP_PORT		        = 60000

NCHAN                   = 1024

# Item Identifiers for the SPEAD packet
# Note: these are constants and should not be modified later
IMMEDIATE		        = 0x800000
POINTER			        = 0x000000

heap_cntr_id 		    = 0x000001 + IMMEDIATE
heap_size_id 		    = 0x000002 + IMMEDIATE
heap_offset_id 		    = 0x000003 + IMMEDIATE
packet_payload_len_id	= 0x000004 + IMMEDIATE

time_cntr_id 		    = 0x000020 + IMMEDIATE
spectrum_cntr_id	    = 0x000021 + IMMEDIATE
integ_size_id		    = 0x000022 + IMMEDIATE
mode_id			        = 0x000023 + IMMEDIATE
status_bits_id		    = 0x000024 + IMMEDIATE
payload_data_off_id	    = 0x000025 + POINTER


#### Function definitions ####

# Transmit a SPEAD heap (may require many packets to transmit)
def send_spead_heap(header, heap):

    offset = 0

    # Determine number of pkts reqd to send heap
    num_pkts = (NCHAN*4*4)/8192

    for pkt_num in range(num_pkts):
    
        packet = ""

        # Compute SPEAD header fields
        header[heap_offset_id] = offset
        header[packet_payload_len_id] = 8192

        if pkt_num == 0:
            header['header_lwr'] = 0x0000000A;
        else:
            header['header_lwr'] = 0x00000004;

        # Write the packet header always
        packet += struct.pack('> 2I xHxL xHxL xHxL xHxL',
	                header['header_upr'],
	                header['header_lwr'],
	                heap_cntr_id, header[heap_cntr_id],
	                heap_size_id, header[heap_size_id], 
	                heap_offset_id, header[heap_offset_id], 
	                packet_payload_len_id, header[packet_payload_len_id]) 

        # If the first packet, also write the constant fields
        if pkt_num == 0:
            packet += struct.pack('> xHxL xHxL xHxL xHxL xHxL xHxL',
	                time_cntr_id , heap[time_cntr_id],
	                spectrum_cntr_id, heap[spectrum_cntr_id], 
	                integ_size_id, heap[integ_size_id], 
	                mode_id, heap[mode_id], 
	                status_bits_id, heap[status_bits_id],
	                payload_data_off_id, heap[payload_data_off_id])

            offset += 6*8

        # Now write the spectrum to the packet
        packet += struct.pack('> 2048L',
                    *(heap['payload'][pkt_num*2048:(pkt_num+1)*2048]))

        offset += 8192

        # Send the packet
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(packet, (UDP_IP, UDP_PORT))



#### Main function ###


# Generate the sampled waveform to transmit
spectrum_list = [[s]*4 for s in range(NCHAN)]
spectrum = []
for s in spectrum_list:
    spectrum.extend(s)

# Create the SPEAD packet
# The header is 8 bytes long
# The Item Identifiers (keys) are 3 bytes long
# The values and addresses are 5 bytes long
spead_header = {
	'header_upr'	        : 0x53040305,
    'header_lwr'            : 0x0000000A,
	heap_cntr_id		    : 0,
	heap_size_id		    : 6*8 + NCHAN*4*4,	    # size of total heap
	heap_offset_id		    : 0,
	packet_payload_len_id	: 0	                    # length of this pkt's payload
}

spead_heap = {
	time_cntr_id		    : 0,
	spectrum_cntr_id	    : 0,
	integ_size_id		    : 100,
	mode_id			        : 1,
	status_bits_id		    : 0,
	payload_data_off_id	    : 0,
	'payload'		        : spectrum
}

# Send 10 heaps
for heap_cntr in range(1500, 12000):

    if heap_cntr < 2000:	
        spead_header[heap_cntr_id] = heap_cntr
    else:	
        spead_header[heap_cntr_id] = heap_cntr - 2000
 	
    spead_heap[time_cntr_id]        = (heap_cntr - 1500) * 10
    spead_heap[spectrum_cntr_id]    = spead_header[heap_cntr_id]

    # Send the heap
    send_spead_heap(spead_header, spead_heap)

    time.sleep(0.0001)  # 0.002 works

#ifndef _SPEAD_HEAP_H_
#define _SPEAD_HEAP_H_

#define SPEAD_NUM_ITEMS     6

#pragma pack(push)
#pragma pack(1)

struct freq_spead_heap {
    unsigned char time_cntr_addr_mode;
    unsigned short time_cntr_id;
    unsigned char pad0;
    unsigned int time_cntr;
    unsigned char spectrum_cntr_addr_mode;
    unsigned short spectrum_cntr_id;
    unsigned char pad1;
    unsigned int spectrum_cntr;
    unsigned char integ_size_addr_mode;
    unsigned short integ_size_id;
    unsigned char pad2;
    unsigned int integ_size;
    unsigned char mode_addr_mode;
    unsigned short mode_id;
    unsigned char pad3;
    unsigned int mode;
    unsigned char status_bits_addr_mode;
    unsigned short status_bits_id;
    unsigned char pad4;
    unsigned int status_bits;
    unsigned char payload_data_off_addr_mode;
    unsigned short payload_data_off_id;
    unsigned char pad5;
    unsigned int payload_data_off;
};

struct time_spead_heap {
    unsigned char time_cntr_addr_mode;
    unsigned short time_cntr_id;
    unsigned char pad0;
    unsigned int time_cntr;
    unsigned char mode_addr_mode;
    unsigned short mode_id;
    unsigned char pad1;
    unsigned int mode;
    unsigned char status_bits_addr_mode;
    unsigned short status_bits_id;
    unsigned char pad2;
    unsigned int status_bits;
    unsigned char payload_data_off_addr_mode;
    unsigned short payload_data_off_id;
    unsigned char pad3;
    unsigned int payload_data_off;
};

#pragma pack(pop)

#endif

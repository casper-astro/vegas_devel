function [heap_offset_out,eoh] = heap_offset(heap_length,pld_length,num_heap,eof)
persistent state, state=xl_state(0,{xlUnsigned,3,0});

state_reset    = 0;
state_single   = 1; 
state_multx1   = 2;
state_multx2   = 3;

eoh            = false;
pld_left       = 0;
heap_offset_out    = 0;

switch double(state)
    case state_reset
        heap_offset_out = 0;
        eoh             = false;
        pld_left        = heap_length - pld_length;
        state           = state_single;
        if pld_left > 66
           state = state_multx1;
        end
    case state_single
        heap_offset_out = 0;
        eoh             = false;
        if eof == true
           state = state_reset;
           eoh   = true;
        end
    case state_multx1
        heap_offset_out = 0;
        eoh             = false;
        if (eof == true)
            state = state_multx2;
        end
    case state_multx2
        heap_offset_out = pld_length * num_heap;
        eoh             = false;
        if (heap_length - heap_offset_out == 66 && eof == true)
            state = state_reset;
            eoh   = true;
        end
end
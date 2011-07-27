function [heap_offset,eoh] = heap_number(heap_length,pld_length,eof)
persistent state, state=xl_state(0,{xlUnsigned,3,0});
persistent out_spectra, out_spectra=xl_state(0,{xlUnsigned,3,0});

state_reset    = 0;
state_single   = 1; 
state_multiple = 2;

eoh            = false;
pld_left       = 0;
heap_offset    = 0;

switch double(state)
    case state_reset
        heap_offset = 0;
        out_spectra = 0;
        eoh         = false;
        pld_left    = heap_length - pld_length;
        state       = state_single;
        if pld_left > 0
           state = state_multiple;
        end
    case state_single
        heap_offset = 1;
        eof         = false;
        if (eof == true)
           state = state_reset;
           eoh   = true;
        end
    case state_multiple
        heap_offset = out_spectra*pld_length;
        pld_left    = heap_length - heap_offset - pld_length;
        eoh         = false;
        if (eof == true && pld_left == 0)
            state = state_reset;
            eoh   = true;
        elseif (eof == true)
            out_spectra = out_spectra + 1;
        end
end
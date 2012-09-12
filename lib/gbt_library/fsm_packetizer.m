function [sel, pkt_valid, pkt_eof, st_ph, st_pd, st_w, st_d] = fsm_packetizer(header_valid, data_valid, header_done, data_done, reset)

state_packing_header = 0;
state_packing_data = 1;
state_waiting = 2;
state_done = 3;

init = state_done;

persistent state, state = xl_state(init, {xlUnsigned, 2, 0});

sel_header = 0;
sel_data = 1;

switch state
    case state_packing_header
        sel = sel_header;
        pkt_valid = true;
        pkt_eof = false;
        st_ph = true;
        st_pd = false;
        st_w = false;
        st_d = false;
        if reset
            state = state_done;
        elseif header_done && data_valid
            state = state_packing_data;
            pkt_valid = true;
        elseif header_done && ~data_valid
            state = state_waiting;
        end

    case state_packing_data
        sel = sel_data;
        pkt_valid = true;
        pkt_eof = false;
        st_ph = false;
        st_pd = true;
        st_w = false;
        st_d = false;
        if reset
            state = state_done;
        elseif data_done
            pkt_eof = true;
            state = state_done;
        elseif ~data_done && ~data_valid
            state = state_waiting;
            pkt_valid = false;
        end

    case state_waiting
        sel = sel_data;
        pkt_valid = false;
        pkt_eof = false;
        st_ph = false;
        st_pd = false;
        st_w = true;
        st_d = false;
        if reset
            state = state_done;
        elseif ~data_done && data_valid
            state = state_packing_data;
            pkt_valid = true;
        end

    case state_done
        sel = sel_header;
        pkt_valid = false;
        pkt_eof = false;
        st_ph = false;
        st_pd = false;
        st_w = false;
        st_d = true;
        if reset
            state = state_done;
        elseif header_valid
            state = state_packing_header;
            pkt_valid = true;
        end

    otherwise
        sel = sel_header;
        pkt_valid = false;
        pkt_eof = false;
        st_ph = false;
        st_pd = false;
        st_w = false;
        st_d = true;
        state = state_done;
end

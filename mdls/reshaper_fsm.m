function [block_count_rst,chunk_count_rst,chunk_count_en,sel_rst,sel_en,header_valid,data_valid,st] = reshaper_fsm(end_chunk,end_block,rst,valid_in,head_done)

persistent state,
state=xl_state(0,{xlUnsigned,3,0});

state_reset = 0;
state_send_header = 1;
state_send_header_1 = 2;
state_send_header_2 = 3;
state_send_data = 4;
state_done = 5;

st = state;

switch double(state)

    case state_reset
        % Reset all counters, always proceed to state_send_header
        block_count_rst = true;
        chunk_count_rst = true;
        chunk_count_en = false;
        sel_rst = true;
        sel_en = false;
        header_valid = false;
        data_valid = false;
        if rst,
            state = state_reset
        else
            state = state_send_header;
        end
        
    case state_send_header,
        % Send the header. This means we will send the next chunk of data
        % after this, so reset the chunk counter.
        % we increment the select counter and indicate that the select
        % value is valid.
        % Immediately proceed to state_send_header_2
        % This state seems to be identical to state_send_header_1 so one of
        % these can be removed
        block_count_rst = false;
        chunk_count_rst = true;
        chunk_count_en = false;
        sel_rst = false;
        sel_en = true;
        header_valid = true;
        data_valid = false;
        if rst,
            state = state_reset
        else,
            state = state_send_header_2
        end
        
    case state_send_header_1,
        block_count_rst = false;
        chunk_count_rst = true;
        chunk_count_en = false;
        sel_rst = false;
        sel_en = true;
        header_valid = true;
        data_valid = false;
        if rst,
            state = state_reset
        else,
            state = state_send_header_2
        end
        
    case state_send_header_2
        % second phase of sending header. The select counter has been
        % incremented in the first phase, so now we can check if the
        % head_done has occured indicating we are finished sending the
        % header. If so, we proceed to send_data
        block_count_rst = false;
        chunk_count_rst = true;
        chunk_count_en = false;
        sel_rst = false;
        sel_en = false;
        header_valid = false;
        data_valid = false;
        if rst,
            state = state_reset
        elseif head_done,
            state = state_send_data
        else,
            state = state_send_header_1
        end
        
    case state_send_data
        % In each clock in this state, we increment the chunk counter,
        % sending a new word of data. when at the end of a chunk but not at
        % the end of a block, we proceed to send the next header. When we
        % finish with the block, we go to the done state.
        block_count_rst = false;
        chunk_count_rst = false;
        chunk_count_en = true;
        sel_rst = true;
        sel_en = false;
        header_valid = false;
        data_valid = true;
        if rst,
            state = state_reset
        elseif end_chunk & ~end_block,
            state = state_send_header
        elseif end_block & ~end_chunk,
            state = state_done
        else,
            state = state_send_data
        end
        
    case state_done
        % wait here after sending data for next reset signal
        block_count_rst = false;
        chunk_count_rst = false;
        chunk_count_en = false;
        sel_rst = false;
        sel_en = false;
        header_valid = false;
        data_valid = false;
        if rst,
            state = state_reset
        else,
            state = state_done
        end
        
    otherwise
        block_count_rst = false;
        chunk_count_rst = false;
        chunk_count_en = false;
        sel_rst = false;
        sel_en = false;
        header_valid = false;
        data_valid = false;
        state = state_reset
            

end
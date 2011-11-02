function genBiplexRlocs(FFTSize,coeff_bit_width,input_bit_width,inter_stage_bit_width, output_bit_width, shift_arr,max_distro_size_coeff,max_distro_size_delay,register_coeffs,stage_pre_delay,stage_mid_delay,stage_post_delay,mux_latency_arr, bram_latency_fft, bram_latency_unscr,unscr_din_latency, current_block, fileName, append)

% FFTSize = 8;
% coeff_bit_width=  18*ones(1,FFTSize-2);
% input_bit_width = 18;
% output_bit_width = 18;
% 
% inter_stage_bit_width = 18*ones(1,FFTSize-1); %up to 24
% shift_arr = ones(1,FFTSize);
% max_distro_size_coeff = 1;
% max_distro_size_delay = 1;
% register_coeffs = 0;
% 
% stage_pre_delay = 0;
% stage_mid_delay = 0;
% stage_post_delay = 0;


% 
% bram_latency_fft = 2;
% bram_latency_unscr = 2;
% unscr_din_latency = 0;
% pathToBlock = 'path:biplex';
% mux_latency_arr = 0;
% current_block = 'spec_2pol_4kw_6tap_pfb/pfb_fir_real';

hashCode = '_??????????/';
constrs = {''};

fileName = 'temp_file_name.ucf';
append=0;

fabricRangeString = ' RLOC_RANGE = X-12Y-5:X12Y0;';
for(i=2:FFTSize)
    
    instList = {''};
    [~, blockName] = buildAutoplaceHeader(current_block);
    
    rlocGroup = strcat(' auto_biplex_', blockName, '_stage_', num2str(i), ';');
    %DSPs
    if(i==2)
        %/stage_2_73fdb36e2a\/twiddle_stage2_f72893c79d\/negate_dsp48_f1de3
        %9e1b4\/negate_dsp48\/dsp48e_inst
        instName = strcat(buildAutoplaceHeader(current_block), 'stage_2_??????????/twiddle_stage2_??????????/negate_dsp48_??????????/negate_dsp48/dsp48e_inst"');
        constrs{(length(constrs)+1)} = strcat(instName, ' RLOC_GRID = GRID;');
        constrs{(length(constrs)+1)} = strcat(instName, ' RLOC = X0Y0;');
        constrs{(length(constrs)+1)} = strcat(instName, ' U_SET ', rlocGroup);
        
    else
        %spec_4kw_4tap_pfb_testNetlist_XSG_core_config/spec_4kw_4tap_pfb_testNetlist_XSG_core_config/spec_4kw_4tap_pfb_testnetlist_x0\/fft_biplex_4x1_ca231afa5c\/stage_3_06b6df0dbc\/twiddle_cheap_38ab150c97\/dsp_apbw_im_2\/dsp48e_inst
        %spec_4kw_4tap_pfb_testNetlist_XSG_core_config/spec_4kw_4tap_pfb_testNetlist_XSG_core_config/spec_4kw_4tap_pfb_testnetlist_x0\/fft_biplex_4x1_ca231afa5c\/stage_3_06b6df0dbc\/twiddle_cheap_38ab150c97\/dsp_apbw_re_2\/dsp48e_inst
        instName = strcat(buildAutoplaceHeader(current_block), 'stage_', num2str(i), '_??????????/twiddle_cheap_??????????/dsp_apbw_im_2/dsp48e_inst"');
        constrs{(length(constrs)+1)} = strcat(instName, ' RLOC_GRID = GRID;');
        constrs{(length(constrs)+1)} = strcat(instName, ' RLOC = X0Y0;');
        constrs{(length(constrs)+1)} = strcat(instName, ' U_SET ', rlocGroup);
        
        instName = strcat(buildAutoplaceHeader(current_block), 'stage_', num2str(i), '_??????????/twiddle_cheap_??????????/dsp_apbw_re_2/dsp48e_inst"');
        constrs{(length(constrs)+1)} = strcat(instName, ' RLOC_GRID = GRID;');
        constrs{(length(constrs)+1)} = strcat(instName, ' RLOC_RANGE = X-12Y-5:X12Y0;');
        constrs{(length(constrs)+1)} = strcat(instName, ' U_SET ', rlocGroup);
        
        %twiddle delay
        %buildAutoplaceHeader(current_block)stage_5_efda710930/twiddle_cheap_b00d275784/delay*
        instList{(length(instList)+1)} = strcat(buildAutoplaceHeader(current_block), 'stage_', num2str(i), '_??????????/twiddle_cheap_??????????/delay*"');
    end
    
    %Muxes
    %buildAutoplaceHeader(current_block)stage_9_4e046a32c6/mux*
    instList{(length(instList)+1)} = strcat(buildAutoplaceHeader(current_block), 'stage_', num2str(i), '_??????????/mux*"');
    
    %delay_mid
    %buildAutoplaceHeader(current_block)stage_3_e94bc539ff/delay_mid*
    instList{(length(instList)+1)} = strcat(buildAutoplaceHeader(current_block), 'stage_', num2str(i), '_??????????/delay_mid*"');
    
    
    
    %coeffs
    if(i<=3)
        %buildAutoplaceHeader(current_block)biplex_muxsel_gen_f7d416f125/coeff_gen3_ca1a7aa860/delay*
        instList{(length(instList)+1)} = strcat(buildAutoplaceHeader(current_block), 'biplex_muxsel_gen_??????????/coeff_gen', num2str(i), '*"');
    end
    
    %sel delay
    %buildAutoplaceHeader(current_block)biplex_muxsel_gen_f7d416f125/sel_delay8*
    instList{(length(instList)+1)} = strcat(buildAutoplaceHeader(current_block), 'biplex_muxsel_gen_??????????/sel_delay', num2str(i), '*"');
    
    
    %coeff sync early (5+)
    %buildAutoplaceHeader(current_block)biplex_muxsel_gen_f7d416f125/coeff_sync_early_5*
    if(i>=5)
        instList{(length(instList)+1)} = strcat(buildAutoplaceHeader(current_block), 'biplex_muxsel_gen_??????????/coeff_sync_early_', num2str(i), '*"');
        instList{(length(instList)+1)} = strcat(buildAutoplaceHeader(current_block), 'biplex_muxsel_gen_??????????/coeff_sync_late_', num2str(i), '*"');
    end
    
    
    %//double delay fabric
    if((FFTSize-i+1) < max_distro_size_delay)
        %buildAutoplaceHeader(current_block)stage_9_4e046a32c6/delay_ddbf*
        instList{(length(instList)+1)} = strcat(buildAutoplaceHeader(current_block), 'stage_', num2str(i), '_??????????/delay_ddbf*"');
    else
        %double delay bram
        %buildAutoplaceHeader(current_block)stage_4_b51b666a4d/double_delay_bram_fast_1493ebf7bf/dual_port_ram/latency*
        instList{(length(instList)+1)} = strcat(buildAutoplaceHeader(current_block), 'stage_', num2str(i), '_??????????/double_delay_bram_fast*"');
    end
    
    for(k=2:length(instList))
        constrs{(length(constrs)+1)} = strcat(instList{k}, ' U_SET =', rlocGroup);
        constrs{(length(constrs)+1)} = strcat(instList{k}, ' RLOC_GRID = GRID;');
        constrs{(length(constrs)+1)} = strcat(instList{k}, fabricRangeString);
    end
end

writeConstrsToFile(constrs, fileName, append);
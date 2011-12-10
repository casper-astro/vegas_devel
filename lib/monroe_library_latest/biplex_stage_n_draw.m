function biplex_stage_n_draw(FFTSize, FFTStage, input_bit_width, coeff_bit_width, output_bit_width, delays_bram, bram_latency, shift, delay_arr, mux_latency)
% This is a generated function based on subsystem:
%     fft_stage_n_improved/fft_stage_6
% Though there are limitations about the generated script, 
% the main purpose of this utility is to make learning
% Sysgen Script easier.
% 
% To test it, run the following commands from MATLAB console:
% cfg.source = str2func('fft_stage_6');
% cfg.toplevel = 'fft_stage_n_improved/fft_stage_6';
% args = {my_FFTSize, my_FFTStage, my_input_bit_width, my_coeff_bit_width, my_output_bit_width, my_delays_bram, my_bram_latency, my_shift, my_delay_arr};
% xBlock(cfg, args);
% 
% You can edit fft_stage_6.m to debug your script.
% 
% You can also replace the MaskInitialization code with the 
% following commands so the subsystem will be generated 
% according to the values of mask parameters.
% cfg.source = str2func('fft_stage_6');
% cfg.toplevel = gcb;
% args = {FFTSize, FFTStage, input_bit_width, coeff_bit_width, output_bit_width, delays_bram, bram_latency, shift, delay_arr};
% xBlock(cfg, args);
% 
% To configure the xBlock call in debug mode, in which mode,
% autolayout will be performed every time a block is added,
% run the following commands:
% cfg.source = str2func('fft_stage_6');
% cfg.toplevel = gcb;
% cfg.debug = 1;
% args = {FFTSize, FFTStage, input_bit_width, coeff_bit_width, output_bit_width, delays_bram, bram_latency, shift, delay_arr};
% xBlock(cfg, args);
% 
% To make the xBlock smart so it won't re-generate the
% subsystem if neither the arguments nor the scripts are
% changes, use as the following:
% cfg.source = str2func('fft_stage_6');
% cfg.toplevel = gcb;
% cfg.depend = {'fft_stage_6'};
% args = {FFTSize, FFTStage, input_bit_width, coeff_bit_width, output_bit_width, delays_bram, bram_latency, shift, delay_arr};
% xBlock(cfg, args);
% 
% See also xBlock, xInport, xOutport, xSignal, xlsub2script.

% Mask Initialization code

% 
% xBlock;
% FFTSize=10;
% FFTStage=4;
% input_bit_width=18;
% coeff_bit_width=18;
% output_bit_width=18;
% delays_bram=1;
% bram_latency=2;
% shift=1;
% delay_arr= [0,0,0];



%% inports
xlsub2_a = xInport('a');
xlsub2_b = xInport('b');

xlsub2_sync = xInport('sync');

if(FFTStage > 2)
    xlsub2_coeff = xInport('coeff');
end
xlsub2_mux_sel = xInport('mux_sel');

%% outports
oApBW = xOutport('a+bw');
oAmBW= xOutport('a-bw');
xlsub2_sync_out = xOutport('sync_out');

%% diagram

% block: fft_stage_n_improved/fft_stage_6/Mux
xlsub2_ddbf_out2 = xSignal;
xlsub2_delay_start_out1 = xSignal;
xlsub2_Mux_out1 = xSignal;
xlsub2_Mux = xBlock(struct('source', 'Mux', 'name', 'Mux'), ...
                    struct('arith_type', 'Signed  (2''s comp)', ...
                           'n_bits', 8, ...
                           'bin_pt', 2, 'latency', mux_latency), ...
                    {xlsub2_mux_sel, xlsub2_ddbf_out2, xlsub2_delay_start_out1}, ...
                    {xlsub2_Mux_out1});

% block: fft_stage_n_improved/fft_stage_6/Mux1
xlsub2_Mux1_out1 = xSignal;
xlsub2_Mux1 = xBlock(struct('source', 'Mux', 'name', 'Mux1'), ...
                     struct('arith_type', 'Signed  (2''s comp)', ...
                            'n_bits', 8, ...
                            'bin_pt', 2, 'latency', mux_latency), ...
                     {xlsub2_mux_sel, xlsub2_delay_start_out1, xlsub2_ddbf_out2}, ...
                     {xlsub2_Mux1_out1});


% block: fft_stage_n_improved/fft_stage_6/ddbf
xlsub2_delay_start1_out1 = xSignal;
xlsub2_ddbf_out1 = xSignal;


long_delay_len = 2^(FFTSize-FFTStage);

if(delays_bram == 1)
%if(long_delay_len > 16)
    
    xlsub2_ddbf = xBlock(struct('source', 'monroe_library/double_delay_bram_fast', 'name', 'double_delay_bram_fast'), ...
                         struct('delay_len', long_delay_len, ...
                                'bram_latency', bram_latency), ...
                        {xlsub2_Mux1_out1, xlsub2_delay_start1_out1}, ...
                        {xlsub2_ddbf_out1, xlsub2_ddbf_out2});   
else
    
    sDelayTemp = xDelay(xlsub2_Mux1_out1,long_delay_len, 'delay_ddbf_1');
    xlsub2_ddbf_out1.bind(sDelayTemp);
    
    sDelayTemp = xDelay(xlsub2_delay_start1_out1,long_delay_len, 'delay_ddbf_2');
    xlsub2_ddbf_out2.bind(sDelayTemp);
end

delay_end_ambw = delay_arr(3);
if(FFTStage > 2)
    delay_end_apbw = delay_arr(3) +2;
else
    delay_end_apbw = delay_arr(3);
end
sTwid_apbw_out = xSignal;
sTwid_ambw_out = xSignal;
% block: fft_stage_n_improved/fft_stage_6/delay_end
sApBW_temp = xDelay(sTwid_apbw_out,delay_end_apbw, 'delay_end');
oApBW.bind(sApBW_temp);

% block: fft_stage_n_improved/fft_stage_6/delay_end1

sAmBW_temp = xDelay(sTwid_ambw_out,delay_end_ambw, 'delay_end1');
oAmBW.bind(sAmBW_temp );

% block: fft_stage_n_improved/fft_stage_6/delay_mid %%%%%%%%%%%%%%%%%this
% is the A delay!
if((FFTStage == 1) || (FFTStage == 2))
    
    extra_delay_buttCheap = 0;
else
    extra_delay_buttCheap = 1;
end


xlsub2_delay_mid_out1 = xDelay(xlsub2_ddbf_out1, extra_delay_buttCheap+delay_arr(2) , 'delay_mid');


% block: fft_stage_n_improved/fft_stage_6/delay_mid1
xlsub2_delay_mid1_out1 = xDelay(xlsub2_Mux_out1, delay_arr(2) , 'delay_mid1');


% block: fft_stage_n_improved/fft_stage_6/delay_start


sDelayTemp = xDelay(xlsub2_a,delay_arr(1), 'delay_start');
xlsub2_delay_start_out1.bind(sDelayTemp);


% block: fft_stage_n_improved/fft_stage_6/delay_start1
sDelayTemp = xDelay(xlsub2_b,delay_arr(1), 'delay_start1');
xlsub2_delay_start1_out1.bind(sDelayTemp);


% block: fft_stage_n_improved/fft_stage_6/sync_delay_fast
xlsub2_sync_delay_fast_out1 = xSignal;
if(2^(FFTSize - FFTStage) >16)
    bSyncDelay = xBlock('monroe_library/sync_delay_fast', struct('delay_len', 2^(FFTSize-FFTStage)), {xlsub2_sync}, {xlsub2_sync_delay_fast_out1});
    %xlsub2_sync_delay_fast = xBlock('monroe_library/sync_delay_fast',{struct('delay_len',2^(FFTSize-FFTStage))}, {xlsub2_sync}, {xlsub2_sync_delay_fast_out1});
else
      xlsub2_sync_delay_fast = xBlock(struct('source', 'Delay', 'name', 'sync_delay'), ...
                                struct('latency', 2^(FFTSize - FFTStage)), ...
                                {xlsub2_sync}, ...
                                {xlsub2_sync_delay_fast_out1});  
end
% block: fft_stage_n_improved/fft_stage_6/twiddle_cheap

if(FFTStage == 1)
    xlsub2_twiddle_cheap_sub = xBlock(struct('source', 'monroe_library/twiddle_stage1', 'name', 'twiddle_stage1'), ...
            struct('input_bit_width',input_bit_width, 'out_n_bits', output_bit_width, 'downshift_at_end', shift), ...
            {xlsub2_delay_mid_out1, xlsub2_delay_mid1_out1, xlsub2_sync_delay_fast_out1}, ...
            {sTwid_apbw_out , sTwid_ambw_out , xlsub2_sync_out});
end

if(FFTStage ==2)
    
    sTwiddleSel = xDelay(xlsub2_mux_sel, delay_arr(2) + mux_latency, 'delay_stg2_sel');
    
                      
        xlsub2_twiddle_cheap_sub = xBlock(struct('source', 'monroe_library/twiddle_stage2', 'name', 'twiddle_stage2'), ...
            struct('FFTSize', FFTSize, 'input_bit_width',input_bit_width, 'out_n_bits', output_bit_width, 'downshift_at_end', shift), ...
            {xlsub2_delay_mid_out1, xlsub2_delay_mid1_out1, sTwiddleSel, xlsub2_sync_delay_fast_out1}, ...
            {sTwid_apbw_out , sTwid_ambw_out , xlsub2_sync_out});
    
end

if(FFTStage > 2)
    xlsub2_twiddle_cheap_sub = xBlock(struct('source', 'monroe_library/twiddle_cheap', 'name', 'twiddle_cheap'), ...
        struct('a_n_bits',input_bit_width, 'a_bin_pt', input_bit_width-1, 'b_n_bits', input_bit_width, 'b_bin_pt', ...
        input_bit_width-1, 'w_n_bits', coeff_bit_width, 'w_bin_pt', coeff_bit_width-1, 'a_delay', 0, ...
        'apbw_delay', 0, 'out_n_bits', output_bit_width, 'downshift_at_end', shift), ...
        {xlsub2_delay_mid_out1, xlsub2_delay_mid1_out1, xlsub2_coeff, xlsub2_sync_delay_fast_out1}, ...
        {sTwid_apbw_out , sTwid_ambw_out ,  xlsub2_sync_out});
end



end


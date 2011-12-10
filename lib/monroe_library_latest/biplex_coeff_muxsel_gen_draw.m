function biplex_coeff_muxsel_gen_draw(FFTSize, bit_width, register_coeffs, max_distro_size, optional_delay_arr, stage_delay_arr, mux_latency_arr)
%optional_delay_arr: user assigned delays in-between or during each fft stage
%stage_delay_arr: mandatory algorithmic delays for each stage.  as of writing, it is 6 for every stage but the first, which is 3.
%register_coeffs: register the coeffs after pulling them out of the bram. This is managed in the coefficient generators, and is just a passed value
%sel_delay_arr: the incremental ammount of delay we must add to the sel, in addition to all the previous delays.
%coeff_delay_arr: the incremental ammount of delay we must add to the coeff sync pulse, in addition to all the previous delays.

%we will ge generating a sel signal for all stages,
%but only a coeff signal for stages 3 to FFTSize.
% % %  
% xBlock;
% 
% FFTSize = 6;
% optional_delay_arr = [[0,0,0];[0,0,0];[0,0,0];[0,0,0];[0,0,0];[0,0,0]];
% stage_delay_arr = [2,5,5,5,5,5];
% register_coeffs = 0;
%  bit_width = 18*ones(1,FFTSize-2);
% memory_type = 'Distributed memory';
% max_distro_size = 3;
% pathToBlock = 'path:biplex_gen';
% %  

%fft_delay = total latency across entire FFT
fft_delay = sum(sum(optional_delay_arr) + sum(stage_delay_arr)) + 2^FFTSize -1;

%I/O
iSync = xInport('sync');

oCoeff = xOutport('coeff');
oSel = xOutport('sel');
oSync = xOutport('sync_out');


coeff_delay_arr(1) = optional_delay_arr(1,1) + optional_delay_arr(1,2) + mux_latency_arr(1);
sel_delay_arr(1) = optional_delay_arr(1,1);
for(i = 2:FFTSize)
   coeff_delay_arr(i) =  optional_delay_arr(i-1,3) + optional_delay_arr(i,1) + optional_delay_arr(i,2) + mux_latency_arr(i);
   sel_delay_arr(i) =  optional_delay_arr(i-1,2) + optional_delay_arr(i-1,3) + optional_delay_arr(i,1) + mux_latency_arr(i-1);
   if(i ~= 0)
      coeff_delay_arr(i) = coeff_delay_arr(i) + stage_delay_arr(i-1);
      sel_delay_arr(i) = sel_delay_arr(i) + stage_delay_arr(i-1);
   end
end

coeff_delay_arr(3) = coeff_delay_arr(3) + coeff_delay_arr(2) + coeff_delay_arr(1);



%make those coefficients
% sCoeffSync{2} = iSync;
sCoeffSync{2} = xSignal;
bSyncDelay = xBlock('monroe_library/sync_delay_fast', struct('delay_len', 2^(FFTSize-2) + 2^(FFTSize-1) ), {iSync}, {sCoeffSync{2}});
for(stageNum = 3:FFTSize)
    
    blockName = strcat('coeff_sync_early_', num2str(stageNum));
    sDelaySyncTemp = xDelay(sCoeffSync{stageNum-1}, coeff_delay_arr(stageNum), blockName);
    if(2^(FFTSize-stageNum) > 16)
        sCoeffSync{stageNum} = xSignal;
        blockName = strcat('coeff_sync_late_', num2str(stageNum));
        bSyncDelay = xBlock(struct('source','monroe_library/sync_delay_fast','name', blockName), struct('delay_len', 2^(FFTSize-stageNum)), {sDelaySyncTemp}, {sCoeffSync{stageNum}});
    else
        blockName = strcat('coeff_sync_late_', num2str(stageNum));
        sCoeffSync{stageNum} = xDelay(sDelaySyncTemp,  2^(FFTSize-stageNum), blockName);
    end
        
    coeff_complex = biplex_coeff_gen_calc(FFTSize,stageNum);
    coeff = [imag(coeff_complex) real(coeff_complex)];
    
    sCoeff{stageNum-2} = xSignal;
    
    
    %if there are too many coefficients, store everything in Block RAM.
    %because we are so efficient in RAM, and fabric is always at a premium,
    %and since with the dual coeff gens we just use one bram18 (thus the
    %butterfly is DSP-limited), we should basically always use BRAMs.
    if( stageNum > max_distro_size)
        memory_type = 'Block RAM';
    else
        memory_type = 'Distributed memory';
    end
    blockName = strcat('coeff_gen',num2str(stageNum));
    blockTemp = xBlock(struct('source', str2func('coeff_gen_dual_draw'), 'name',blockName), {coeff, memory_type, bit_width(stageNum-2),FFTSize-stageNum, register_coeffs});
    blockTemp.bindPort({sCoeffSync{stageNum}}, {sCoeff{stageNum-2}});
    
    
end

sCramOut = xCram(sCoeff, 'cram');
oCoeff.bind(sCramOut);


%make the counter that will ultimately be used for all the mux selectors.
%by doing it this way, we only need to make one counter.
sCountArr{1} = xSignal;
bCountUp = xBlock('Counter', struct('cnt_type', 'Free Running', 'operation', 'Up', 'start_count',...
    0, 'cnt_by_val', 1, 'arith_type', 'Unsigned', 'n_bits', FFTSize, 'bin_pt', 0, 'load_pin', 'off',...
    'rst', 'on', 'en', 'off', 'period', 1, 'explicit_period', 'on', 'implementation', 'Fabric'), ...
        {iSync}, {sCountArr{1}});
    
%now delay all the bits the appropriate ammount, and spin them off as needed.
%note that several delay elements could be saved by grouping delay
%elements.  The typical stage will only have a latency of 7 or 8, but a
%SRL16 can hold a latency of up to 16.
for(stageNum = 1:FFTSize)
    sSelOut{stageNum}=xSignal;
    sCountArr{stageNum+1}= xSignal;
    
    blockName = strcat('sel_delay', num2str(stageNum));
    sCountArr{stageNum}  = xDelay(sCountArr{stageNum}, sel_delay_arr(stageNum), blockName);
    
    
    %individual bit-slice to be presented to the sel output
    blockName =  strcat('slice_sel_top_',num2str(stageNum));
    sSelOut{stageNum} = xSliceBool(sCountArr{stageNum}, 'upper', 0, blockName);
    
    if(stageNum ~= FFTSize)
        %rest of the bits to be passed on...
        blockName = strcat('slice_sel_bottom_',num2str(stageNum));
        sCountArr{stageNum+1} = xSlice(sCountArr{stageNum}, FFTSize-stageNum, ...
            'lower', 0, blockName);
    end
end

sConcatOut = xConcat(sSelOut, 'concat');
oSel.bind(sConcatOut);

sSyncOut = xDelay(sCoeffSync{FFTSize},  stage_delay_arr(FFTSize) + optional_delay_arr(FFTSize,3), ...
    'delay_sync_final');
oSync.bind(sSyncOut);





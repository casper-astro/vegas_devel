%optional_delay_arr: user assigned delays in-between or during each fft stage
%stage_delay_arr: mandatory algorithmic delays for each stage.  as of writing, it is 6 for every stage but the first, which is 3.
%register_coeffs: register the coeffs after pulling them out of the bram. This is managed in the coefficient generators, and is just a passed value
%sel_delay_arr: the incremental ammount of delay we must add to the sel, in addition to all the previous delays.
%coeff_delay_arr: the incremental ammount of delay we must add to the coeff sync pulse, in addition to all the previous delays.

%we will ge generating a sel signal for all stages,
%but only a coeff signal for stages 3 to FFTSize.

xBlock;

FFTSize = 4;
optional_delay_arr = [[1,2,3];[1,2,3];[1,2,3];[1,2,3]];
stage_delay_arr = [3,6,6,6];
register_coeffs = 0;
bit_width = 18



global drawing_parameters;



fft_delay = sum(sum(optional_delay_arr) + sum(stage_delay_arr)) + 2^FFTSize -1;


iSync = xInport('sync');

oCoeff = xOutport('coeff');
oSel = xOutport('sel');
oSync = xOutport('sync_out');


coeff_delay_arr(1) = optional_delay_arr(1,1) + optional_delay_arr(1,2);
sel_delay_arr(1) = optional_delay_arr(1,1);
for(i = 2:FFTSize)
   coeff_delay_arr(i) =  optional_delay_arr(i-1,3) + optional_delay_arr(i,1) + optional_delay_arr(i,2);
   sel_delay_arr(i) =  optional_delay_arr(i-1,2) + optional_delay_arr(i-1,3) + optional_delay_arr(i,1);
   if(i ~= 0)
      coeff_delay_arr(i) = coeff_delay_arr(i) + stage_delay_arr(i-1);
      sel_delay_arr(i) = sel_delay_arr(i) + stage_delay_arr(i-1);
   end
end

coeff_delay_arr(3) = coeff_delay_arr(3) + coeff_delay_arr(2) + coeff_delay_arr(1);



%make those coefficients
sCoeffSync{2} = iSync;
for(stageNum = 3:FFTSize)
    sCoeffSync{stageNum} = xSignal;
    blockTemp = xBlock('Delay', struct('latency', coeff_delay_arr(stageNum), 'en', 'off'), ...
            {sCoeffSync{stageNum-1}}, {sCoeffSync{stageNum}});
    
    coeff_complex = biplex_coeff_gen_calc(FFTSize,stageNum);
    coeff = [imag(coeff_complex) real(coeff_complex)];
    
    sCoeff{stageNum-2} = xSignal;
    
    blockTemp = xBlock(struct('source','monroe_library/coeff_gen_dual','name',strcat('coeff_gen',num2str(stageNum))),struct('coeff', coeff, 'bit_width',bit_width, 'register_output',register_coeffs));
    blockTemp.bindPort({sCoeffSync{stageNum}}, {sCoeff{stageNum-2}});
    %drawing_parameters.coeff=coeff;
    
    
end

%now pack them all together
drawing_parameters.numInputs = FFTSize-2;
blockTemp = xBlock(struct('source', str2func('cram_draw'), 'name', 'cram'));
blockTemp.bindPort(sCoeff,{oCoeff});


sCountArr{1} = xSignal;
bCountUp = xBlock('Counter', struct('cnt_type', 'Free Running', 'operation', 'Up', 'start_count',...
    0, 'cnt_by_val', 1, 'arith_type', 'Unsigned', 'n_bits', FFTSize-1, 'bin_pt', 0, 'load_pin', 'off',...
    'rst', 'on', 'en', 'off', 'period', 1, 'explicit_period', 'on', 'implementation', 'Fabric'), ...
        {iSync}, {sCountArr{1}});
    
    
for(stageNum = 1:FFTSize)
    sSelOut{stageNum}=xSignal;
    sCountArr{stageNum+1}= xSignal;
    sDelayTemp = xSignal;
    
    blockTemp = xBlock('Delay', struct('latency', sel_delay_arr(stageNum), 'en', 'off'), ...
            {sCountArr{stageNum}}, {sDelayTemp});
    sCountArr{stageNum} = sDelayTemp;
        
    blockTemp = xBlock('Slice', struct('nbits', 1, 'boolean_output','off', 'mode', 'Upper Bit Location + Width', 'base1', 'MSB of Input', 'base0', 'MSB of Input', 'bit1', 0), {sCountArr{stageNum}}, {sSelOut{stageNum}})
    
    if(stageNum ~= FFTSize)
        
        blockTemp = xBlock('Slice', struct('nbits', 1, 'boolean_output','off', 'mode', 'Upper Bit Location + Width', 'base1', 'MSB of Input', 'base0', 'MSB of Input', 'bit1', 0), {sCountArr{stageNum}}, {sCountArr{stageNum+1}})
    end
end


blockTemp = xBlock(struct('source', 'Concat', 'name', 'Concat'), struct('num_inputs', FFTSize), ...
                       sSelOut, ...
                       {oSel});
                   
sDelayTempB{1}=xSignal;                   
sDelayTempB{2}=xSignal;                   
sDelayTempB{3}=xSignal;                   
sDelayTempB{4}=xSignal;                   

 blockTemp = xBlock('Delay', struct('latency', 1, 'en', 'off'), ...
            {iSync}, {sDelayTempB{1}});                  
 blockTemp = xBlock('Delay', struct('latency', 1, 'en', 'off'), ...
            {sDelayTempB{1}}, {sDelayTempB{2}});                  
 blockTemp = xBlock('Delay', struct('latency', 1, 'en', 'off'), ...
            {sDelayTempB{2}}, {sDelayTempB{3}});                  
 blockTemp = xBlock('Delay', struct('latency', 1, 'en', 'off'), ...
            {sDelayTempB{3}}, {sDelayTempB{4}});                  
        
bSyncDelay = xBlock('monroe_library/sync_delay_fast', struct('delay_len', fft_delay-4), {sDelayTempB{4}}, {oSync});
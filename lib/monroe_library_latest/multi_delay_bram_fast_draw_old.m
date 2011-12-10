function multi_delay_bram_fast_draw(numSignals, delay_len, bitWidth, binPt, signalType, bram_latency)

% % 
% xBlock;
% numSignals=3;
% delay_len=1024;
% bitWidth = 18;
% binPt = 17;
% signalType = 'Signed';
%  bram_latency = 2;

%% inports / outputs




for(i= 1:numSignals)
    blockTemp=xInport(strcat('in', num2str(i-1)));
    %iDataIn.(strcat('s',num2str(i))) = blockTemp;
    sInputs{i} = blockTemp;
    
    blockTemp= xOutport(strcat('out', num2str(i-1)));
    sOutputs{i} = blockTemp;
    %oTapOut.(strcat('s',num2str(i))) = blockTemp;
    
    sCramIn{i} = xSignal;
end





if(mod(numSignals*bitWidth,2) ~= 0)
   error('bit width must be divisible by two') 
end

%% diagram

sCramOut = xSignal;

%drawing_parameters.numInputs = numSignals;
blockTemp = xBlock(struct('source', str2func('cram_draw'), 'name', 'cram'),{numSignals});
%blockTemp = xBlock(struct('source', str2func('cram_draw'), 'name', 'cram'));
blockTemp.bindPort(sCramIn,{sCramOut});

for(i= 1:numSignals)
    sCramIn{i}.bind(sInputs{i});
end

sSliceA = xSignal;
sSliceB = xSignal;

blockTemp = xBlock('Slice', struct('nbits', numSignals*bitWidth/2, 'boolean_output','off', 'mode', 'Upper Bit Location + Width', 'base1', 'MSB of Input', 'base0', 'MSB of Input', 'bit1', 0), {sCramOut}, {sSliceA});
blockTemp = xBlock('Slice', struct('nbits', numSignals*bitWidth/2, 'boolean_output','off', 'mode', 'Upper Bit Location + Width', 'base1', 'MSB of Input', 'base0', 'MSB of Input', 'bit1', -1*numSignals*bitWidth/2), {sCramOut}, {sSliceB});



xlsub2_counter_limited_fast_out1 = xSignal('xlsub2_counter_limited_fast_out1');
% block: untitled/double_delay_bram_fast/counter_limited_fast
xlsub2_counter_limited_fast = xBlock(struct('source', 'monroe_library/counter_limited_fast', 'name', 'counter_limited_fast'), ...
                                     struct('bit_width', ceil(log2(delay_len-bram_latency)), ...
                                            'count_to', delay_len-bram_latency-1), ...
                                     {}, ...
                                     {xlsub2_counter_limited_fast_out1});

                                 
xlsub2_counter_limited_fast2_out1 = xSignal('xlsub2_counter_limited_fast2_out1');
% block: untitled/double_delay_bram_fast/counter_limited_fast
xlsub2_counter_limited_fast2 = xBlock(struct('source', 'monroe_library/counter_limited_fast', 'name', 'counter_limited_fast2'), ...
                                     struct('bit_width', ceil(log2(delay_len-bram_latency)), ...
                                            'count_to', delay_len-bram_latency-1), ...
                                     {}, ...
                                     {xlsub2_counter_limited_fast2_out1});

                                 
                                 
% block: untitled/double_delay_bram_fast/Concat
% xlsub2_Constant1_out1 = xSignal;
% xlsub2_Concat_out1 = xSignal('xlsub2_Concat_out1');
% xlsub2_Concat = xBlock(struct('source', 'Concat', 'name', 'Concat'), ...
%                        [], ...
%                        {xlsub2_Constant1_out1, xlsub2_counter_limited_fast_out1}, ...
%                        {xlsub2_Concat_out1});



% block: untitled/double_delay_bram_fast/Constant1


xlsub2_Constant1_out1 = xSignal('xlsub2_Constant1_out1');
xlsub2_Constant1 = xBlock(struct('source', 'Constant', 'name', 'Constant1'), ...
                          struct('arith_type', 'Boolean', ...
                                 'n_bits', 1, ...
                                 'bin_pt', 0, ...
                                 'explicit_period', 'on'), ...
                          {}, ...
                          {xlsub2_Constant1_out1});
                      
                      
xlsub2_Concat2_out1 = xSignal('xlsub2_Concat2_out1');
xlsub2_Concat2 = xBlock(struct('source', 'Concat', 'name', 'Concat3'), ...
                       [], ...
                       {xlsub2_Constant1_out1, xlsub2_counter_limited_fast2_out1}, ...
                       {xlsub2_Concat2_out1});

                   



% block: untitled/double_delay_bram_fast/Constant2
xlsub2_Constant2_out1 = xSignal('xlsub2_Constant2_out1');
xlsub2_Constant2 = xBlock(struct('source', 'Constant', 'name', 'Constant2'), ...
                          struct('arith_type', 'Boolean', ...
                                 'n_bits', 1, ...
                                 'bin_pt', 0, ...
                                 'explicit_period', 'on'), ...
                          {}, ...
                          {xlsub2_Constant2_out1});
                      
                      
xlsub2_Constant3_out1 = xSignal;
xlsub2_Constant3 = xBlock(struct('source', 'Constant', 'name', 'Constant3'), ...
                          struct('arith_type', 'Boolean', ...
                                 'const', 0, ...
                                 'n_bits', 1, ...
                                 'bin_pt', 0, ...
                                 'explicit_period', 'on'), ...
                          {}, ...
                          {xlsub2_Constant3_out1});
                      
                      
                      
xlsub2_Concat1_out1 = xSignal;
xlsub2_Concat = xBlock(struct('source', 'Concat', 'name', 'Concat1'), ...
                       [], ...
                       {xlsub2_Constant3_out1, xlsub2_counter_limited_fast_out1}, ...
                       {xlsub2_Concat1_out1});
                   
                   
                   

sBramOutA = xSignal;
sBramOutB = xSignal;
% block: untitled/double_delay_bram_fast/Dual Port RAM
xlsub2_Dual_Port_RAM = xBlock(struct('source', 'Dual Port RAM', 'name', 'Dual Port RAM'), ...
                              struct('depth', 2^ceil(log(2*(delay_len-bram_latency))/log(2)), ...
                                     'initVector', 0, ...
                                     'latency', bram_latency, ...
                                     'write_mode_A', 'Read Before Write', ...
                                     'write_mode_B', 'Read Before Write'), ...
                              {xlsub2_Concat2_out1, sSliceA, xlsub2_Constant2_out1, xlsub2_Concat1_out1, sSliceB, xlsub2_Constant2_out1}, ...
                              {sBramOutA, sBramOutB});

                          
sBramConcatOut = xSignal;
 xlsub2_Concat = xBlock(struct('source', 'Concat', 'name', 'Concat2'), ...
                       [], ...
                       {sBramOutA, sBramOutB}, ...
                       {sBramConcatOut});
                          


                                 
                                 


% drawing_parameters.numSignals = numSignals;
% drawing_parameters.bitWidth = bitWidth;
% drawing_parameters.binPt = binPt;
% drawing_parameters.signalType = signalType;

blockTemp = xBlock(struct('source', str2func('uncram_draw'), 'name', 'uncram'),{numSignals,bitWidth,binPt,signalType});
blockTemp.bindPort({sBramConcatOut},sOutputs);

end


function pfb_improved_draw(pfb_size, n_sim_inputs,  n_taps , data_n_bits, coeff_n_bits, output_bit_width, window_fn, bram_latency_coeff, bram_latency_delay, bin_width, end_scale_val, endDelay, cheap_sync, register_delay_counter, multi_delay_bram_share,  autoplace_mode, autoplace_optimize)


%pfb_fir_real drawer

%1. draw sync in/out
%2. draw inputs, outputs.  Store in an array so I can handle dynamic sizing
%3. draw tap pairs. As per pfb design plan

%
% % %
% xBlock;
% 
% 
% data_n_bits=18;
% coeff_n_bits=18;
% pfb_size=14;
% n_taps=3;
% window_fn = 'hamming';
% bram_latency_coeff = 3;
% bram_latency_delay = 3;
% n_sim_inputs = 3;
% bin_width = 1;
% output_bit_width= 18;
% end_scale_val = n_taps/3;
% cheap_sync=0;
% endDelay=1;
% register_delay_counter = 1;
% multi_delay_bram_share = 1;
% % % end comment block



pathToBlock = 'path:pfb';


%check that everything is an ingeger:
if((~isInt([pfb_size, n_sim_inputs, n_taps, data_n_bits,coeff_n_bits, output_bit_width,bram_latency_coeff,bram_latency_delay,end_scale_val,endDelay,cheap_sync,register_delay_counter])))
    strError = 'The following parameters must be integers: pfb_size, n_sim_inputs, n_taps, data_n_bits, data_bin_pt, coeff_n_bits, coeff_bin_pt, output_bit_width, bram_latency_coeff, bram_latency_delay, end_scale_val, endDelay, cheap_sync, register_delay_counter';
    throwError(strError);
    %check that everything is inside the allowed bounds
elseif(pfb_size < 0 || pfb_size > 20)
    strError = strcat('pfb_size must be an integer between 0 and 20 (inclusive); pfb_size= ', num2str(pfb_size));
    throwError(strError);
elseif(n_sim_inputs < 1 || n_sim_inputs > 6)
    strError = strcat('n_sim_inputs must be an integer between 1 and 6 (inclusive); n_sim_inputs= ', num2str(n_sim_inputs));
    throwError(strError);
elseif(data_n_bits < 1 || data_n_bits > 25)
    strError = strcat('data_n_bits must be an integer between 1 and 25 (inclusive); data_n_bits= ', num2str(data_n_bits));
    throwError(strError);
elseif(coeff_n_bits < 1 || coeff_n_bits  > 18)
    strError = strcat('coeff_n_bits  must be an integer between 1 and 18 (inclusive); coeff_n_bits = ', num2str(coeff_n_bits ));
    throwError(strError);
elseif(n_taps < 1 || n_taps  > 32)
    strError = strcat('n_taps  must be an integer between 1 and 16 (inclusive); n_taps = ', num2str(n_taps )); %this limit is somewhat arbitrary.  After 16 taps, the channel shape improvement is nil and the performance will be abysmal.  It's really just to help avoid long draw times.
    throwError(strError);
elseif(bram_latency_coeff ~=2 && bram_latency_coeff  ~= 3)
    strError = strcat('bram_latency_coeff  must be either 2 or 3; bram_latency_coeff = ', num2str(bram_latency_coeff ));
    throwError(strError);
elseif(bram_latency_delay ~=2 && bram_latency_delay  ~= 3)
    strError = strcat('bram_latency_delay  must be either 2 or 3; bram_latency_delay = ', num2str(bram_latency_delay ));
    throwError(strError);
elseif(endDelay ~=0 && endDelay  ~= 1)
    strError = strcat('endDelay  must be either 0 or 1; endDelay = ', num2str(endDelay ));
    throwError(strError);
elseif(cheap_sync ~=0 && cheap_sync  ~= 1)
    strError = strcat('cheap_sync  must be either 0 or 1; cheap_sync = ', num2str(cheap_sync ));
    throwError(strError);
elseif(register_delay_counter ~=0 && register_delay_counter  ~= 1)
    strError = strcat('register_delay_counter  must be either 0 or 1; register_delay_counter = ', num2str(register_delay_counter ));
    throwError(strError);
elseif(bin_width < 0)
    strError = strcat('bin_width must be non-negative; bin_width= ', num2str(bin_width));
    throwError(strError);
elseif(output_bit_width < 1)
    strError = strcat('output_bit_width must be positive; output_bit_width= ', num2str(output_bit_width));
    throwError(strError);
end







data_bin_pt = data_n_bits-1;

coeff_bin_pt= coeff_n_bits -1;

a_bind_index = 0;
b_bind_index = 0;


if (end_scale_val == -1)
    end_scale_val = n_taps/3;
end

vector_len = pfb_size - n_sim_inputs;

if(n_taps < 2)
    error('must have at least two taps.')
end

%generate inputs.  Someday, I'll have to make a sync tree for this

iSync= xInport('sync');
oSyncOut= xOutport('sync_out');

for(i= 1:2^n_sim_inputs)
    blockTemp=xInport(strcat('in', num2str(i-1)));
    iDataIn{i} = blockTemp;
    blockTemp= xOutport(strcat('out', num2str(i-1)));
    oTapOut{i} = blockTemp;
end

bSyncTree = xBlock(struct('source', 'monroe_library/delay_tree_z^-6_0'),{}, {iSync}, ...
    {xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal, ...
    xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal, ...
    xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal, ...
    xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal});
sSyncTree = bSyncTree.getOutSignals();

%procedure for making filter pairs:
%1. make coefficient generator
%2. add first tap
%2.1. add dual bram delay for first tap
%3. add 2nd to n-1th tap (for loop)
%3.1 add each dual bram delay with that tap
%4. add nth tap
%5. add scale/converts
%6. add sync delay


%j = filter pair we are on
%k = tap inside filter we are on.
sDelayIn = {xSignal};
sDelayOut = {xSignal};

sCoeff0SyncOut = xSignal;
for filterPairNum=1:(2^(n_sim_inputs-1))
    
    
    
    %<<<<<<<<<<<<<BEGIN FILTER PAIR LOOP>>>>>>>>>>>>%
    sGenSync=xSignal;
    sGenCoeff=xSignal;
    sGenCoeffRev=xSignal;
    
    
    %coefficient generator
    
    % signalTemp1=sGenSync(filterPairNum);
    % signalTemp2=sGenCoeff(filterPairNum);
    % signalTemp3=sGenCoeffRev(filterPairNum);
    %
    
    % config.source = str2func('pfb_coeff_gen_dual_draw');
    % config.toplevel=gcb;
    % config.debug=1;
    % xBlock(config, {pfb_size,n_sim_inputs,n_taps,input_num,n_bits,window_fn,mem_type,bram_latency,bin_width})
    
    num_sim_inputs = n_sim_inputs;
    
    % block: pfb_fir_real_plan/Subsystem/coeff_gen_00_15
    % blockTemp = xBlock(struct('source', @pfb_coeff_gen_dual_draw, 'name', strcat('coeff_gen_', num2str(filterPairNum-1), '_', num2str(2^num_sim_inputs-filterPairNum+1))), ...
    %                                 struct('pfb_size', pfb_size,'n_sim_inputs',n_sim_inputs, 'n_taps',n_taps,'input_num',filterPairNum, ...
    %                                 'n_bits',coeff_n_bits,'window_fn',window_fn,'mem_type',mem_type,'bram_latency',bram_latency,'bin_width',bin_width), ...
    %                                 {iSync}, ...
    %                                 {sGenSync(filterPairNum), sGenCoeff(filterPairNum), sGenCoeffRev(filterPairNum)});
    
    
    
    
    
    %
    %     drawing_parameters.pfb_size=pfb_size;
    %     drawing_parameters.n_sim_inputs=n_sim_inputs;
    %     drawing_parameters.n_taps=n_taps;
    %     drawing_parameters.input_num=filterPairNum-1;
    %     drawing_parameters.n_bits=coeff_n_bits;
    %     drawing_parameters.window_fn=window_fn;
    %     drawing_parameters.mem_type=mem_type;
    %     drawing_parameters.bram_latency=bram_latency;
    %     drawing_parameters.bin_width=bin_width;
    %
    
    blockName = strcat('coeff_gen_',num2str(filterPairNum-1), '_', num2str(2^n_sim_inputs-filterPairNum));
    blockTemp = xBlock(struct('source', str2func('pfb_coeff_gen_dual_draw'), 'name', blockName), ...
        {pfb_size,n_sim_inputs,n_taps,filterPairNum-1,coeff_n_bits,window_fn, 'Block RAM',bram_latency_coeff,bin_width});
    blockTemp.bindPort({sSyncTree{filterPairNum}},{sGenSync,sGenCoeff,sGenCoeffRev});
    
    
    if(filterPairNum == 1)
        sCoeff0SyncOut = sGenSync;
    end
    %syncTemp = xInport('in1');
    
    
    % blockTemp = xBlock('monroe_library/pfb_coeff_gen_dual',struct('pfb_size', pfb_size,'n_sim_inputs',n_sim_inputs, ...
    %     'n_taps',n_taps,'input_num',filterPairNum,'n_bits',coeff_n_bits,'window_fn',window_fn,'mem_type',mem_type,'bram_latency',bram_latency,'bin_width',bin_width), ...
    %     {iSync}, {sGenSync(filterPairNum), sGenCoeff(filterPairNum), sGenCoeffRev(filterPairNum) })
    bGen(filterPairNum) = blockTemp;
    
    %first tap
    stageNum=1;
    
    
    %apparently, xBlocks hates putting *anything* into arrays.  must use
    %structs or unique names.  cell arrays maybe?
    sTapACoeffOutPrevious = xSignal;
    sTapADataOutPrevious = xSignal;
    
    sTapBCoeffOutPrevious = xSignal;
    sTapBDataOutPrevious = xSignal;
    
    
    blockTemp = xBlock(struct('source','monroe_library/first_tap_improved','name',strcat('filter_in',num2str(filterPairNum-1), '_stage', num2str(stageNum))),struct('data_bin_pt',data_bin_pt, 'coeff_bit_width',coeff_n_bits));
    blockTemp.bindPort({iDataIn{filterPairNum}, sGenCoeff}, {sTapACoeffOutPrevious, sTapADataOutPrevious});
    
    
    blockTemp = xBlock(struct('source','monroe_library/first_tap_improved','name',strcat('filter_in',num2str(2^n_sim_inputs-filterPairNum), '_stage', num2str(stageNum))),struct('data_bin_pt',data_bin_pt, 'coeff_bit_width',coeff_n_bits), ...
        {iDataIn{(2^n_sim_inputs)-filterPairNum+1}, sGenCoeffRev},{sTapBCoeffOutPrevious, sTapBDataOutPrevious});
    
    
    
    %first inter-tap delay
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %     sDataDelayCounter = xSignal
    %
    %     blockTemp = xBlock(struct('source', 'monroe_library/counter_limited_fast','name', strcat('delay_counter_', num2str(filterPairNum-1))), {  'bit_width',ceil(log(((2^vector_len)-bram_latency))/log(2)), 'count_to', 2^vector_len - bram_latency-1}, {}, {sDataDelayCounter})
    %     sDataDelayAPrevious = xSignal;
    %     sDataDelayBPrevious = xSignal;
    
    
    %     drawing_parameters.numSignals=(2*(n_taps-1));
    %     drawing_parameters.delay_len=2^vector_len;
    %     drawing_parameters.bitWidth=data_n_bits;
    %     drawing_parameters.binPt=data_bin_pt;
    %     drawing_parameters.signalType='Signed';
    %     drawing_parameters.bram_latency=bram_latency;
    
    if(multi_delay_bram_share == 0)
        for(kk= 1:(2*(n_taps-1)))
            sDelayIn{kk}=xSignal;
            sDelayOut{kk}=xSignal;
            
        end
        sDelayIn{1} = iDataIn{filterPairNum};
        sDelayIn{2} = iDataIn{2^n_sim_inputs-filterPairNum+1};
        
        blockName = strcat('multi_delay_bram_fast_', num2str(filterPairNum-1));
        blockTemp =  xBlock(struct('source', @multi_delay_bram_fast_draw, 'name', blockName), ...
            {(2*(n_taps-1)),2^vector_len,data_n_bits,data_bin_pt, ...
            'Signed',bram_latency_delay,register_delay_counter}, ...
            sDelayIn, sDelayOut);
    else
        
        if(filterPairNum == 1)
            for(kk= 1:((2*(n_taps-1))* (2^(n_sim_inputs-1))))
                sDelayIn{kk}=xSignal;
                sDelayOut{kk}=xSignal;
                
            end
            
            for(kk = 1:(2^(n_sim_inputs-1)))
                
                offset = (n_taps-1) * (kk-1)*2;
                sDelayIn{1+offset} = iDataIn{kk};
                sDelayIn{2+offset} = iDataIn{2^n_sim_inputs-kk+1};
            end
            blockName = strcat('multi_delay_bram_fast_', num2str(filterPairNum-1));
            bMultiDelay =  xBlock(struct('source', @multi_delay_bram_fast_draw, 'name', blockName), ...
                {(2*(n_taps-1))*(2^(n_sim_inputs-1)),2^vector_len,data_n_bits,data_bin_pt, ...
                'Signed',bram_latency_delay,register_delay_counter}, sDelayIn, sDelayOut);
        end
    end
    %make the other taps
    for(stageNum=2:(n_taps-1))
        %nth tap
        sTapACoeffOut= xSignal;
        sTapADataOut= xSignal;
        
        sTapBCoeffOut = xSignal;
        sTapBDataOut = xSignal;
        
        
        %         sDelayOut{stageNum-1}.bind(sDelayIn{stageNum});
        %         sDelayOut{(stageNum-1)+(n_taps-1)}.bind(sDelayIn{(stageNum)+(n_taps-1)});
        %
        if(multi_delay_bram_share == 0)
            a_bind_index = 2*(stageNum-1) - 1;
            b_bind_index = 2*(stageNum-1);
        else
            offset = (n_taps-1) * (filterPairNum-1)*2;
            a_bind_index = 2*(stageNum-1) - 1 + offset;
            b_bind_index = 2*(stageNum-1) + offset;
            
            
        end
        sDelayOut{a_bind_index}.bind(sDelayIn{a_bind_index+2});
        sDelayOut{b_bind_index}.bind(sDelayIn{b_bind_index+2});
        
        
        
        
        
        
        blockTemp =  xBlock(struct('source','monroe_library/middle_tap_improved','name',strcat('filter_in',num2str(filterPairNum-1), '_stage', num2str(stageNum))) ...
            ,struct('data_bin_pt',data_bin_pt, 'coeff_bit_width',coeff_n_bits, 'stage_num', stageNum), ...
            {sDelayOut{a_bind_index}, sTapACoeffOutPrevious, sTapADataOutPrevious}, {sTapACoeffOut, sTapADataOut});
        %bTapA(filterPairNum,stageNum) = blockTemp;
        
        blockTemp =  xBlock(struct('source','monroe_library/middle_tap_improved','name',strcat('filter_in',num2str(2^n_sim_inputs-filterPairNum), '_stage', num2str(stageNum))) ...
            ,struct('data_bin_pt',data_bin_pt, 'coeff_bit_width',coeff_n_bits, 'stage_num', stageNum), ...
            {sDelayOut{b_bind_index}, sTapBCoeffOutPrevious, sTapBDataOutPrevious}, {sTapBCoeffOut, sTapBDataOut});
        %bTapB(filterPairNum,stageNum) = blockTemp;
        sTapACoeffOutPrevious=sTapACoeffOut;
        sTapBCoeffOutPrevious=sTapBCoeffOut;
        sTapADataOutPrevious=sTapADataOut;
        sTapBDataOutPrevious=sTapBDataOut;
        
        
    end
    
    
    
    %all the taps but the last are now finished.  All the data delays are too.
    
    
    %Let's make that last tap.
    
    
    stageNum= n_taps;
    
    sTapACoeffOut(stageNum) = xSignal;
    sTapADataOut(stageNum) = xSignal;
    
    sTapBCoeffOut(stageNum) = xSignal;
    sTapBDataOut(stageNum) = xSignal;
    
    
    if(multi_delay_bram_share == 0)
        a_bind_index = 2*(stageNum-1) - 1;
        b_bind_index = 2*(stageNum-1);
    else
        offset = (n_taps-1) * (filterPairNum-1)*2;
        a_bind_index = 2*(stageNum-1) - 1 + offset;
        b_bind_index = 2*(stageNum-1) + offset;
        
        
    end
    
    
    
    bLastTapA = xBlock(struct('source','monroe_library/last_tap_improved','name',strcat('filter_in',num2str(filterPairNum-1), '_stage', num2str(stageNum))) ...
        ,struct('data_bin_pt',data_bin_pt,'coeff_bin_pt', coeff_bin_pt, 'stage_num', stageNum));
    %blockTemp.bindPort({sDataDelayAPrevious, sTapACoeffOutPrevious, sTapADataOutPrevious}, {sTapADataOut});
    bLastTapA.bindPort({sDelayOut{a_bind_index}, sTapACoeffOutPrevious, sTapADataOutPrevious}, {xSignal});
    sTapADataOut_ca=bLastTapA.getOutSignals();
    sTapADataOut=sTapADataOut_ca{1};
    %bTapA(filterPairNum,stageNum) = blockTemp;
    
    bLastTapB= xBlock(struct('source','monroe_library/last_tap_improved','name',strcat('filter_in',num2str(2^n_sim_inputs-filterPairNum), '_stage', num2str(stageNum))) ...
        ,struct('data_bin_pt',data_bin_pt,'coeff_bin_pt', coeff_bin_pt, 'stage_num', stageNum));
    bLastTapB.bindPort({sDelayOut{b_bind_index}, sTapBCoeffOutPrevious, sTapBDataOutPrevious}, {xSignal});
    sTapBDataOut_ca=bLastTapB.getOutSignals();
    sTapBDataOut=sTapBDataOut_ca{1};
    
    %bTapB(filterPairNum,stageNum) = blockTemp;
    
    
    
    sScaleA= xSignal;
    sScaleB = xSignal;
    
    blockTemp =  xBlock('Scale', struct('scale_factor', -1* ceil(end_scale_val) )...
        , {sTapADataOut} , {sScaleA});
    %bScaleA(filterPairNum) = blockTemp;
    blockTemp =  xBlock('Scale', struct('scale_factor', -1* ceil(end_scale_val)) ...
        , {sTapBDataOut} , {sScaleB});
    
    
    
    
    blockName = strcat('reinterpret_', num2str(filterPairNum-1));
    %sReinterpretA = xReinterpret(sScaleA, 1, 'signed', 1, data_bin_pt + coeff_bin_pt, blockName);
    sReinterpretA =sScaleA;
    
    blockName = strcat('reinterpret_', num2str(2^n_sim_inputs - (filterPairNum-1)));
    %sReinterpretB = xReinterpret(sScaleB, 1, 'signed', 1, data_bin_pt + coeff_bin_pt, blockName);
    sReinterpretB =sScaleB;
    
    xlsub2_Convert1 = xBlock(struct('source', 'Convert'), ...
        struct('n_bits', output_bit_width, ...
        'bin_pt', output_bit_width-1, ...
        'latency', endDelay), ...
        {sReinterpretA}, ...
        {oTapOut{filterPairNum}});
    
    
    xlsub2_Convert1 = xBlock(struct('source', 'Convert'), ...
        struct('n_bits', output_bit_width, ...
        'bin_pt', output_bit_width-1, ...
        'latency', endDelay), ...
        {sReinterpretB}, ...
        {oTapOut{(2^num_sim_inputs)-filterPairNum+1}});
    
    
    
    
end
%<<<<<<<<<<<<<END FILTER PAIR LOOP>>>>>>>>>>>>%


oSyncOut1 = xOutport('sync_out1');

if( cheap_sync == 1)
    %this line for a cheaper sync delay.  OK if all your hardware elements run
    %with periods at subdivisions of your vector length (they probably do)
    sSyncOut0 = xDelay(iSync, endDelay + n_taps + 2, 'sync_delay_end0');
    sSyncOut1 = xDelay(iSync, endDelay + n_taps + 2, 'sync_delay_end1');
    oSyncOut.bind(sSyncOut0);
    oSyncOut1.bind(sSyncOut1);
else
    %this line for an "honest" sync delay.  More hardware expensive
    bSyncDelay = xBlock('monroe_library/sync_delay_fast', struct('delay_len', endDelay + (n_taps +2) + ((2^vector_len)*(n_taps-1))  -8), {sCoeff0SyncOut}, {oSyncOut});
    bSyncDelay = xBlock('monroe_library/sync_delay_fast', struct('delay_len', endDelay + (n_taps +2) + ((2^vector_len)*(n_taps-1))  -8), {sGenSync}, {oSyncOut1});
end



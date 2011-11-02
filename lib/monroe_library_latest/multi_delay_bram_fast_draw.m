function multi_delay_bram_fast_draw(numSignals, delay_len, bitWidth, binPt, signalType, bram_latency,register_counter)

% % %
% xBlock;
% numSignals=3;
% delay_len=23;
% bitWidth = 18;
% binPt = 17;
% signalType = 'Signed';
% bram_latency = 2;
% register_counter = 1;
% pathToBlock = 'path:multi_delay_bram_fast';
% %% inports / outputs



signalTypeGood = (strcmpi(signalType, 'signed') || ...
    strcmpi(signalType, 'unsigned'));

if(~isInt([numSignals,delay_len, bitWidth, binPt, bram_latency, register_counter]))
    throwError('The following values must be integers:  numSignals, delay_len, bitWidth, binPt, bram_latency,register_counter.');
elseif(numSignals <1)
    strError = strcat('numSignals must be a positive integer;  numSignals = ', num2str(numSignals));
    throwError(strError);
elseif(bitWidth <1)
    strError = strcat('bitWidth must be a positive integer;  bitWidth = ', num2str(bitWidth));
    throwError(strError);
elseif(binPt <1)
    strError = strcat('binPt must be a positive integer;  binPt = ', num2str(binPt));
    throwError(strError);
elseif((bram_latency ~= 2) && (bram_latency ~= 3))
    strError = strcat('bram_latency must be either 2 or 3;  bram_latency = ', num2str(bram_latency));
    throwError(strError);
elseif((register_counter ~= 0) && (register_counter~= 1))
    strError = strcat('register_counter must be either 0 or 1;  register_counter = ', num2str(register_counter));
    throwError(strError);
elseif(binPt > bitWidth)
    strError = strcat('binal point must be >= bit width;  binPt = ', num2str(binPt), '; bitWidth = ', num2str(bitWidth));
    throwError(strError);
elseif(~signalTypeGood)
    throwError('The signal type must be either ''Signed'' or ''Unsigned''');
end



for(i= 1:numSignals)
    blockTemp=xInport(strcat('in', num2str(i-1)));
    %iDataIn.(strcat('s',num2str(i))) = blockTemp;
    sInputs{i} = blockTemp;
    
    blockTemp= xOutport(strcat('out', num2str(i-1)));
    sOutputs{i} = blockTemp;
    %oTapOut.(strcat('s',num2str(i))) = blockTemp;
    
    sCramIn{i} = xSignal;
    sCramIn{i}.bind(sInputs{i});
end





sCramOut = xCram(sCramIn, 'cram');

if(delay_len > 32)
    
    
    %% diagram
    
    
    if(mod(numSignals*bitWidth,2) ~= 0)
        throwError(strcat('bit width must be divisible by two; bitWidth:', num2str(bitWidth)));
    end
    sSliceA = xSignal;
    sSliceB = xSignal;
    
    slice_nbits = numSignals*bitWidth;
    slice_nbits_remaining = slice_nbits;
    %blockTemp = xBlock('Slice', struct('nbits', slice_nbits, 'boolean_output','off', 'mode', 'Upper Bit Location + Width', 'base1', 'MSB of Input', 'base0', 'MSB of Input', 'bit1', 0), {sCramOut}, {sSliceA});
    %blockTemp = xBlock('Slice', struct('nbits', slice_nbits, 'boolean_output','off', 'mode', 'Upper Bit Location + Width', 'base1', 'MSB of Input', 'base0', 'MSB of Input', 'bit1', -1*numSignals*bitWidth/2), {sCramOut}, {sSliceB});
    %the bits are now convienently split in two: lets assign them in blocks of
    %18 each (so targeting BRAM36s when possible)
    
    %%%%%%%%%begin my changes
    i=1;
    while(slice_nbits_remaining >=36)
        sDelayedA{i} = xSignal;
        sDelayedB{i} = xSignal;
        sConcated{i} = xSignal;
        
        blockName =  strcat('sliceA_', num2str(i));
        sSliceBramA{i} = xSlice(sCramOut, 18, 'upper', -36*(i-1) , blockName);
        
        blockName =  strcat('sliceB_', num2str(i));
        sSliceBramB{i} = xSlice(sCramOut, 18, 'upper', -36*(i-1) -18 , blockName);
        
        
        xlsub2_ddbf = xBlock(struct('source', 'monroe_library/double_delay_bram_fast', 'name', strcat('double_delay_bram_fast',num2str(i))), ...
            struct('delay_len', delay_len, ...
            'bram_latency', bram_latency, ...
            'register_counter', register_counter), ...
            {sSliceBramA{i}, sSliceBramB{i}}, ...
            {sDelayedA{i}, sDelayedB{i}});
        
        
        blockName = strcat('Concat2_',num2str(i));
        sConcated{i} = xConcat({sDelayedA{i}, sDelayedB{i}}, blockName);
        
        slice_nbits_remaining = slice_nbits_remaining - 36;
        i=i+1;
    end
    
    sSliceBramA{i} = xSignal;
    sSliceBramB{i} = xSignal;
    sDelayedA{i} = xSignal;
    sDelayedB{i} = xSignal;
    if(slice_nbits_remaining > 18)   %if there are more than 9 bits remaining, it is probably NOT cheaper to switch to a single-port ram.  Lets stay with a dual-port.
        
        sConcated{i} = xSignal;
        blockName =  strcat('sliceA_', num2str(i));
        sSliceBramA{i} = xSlice(sCramOut, slice_nbits_remaining/2, 'lower', slice_nbits_remaining/2 , blockName);
        
        blockName =  strcat('sliceB_', num2str(i));
        sSliceBramB{i} = xSlice(sCramOut, slice_nbits_remaining/2, 'lower', 0 , blockName);
        
        
        xlsub2_ddbf = xBlock(struct('source', 'monroe_library/double_delay_bram_fast', 'name', strcat('double_delay_bram_fast',num2str(i))), ...
            struct('delay_len', delay_len, ...
            'register_counter', register_counter, ...
            'bram_latency', bram_latency), ...
            {sSliceBramA{i}, sSliceBramB{i}}, ...
            {sDelayedA{i}, sDelayedB{i}});
        
        
        
        blockName = strcat('Concat2_',num2str(i));
        sConcated{i} = xConcat({sDelayedA{i}, sDelayedB{i}}, blockName);
        
        
    elseif(slice_nbits_remaining >0) %now we have 1-9 bits per port, inclusive.  In the event that we have a vector length of 1024 or more, a single-port bram is cheaper.  Lets use that instead.
        
        sConcated{i} = xSignal;
        sSliceLast = xSlice(sCramOut, slice_nbits_remaining, 'lower', 0 , 'slice_last');
        
        
        
        sSingleDelayOut = xSignal;
        xlsub2_ddbf = xBlock(struct('source', 'monroe_library/delay_bram_fast', 'name', strcat('double_delay_bram_fast',num2str(i))), ...
            struct('delay_len', delay_len, ...
            'register_counter', register_counter, ...
            'bram_latency', bram_latency), ...
            {sSliceLast}, ...
            {sConcated{i}});
    else %in the event that we have NO bits left, we're just done.  proceed to the reassembly.  Also decrease i by one, 'cause its too big now.
        i = i-1;
    end
    
    
    sCramAll = xCram(sConcated, 'cram_all');
else
    sCramAll = xDelay(sCramOut,delay_len, 'delay_all');
end
blockTemp = xBlock(struct('source', str2func('uncram_draw'), 'name', 'uncram'),{numSignals,bitWidth,binPt,signalType});




    blockTemp.bindPort({sCramAll},sOutputs);


%    blockTemp.bindPort(sCramAll,sOutputs);



end


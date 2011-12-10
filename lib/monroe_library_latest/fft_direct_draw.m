function fft_direct_draw(FFTSize, larger_fft_size, input_bit_width, coeff_bit_width, inter_stage_bit_width, output_bit_width, register_coeffs, delay_input, delay_output, sync_tree_input, optional_delay_arr, coeff_group_arr_in, coeff_stage_step_arr, shift_arr)
% xBlock;
% FFTSize = 4;
% larger_fft_size = 13;
% coeff_bit_width = 18*ones(FFTSize, 1);
% register_coeffs = 1;
% delay_input = 1;
% delay_output = 1;
% sync_tree_input = 1;
% optional_delay_arr = [1,1,1];
% coeff_group_arr_in = [];
% coeff_stage_step_arr = zeros(1,FFTSize);
% 
% input_bit_width = 18;
% inter_stage_bit_width = 18*ones(1,FFTSize-1);
% output_bit_width = 18;
% shift_arr = ones(1,FFTSize);
% pathToBlock = 'path:fft_direct';
% 
%  


defaultGroup=0; 


%check if any of the input arrays are just a single value, make an
%appropriately-sized array out of it: promotes user-friendlieness
if(length(inter_stage_bit_width) == 1)
    inter_stage_bit_width = inter_stage_bit_width *ones(1,FFTSize-1);
end

if(length(coeff_bit_width) == 1)
    coeff_bit_width = coeff_bit_width *ones(1,FFTSize);
end

if(length(optional_delay_arr) == 1)
    optional_delay_arr = optional_delay_arr *ones(1,FFTSize-1);
end

if(length(coeff_stage_step_arr) == 1)
    coeff_stage_step_arr = coeff_stage_step_arr *ones(1,FFTSize);
end

if(length(shift_arr) == 1)
    shift_arr = shift_arr *ones(1,FFTSize);
end




%check that everything is an ingeger:
if((~isInt([FFTSize, larger_fft_size, input_bit_width, output_bit_width, register_coeffs,delay_input,delay_output, sync_tree_input])))
    strError = 'The following parameters must be integers: FFTSize, larger_FFTSize, register_coeffs,delay_input';
    throwError(strError);
elseif( ~isInt(inter_stage_bit_width) || (~isInt(coeff_bit_width)) || (~isInt(isInt(coeff_group_arr_in))) || (~isInt(optional_delay_arr)) || (~isInt(coeff_stage_step_arr)) || (~isInt(shift_arr)))
    strError = 'The following parameter arrays must be composed entirely of integers: optional_delay_arr, coeff_bit_width, coeff_group_arr_in, coeff_stage_step_arr';
    throwError(strError);
    
    %check that the arrays are of the correct sizes
elseif(length(inter_stage_bit_width) ~= (FFTSize-1))
    strError = strcat('the array ''inter_stage_bit_width'' must be FFTSize-1 elements long;  length(inter_stage_bit_width) = ', num2str(length(inter_stage_bit_width)));
    throwError(strError);
elseif(length(coeff_bit_width) ~= FFTSize)
    strError = strcat('the array ''coeff_bit_width'' must be FFTSize elements long;  length(coeff_bit_width) = ', num2str(length(coeff_bit_width)));
    throwError(strError);
elseif(length(optional_delay_arr) ~= (FFTSize-1))
    strError = strcat('the array ''optional_delay_arr'' must be FFTSize-1 elements long;  length(optional_delay_arr) = ', num2str(length(optional_delay_arr)));
    throwError(strError);
elseif(length(coeff_stage_step_arr) ~= FFTSize)
    strError = strcat('the array ''coeff_stage_step_arr'' must be FFTSize elements long;  length(coeff_stage_step_arr) = ', num2str(length(coeff_stage_step_arr)));
    throwError(strError);
elseif(length(shift_arr) ~= FFTSize)
    strError = strcat('the array ''shift_arr'' must be FFTSize elements long;  length(shift_arr) = ', num2str(length(shift_arr)));
    throwError(strError);
end


if(~min(size(coeff_group_arr_in) == [FFTSize, 2^(FFTSize-1)]))
    if(~(min(size(coeff_group_arr_in) == [0,0])))
        strError = strcat('the matrix ''coeff_group_arr_in'' must be FFTSize_by_2^(FFTSize-1) in size. (that is, ', num2str(FFTSize), '_by_', num2str(2^(FFTSize-1)), '); this is an expert feature... consider replacing this parameter with ''[]''  size(coeff_group_arr_in) = ', num2str(size(coeff_group_arr_in(1))), '_by_', num2str(size(coeff_group_arr_in(2))));
        throwError(strError);
    else
        defaultGroup=1;
        %throwWarning('there are no checks to ensure a valid coefficient group array: your custom array may cause undesired results.');
    end
else
    if((min(min(coeff_group_arr_in) < 0)) && (defaultGroup ~= 1))
        strError = strcat('coeff_group_arr_in must composed of non-negative integers; coeff_group_arr_in = ', num2str(coeff_group_arr_in));
        throwError(strError);
    end
    
end

%check that everything is inside the allowed bounds
if((min(optional_delay_arr) < 0))
    strError = strcat('optional_delay_arr must composed of non-negative integers; optional_delay_arr = ', num2str(optional_delay_arr));
    throwError(strError);
elseif(((min(coeff_bit_width) < 0))|| max(coeff_bit_width) > 18)
    strError = strcat('coeff_bit_width must composed of non-negative integers no greater than 18; coeff_bit_width = ', num2str(coeff_bit_width));
    throwError(strError);
elseif(((min(inter_stage_bit_width) < 0))|| max(inter_stage_bit_width) > 25)
    strError = strcat('inter_stage_bit_width must composed of non-negative integers no greater than 24; inter_stage_bit_width = ', num2str(inter_stage_bit_width));
    throwError(strError);
    
elseif((min(coeff_stage_step_arr) < 0))
    strError = strcat('coeff_stage_step_arr must composed of non-negative integers; coeff_stage_step_arr = ', num2str(coeff_stage_step_arr));
    throwError(strError);
elseif(((min(shift_arr) < 0))|| max(shift_arr) > 1)
    strError = strcat('shift_arr must composed of only 0''s and 1''s; shift_arr = ', num2str(shift_arr));
    throwError(strError);
elseif(FFTSize > larger_fft_size)
    throwError('FFTSize must be <= larger_fft_size');
elseif(input_bit_width < 0)
    strError = strcat('input_bit_width must be non-negative; input_bit_width= ', num2str(input_bit_width));
    throwError(strError);
elseif(output_bit_width < 0)
    strError = strcat('output_bit_width must be non-negative; output_bit_width= ', num2str(output_bit_width));
    throwError(strError);
elseif(register_coeffs ~= 0 && register_coeffs ~= 1)
    strError = strcat('register_coeffs must be 0 or 1; register_coeffs= ', num2str(register_coeffs));
    throwError(strError);
elseif(sync_tree_input ~= 0 && sync_tree_input ~= 1)
    strError = strcat('sync_tree_input must be 0 or 1; sync_tree_input= ', num2str(sync_tree_input));
    throwError(strError);
end






inter_stage_bit_width = [inter_stage_bit_width output_bit_width];
iSync = xInport('sync');
oSync = xOutport('sync_out');
optional_delay_arr_coeff = [optional_delay_arr, delay_output];
optional_delay_arr_direct = [delay_input, optional_delay_arr, delay_output];

for(i = 1: 2^FFTSize)
    iData{i} = xInport(strcat('in',num2str(i-1)));
    oData{i} = xOutport(strcat('out',num2str(i-1)));
end


%arrange the input ports appropriately
k=1;
for(i = 1:2:2^FFTSize)
    sData{i} = iData{k};
    k=k+1;
end
%k=2^(FFTSize-1)+1;
for(i = 2:2:2^FFTSize)
    sData{i} = iData{k};
    k=k+1;
end

%add the bulk delay to the input (special 'cause it does not worry about
%the 2-cycle delay from a+bw)

%each of these delays has two 'parts'.  The first is mandatory, taking care
%of all the delays that are needed to make the data line up in the right
%ways.  The second is optional, based on demand by the user.  It is assumed
%that the if the user wants the data path to use a uniform ammount of
%hardware between stages, so the mandatory delays are rolled into the same
%slices as the first set of optional delays (if they exist).
delay_arr = zeros(1,2^FFTSize);
delay_arr(1:2:(2^FFTSize)) = 1;
if(sync_tree_input > 0)
    delay_arr = delay_arr + 6;
end
if(optional_delay_arr_direct(1) > 0)
    delay_arr = delay_arr + 1;
    optional_delay_arr_direct(1) = optional_delay_arr_direct(1) -1;
end


sData = xBulkDelay(sData, 2^FFTSize, delay_arr, 'bulk_delay_0_1');
%
% for(i=1:(2^FFTSize))
%     sDelayTemp{i} = xSignal;
% end
% bBulkDelay = xBlock(struct('source',str2func('bulk_delay_draw'), ...
%     'name', 'bulk_delay_0_1'), ...
%     {2^(FFTSize), delay_arr}, sData, sDelayTemp);
% sData = sDelayTemp;
%
% %after the first set of delays, we make each individual delay its own
% %element, so they can help with routing.
% for(i=1:(2^FFTSize))
%     sDelayTemp{i} = xSignal;
% end
for(i = (1:optional_delay_arr_direct(1)))
    delay_arr = ones(1,2^FFTSize);
    blockName = strcat('bulk_delay_0_', num2str(i+1));
    sData = xBulkDelay(sIn, 2^FFTSize, delay_arr, blockName);
end


%coeff generator drawing:
for(i = 1:FFTSize)
    sStageCoeffs{i} = xSignal;
end

coeff_delay_arr= [optional_delay_arr, delay_output];
blockTemp = xBlock(struct('source', @direct_coeff_gen_draw, 'name', 'direct_coeff_gen'), ...
    {FFTSize, larger_fft_size, coeff_bit_width, register_coeffs, delay_input, ...
    sync_tree_input, coeff_delay_arr , coeff_group_arr_in, coeff_stage_step_arr});
blockTemp.bindPort({iSync}, {sStageCoeffs{1:FFTSize},oSync});


for(FFTStage = 1:FFTSize)
    %make the signals for the next stage'es outputs.
    
    for(i=1:2^FFTSize)
        sStageDataOut{i} = xSignal;
    end
    
    if(FFTStage == 1)
        stage_in_bit_width = input_bit_width;
    else
        stage_in_bit_width = inter_stage_bit_width(FFTStage-1);
    end
    blockName =  strcat('stage_', num2str(FFTStage));
    blockTemp = xBlock(struct('source', @fft_direct_stage_draw, 'name',blockName), ...
        {FFTSize, FFTStage, stage_in_bit_width, coeff_bit_width(FFTStage), ...
        inter_stage_bit_width(FFTStage), shift_arr(FFTStage)});
    blockTemp.bindPort({sData{:}, sStageCoeffs{FFTStage}}, sStageDataOut);
    
    
    %establish the mapping between the exit ports of one stage and the
    %input ports of the next.
    map = -1 * ones(1,2^FFTSize);
    if(FFTStage ~= FFTSize)
        frameSize = 2^(FFTSize-FFTStage + 1);
        bitRevParam = FFTSize-FFTStage + 1;
    else
        frameSize = 2^(FFTSize);
        bitRevParam = FFTSize;
    end
 
    for(i=0:((2^FFTSize)-1))
        
        if(FFTStage ~= FFTSize)
            frameSize = 2^(FFTSize-FFTStage + 1);
            bitRevParam = FFTSize-FFTStage + 1;
            
            isApBW = mod(i+1,2);
            frameNum = floor(i/frameSize);
            indexInFrame = i - (frameNum*frameSize);
            
            bottomOfFrame = floor(indexInFrame / (frameSize/2));
            k = i-1;
            %         map(i+1) = frameNum * frameSize + bit_reverse(indexInFrame, bitRevParam ) +1;
            %  map(i+1) = frameNum * frameSize + bit_reverse(indexInFrame, FFTSize-FFTStage + 1) +1;
            if((~bottomOfFrame) && isApBW)
                map(i+1) = i;
            elseif ((~bottomOfFrame) && (~isApBW))
                map(i+1) = i + frameSize/2 -1 ;
            elseif ((bottomOfFrame) && (isApBW))
                map(i+1) = i - frameSize/2 +1;
            elseif ((bottomOfFrame) && (~isApBW))
                map(i+1) = i;
            end
            
        else
            map(i+1) = bit_reverse(i, FFTSize);
        end
        
    end %end map generator... what a mess.
    
    map= map+1;
    
    
    %inter-stage delays
    delay_arr_beginning = zeros(1,2^FFTSize);
    
    delay_arr_end = zeros(1,2^FFTSize);
    delay_arr_end(1:2:(2^FFTSize)) = 2;
    
    
    if(FFTStage ~= FFTSize)
        delay_arr_beginning(1:2:(2^FFTSize)) = 1;
    end
    
    delay_arr = delay_arr_beginning(:)' + delay_arr_end(map);
    if(optional_delay_arr_direct(FFTStage+1) > 0)
        delay_arr = delay_arr + 1;
        optional_delay_arr_direct(FFTStage+1) = optional_delay_arr_direct(FFTStage+1) -1;
    end
    
    
    %sStageDataOut = sStageDataOut{map};
    %if(FFTStage ~= FFTSize)
    if(1)
        for(i=1:(2^FFTSize))%we are trying to accomplish what the above line SHOULD do (it instead returns a single xSignal object)
            sStageDataOutNew{i} =  sStageDataOut{map(i)};
        end
    else
        sStageDataOutNew = sStageDataOut;
    end
    
    blockName = strcat('bulk_delay_', num2str(FFTStage) , '_1');
    sData = xBulkDelay(sStageDataOutNew, 2^FFTSize, delay_arr, blockName);
    
    %optional inter-stage delays
    for(i = (1:optional_delay_arr_direct(FFTStage+1)))
        delay_arr = ones(1,2^FFTSize);
        blockName = strcat('bulk_delay_', num2str(FFTStage) , '_', num2str(i+1));
        sData = xBulkDelay(sData, 2^FFTSize, delay_arr,blockName);
    end
    
end

for(i=1:2^FFTSize)
    %    oData{i}.bind(sData{bit_reverse(i-1,FFTSize)+1});
    oData{i}.bind(sData{i});
end
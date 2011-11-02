function direct_coeff_gen_draw(FFTSize, larger_FFTSize, bit_width, register_coeffs, delay_input, sync_tree_input, optional_delay_arr, coeff_group_arr_in, coeff_stage_step_arr)
%optional_delay_arr: user assigned delays in-between or during each fft stage
%stage_delay_arr: mandatory algorithmic delays for each stage.  as of writing, it is 6 for every stage but the first, which is 3.
%register_coeffs: register the coeffs after pulling them out of the bram. This is managed in the coefficient generators, and is just a passed value
%sel_delay_arr: the incremental ammount of delay we must add to the sel, in addition to all the previous delays.
%coeff_delay_arr: the incremental ammount of delay we must add to the coeff sync pulse, in addition to all the previous delays.
%
% % 
% xBlock;
% FFTSize= 4;
% larger_FFTSize = 13;
% 
% delay_input = 1;
% optional_delay_arr = ones(1,FFTSize);
% sync_tree_input = 0;
% register_coeffs = 1;
% bit_width = 18*ones(1,FFTSize);
% coeff_group_arr = [];
% coeff_group_arr_in = coeff_group_arr;
% coeff_stage_step_arr = zeros(1,FFTSize);
% % coeff_stage_step_arr = [0,0,0]; %if these are set to '1', then the coefficient generator will make a design with every *other* coefficient.
% pathToBlock = 'path:direct_coeff_gen';
% % % %if it is '2', every fourth.  etc.

 
if(length(optional_delay_arr) == 1)
    optional_delay_arr = optional_delay_arr *ones(1,FFTSize);
end

if(length(bit_width) == 1)
    bit_width = bit_width *ones(1,FFTSize);
end

if(length(coeff_stage_step_arr) == 1)
    coeff_stage_step_arr = coeff_stage_step_arr *ones(1,FFTSize);
end

if((~isInt([FFTSize, larger_FFTSize, register_coeffs,delay_input,sync_tree_input])))
    strError = 'The following parameters must be integers: FFTSize, larger_FFTSize, register_coeffs,delay_input';
    throwError(strError);
elseif(~isInt(optional_delay_arr)) || (~isInt(bit_width)) || (~isInt(isInt(coeff_group_arr_in))) || (~isInt(coeff_stage_step_arr))
    strError = 'The following parameter arrays must be composed entirely of integers: optional_delay_arr, coeff_bit_width, coeff_group_arr_in, coeff_stage_step_arr';
    throwError(strError);
    
elseif(length(optional_delay_arr) ~= FFTSize)
    strError = strcat('the array ''optional_delay_array'' must be FFTSize elements long;  length(optional_delay_array) = ', num2str(length(optional_delay_arr)));
    throwError(strError);
elseif(length(bit_width) ~= FFTSize)
    strError = strcat('the array ''coeff_bit_width'' must be FFTSize elements long;  length(coeff_bit_width) = ', num2str(length(bit_width)));
    throwError(strError);
elseif(length(coeff_stage_step_arr) ~= FFTSize)
    strError = strcat('the array ''coeff_stage_step_arr'' must be FFTSize elements long;  length(coeff_stage_step_arr) = ', num2str(length(coeff_stage_step_arr)));
    throwError(strError);
end

if(~min(size(coeff_group_arr_in) == [FFTSize, 2^(FFTSize-1)]))
    if(~(min(size(coeff_group_arr_in) == [0,0])))
        strError = strcat('the matrix ''coeff_group_arr_in'' must be FFTSize_by_2^(FFTSize-1) in size. (that is, ', num2str(FFTSize), '_by_', num2str(2^(FFTSize-1)), '); this is an expert feature... consider replacing this parameter with ''[]''  size(coeff_group_arr_in) = ', num2str(size(coeff_group_arr_in(1))), '_by_', num2str(size(coeff_group_arr_in(2))));
        throwError(strError);
    else 
        defaultGroupArray = 0;
        
    end
end


if((min(optional_delay_arr) < 0))
    strError = strcat('optional_delay_arr must composed of non-negative integers; optional_delay_arr = ', num2str(optional_delay_arr));
    throwError(strError);
elseif(((min(bit_width) < 0))|| max(bit_width) > 18)
    strError = strcat('bit_width must composed of non-negative integers no greater than 18; bit_width = ', num2str(bit_width));
    throwError(strError);
    
elseif((min(coeff_stage_step_arr) < 0))
    strError = strcat('coeff_stage_step_arr must composed of non-negative integers; coeff_stage_step_arr = ', num2str(coeff_stage_step_arr));
    throwError(strError);
end    
if((min(min(coeff_group_arr_in) < 0)))
    strError = strcat('coeff_group_arr_in must composed of non-negative integers; coeff_group_arr_in = ', num2str(coeff_group_arr_in));
    throwError(strError);
end    
    
if(FFTSize > larger_FFTSize)
    throwError('FFTSize must be <= larger_FFTSize');
elseif(register_coeffs ~= 0 && register_coeffs ~= 1)
    strError = strcat('register_coeffs must be 0 or 1; register_coeffs= ', num2str(register_coeffs));
    throwError(strError);
elseif(sync_tree_input ~= 0 && sync_tree_input ~= 1)
    strError = strcat('sync_tree_input must be 0 or 1; sync_tree_input= ', num2str(sync_tree_input));
    throwError(strError);
end



defaultGroupArray  = (min(size(coeff_group_arr_in) == [0,0]));




if(defaultGroupArray == 1) %default to the same as the stock CASPER design
    coeff_group_subarray = 1:2^(FFTSize-1);
    for(i=1:FFTSize)
        coeff_group_arr(i,:) = coeff_group_subarray;
    end
else
    for(i=1:FFTSize)
        coeff_group_arr(i,:) = coeff_group_arr_in(i,:);
    end
end


%I/O
iSync = xInport('sync');
%
% oCoeff = xOutport('coeff');


fft_start_stage = larger_FFTSize - FFTSize +1;
stage_delay_arr = 6* ones(1,FFTSize);
%coeff_delay_arr = [(6*sync_tree_input + delay_input), (optional_delay_arr+stage_delay_arr)];
coeff_delay_arr = [(delay_input), (optional_delay_arr+stage_delay_arr)];
coeffMadeTracker = zeros(FFTSize, 2^FFTSize); %this variable will track if we have made a coefficient generator for a given coefficient group.
%the value will be zero until that group has been made

if(sync_tree_input == 1)
    bSyncTree = xBlock(struct('source', 'monroe_library/delay_tree_z^-6_0'),{}, {iSync}, ...
        {xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal, ...
        xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal, ...
        xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal, ...
        xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal,xSignal});
    sSyncTree = bSyncTree.getOutSignals();
else
    sSyncTree = {iSync,iSync,iSync,iSync,iSync,iSync,iSync,iSync, ...
        iSync,iSync,iSync,iSync,iSync,iSync,iSync,iSync, ...
        iSync,iSync,iSync,iSync,iSync,iSync,iSync,iSync, ...
        iSync,iSync,iSync,iSync,iSync,iSync,iSync,iSync};
    
end


%now, for the coefficients
for(stageNum = 1:FFTSize)
    for(butterflyNum = 1:2^(FFTSize-1))
        groupNum = coeff_group_arr(stageNum, butterflyNum);
        if( coeffMadeTracker(stageNum, groupNum)==0) %if this coeff generation group does not yet exist...
            
            %....make a new coefficient generator to serve those butterflies.
            %sCoeffArr(stageNum, groupNum) = xSignal;
            
            coeff= direct_coeff_gen_calc(FFTSize, stageNum, larger_FFTSize, fft_start_stage, butterflyNum-1);
            coeff = [imag(coeff), real(coeff)];
            coeff = coeff(1:(2^coeff_stage_step_arr(stageNum)):length(coeff)); %this shortens our coeff array so that it corresponds to what the coefficient generator expects.
            if(stageNum == 1)
                sCoeffSyncStage{groupNum} =  xSignal;
                sDelayIn{groupNum} = sSyncTree{groupNum};
                %xDelay(sSyncTree{groupNum}, coeff_delay_arr(stageNum), ...
                %     strcat('coeff_sync_delay_', num2str(stageNum), 'x', num2str(groupNum)));
            else
                %we must now choose our sync driver based on who we got our
                %last sync from.  By doing this, we help keep the routing
                %problem simple.  This degenerates to the stock casper design
                %in the event that all the coefficient groups are
                %different.
                groupNum_last_stage = coeff_group_arr(stageNum-1, butterflyNum);
                sCoeffSyncStage{groupNum} =  xSignal;
                sDelayIn{groupNum} = sCoeffSync{stageNum-1}{groupNum_last_stage};
                
                %sCoeffSyncStage{groupNum} = xDelay(sCoeffSync{stageNum-1}{groupNum_last_stage}, coeff_delay_arr(stageNum), ...
                %strcat('coeff_sync_delay_', num2str(stageNum), 'x', num2str(groupNum)));
            end
            
            bCoeffGenDual = xBlock(struct('source', str2func('coeff_gen_dual_draw'), ...
                'name',strcat('coeff_gen_stage_',num2str(stageNum), '_group_', num2str(groupNum))), ...
                {coeff, 'Block RAM', bit_width(stageNum), coeff_stage_step_arr(stageNum), register_coeffs});
            
            %          bCoeffGenDual.bindPort({sCoeffSync{stageNum,groupNum}}, {sCoeffArr{stageNum, groupNum}});
            bCoeffGenDual.bindPort({sCoeffSyncStage{groupNum}},{xSignal});
            tempArr = bCoeffGenDual.getOutSignals();
            sCoeffArrStage{groupNum} = tempArr{1}; %#ok<AGROW>
            
            
            
            coeffMadeTracker(stageNum, groupNum)=1;
        end
        
        %now we are sure that the coefficient generator has been made.
        %time to compile the signals.
        sCoeffByButterflyArrStage{butterflyNum} = sCoeffArrStage{groupNum};
    end
    %remove empty cells
    sCoeffSyncStage_stripped = sCoeffSyncStage(~cellfun('isempty',sCoeffSyncStage));
    %sCoeffSyncStage_stripped = sCoeffSyncStage;
    sDelayIn_stripped = sDelayIn(~cellfun('isempty',sDelayIn));
    %sDelayIn_stripped = sDelayIn;
    
    blockName = strcat('bulk_delay_', num2str(stageNum), '_1');
    clear sTemp
    for(kkk = 1:length(sDelayIn_stripped))
       sTemp{kkk} = xSignal; 
    end
    
   if(coeff_delay_arr(stageNum)>0)
    bBulkDelay = xBlock(struct('source',str2func('bulk_delay_draw'), ...
        'name', blockName), ...
        {length(sDelayIn_stripped),1}, sDelayIn_stripped, sTemp);
   else
       sTemp=sDelayIn_stripped;
   end
   coeff_delay_thisStage = 0;
    if(coeff_delay_arr(stageNum) -1 < 0)
        coeff_delay_thisStage  = 0;
    else
        coeff_delay_thisStage  = coeff_delay_arr(stageNum) -1;
    end
    blockName = strcat('bulk_delay_', num2str(stageNum), '_2');
    bBulkDelay = xBlock(struct('source',str2func('bulk_delay_draw'), ...
        'name', blockName), ...
        {length(sDelayIn_stripped),coeff_delay_thisStage}, sTemp, sCoeffSyncStage_stripped);
    
    sCoeffSync{stageNum} = sCoeffSyncStage;
    sCoeffArr{stageNum} = sCoeffArrStage;
    sCoeffStageConcated{stageNum} = xConcat(sCoeffByButterflyArrStage,strcat('concat_stage_', num2str(stageNum)));
    oCoeff{stageNum} = xOutport(strcat('coeff_stg',num2str(stageNum)));
    oCoeff{stageNum}.bind(sCoeffStageConcated{stageNum});
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for(i=1:length(sDelayIn))
        sDelayIn{i} = [];
    end
    for(i=1:length(sCoeffSyncStage))
        sCoeffSyncStage{i} = [];
    end
    
end



oSync = xOutport('sync_out');
total_fft_latency =sum(coeff_delay_arr);

%here's our sync output.
blockName = 'delay_sync_1';
sSyncDelay = xDelay(sSyncTree{1}, 1, blockName);
blockName = 'delay_sync_2';
sSyncDelay = xDelay(sSyncDelay, 1, blockName);
blockName = 'delay_sync_3';
sSyncDelay = xDelay(sSyncDelay, total_fft_latency-4, blockName);%put it in the middle because it'll have a bit more hardware cost, want to make that location flexible.
blockName = 'delay_sync_4';
sSyncDelay = xDelay(sSyncDelay, 1, blockName);
blockName = 'delay_sync_5';
sSyncDelay = xDelay(sSyncDelay, 1, blockName);
oSync.bind(sSyncDelay);



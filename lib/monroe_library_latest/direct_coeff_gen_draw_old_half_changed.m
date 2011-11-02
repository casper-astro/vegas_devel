function direct_coeff_gen_draw(fft_size, larger_fft_size, bit_width, register_coeffs, delay_input, sync_tree_input, optional_delay_arr, coeff_group_arr_in, coeff_stage_step_arr)
%optional_delay_arr: user assigned delays in-between or during each fft stage
%stage_delay_arr: mandatory algorithmic delays for each stage.  as of writing, it is 6 for every stage but the first, which is 3.
%register_coeffs: register the coeffs after pulling them out of the bram. This is managed in the coefficient generators, and is just a passed value
%sel_delay_arr: the incremental ammount of delay we must add to the sel, in addition to all the previous delays.
%coeff_delay_arr: the incremental ammount of delay we must add to the coeff sync pulse, in addition to all the previous delays.

%we will ge generating a sel signal for all stages,
%but only a coeff signal for stages 3 to fft_size.
% 
%  xBlock;
%  fft_size= 3;
% larger_fft_size = 13;
% 
% delay_input = 1;
%  optional_delay_arr = [1,1,1];
% sync_tree_input = 0;
%  register_coeffs = 1;
%  bit_width = 18;
% coeff_group_arr = [[1,2,3,4]; ...
%                    [1,2,3,4]; ...
%                    [1,2,3,4]];
% coeff_stage_step_arr = [0,0,0,0]; %if these are set to '1', then the coefficient generator will make a design with every *other* coefficient.
% %if it is '2', every fourth.  etc.

coeff_group_array = zeros(fft_size, 2^(fft_size-1));
if(exist('coeff_group_arr_in') == 0)
    defaultGroupArray = 1;
elseif(size(coeff_group_arr_in) == [0,0])
    defaultGroupArray = 1;
else
    defaultGroupArray = 0;
end


if(defaultGroupArray == 1) %default to the same as the stock CASPER design
   coeff_group_subarray = 1:2^(fft_size-1);
   for(i=1:fft_size)
      coeff_group_arr(i,:) = coeff_group_subarray; 
   end
else
    for(i=1:fft_size)
        coeff_group_arr(i,:) = coeff_group_arr_in(i,:);
    end
end

if(exist('coeff_stage_step_array') == 0)
    defaultStepArray = 1;
elseif(size(coeff_stage_step_array) == [0,0])
    defaultStepArray  = 1;
else
    defaultStepArray  = 0;
end
if(defaultStepArray) %default to using full size coeff banks
   coeff_stage_step_array = zeros(1,fft_size);
end


%I/O
iSync = xInport('sync');
% 
% oCoeff = xOutport('coeff');
oSync = xOutport('sync_out');


fft_start_stage = larger_fft_size - fft_size +1;
stage_delay_arr = 5* ones(1,fft_size);
coeff_delay_arr = [(6*sync_tree_input + delay_input), (optional_delay_arr+stage_delay_arr)];
coeffMadeTracker = zeros(fft_size, 2^fft_size); %this variable will track if we have made a coefficient generator for a given coefficient group.
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
for(stageNum = 1:fft_size)
    for(butterflyNum = 1:2^(fft_size-1))
        groupNum = coeff_group_arr(stageNum, butterflyNum);
        if( coeffMadeTracker(stageNum, groupNum)==0) %if this coeff generation group does not yet exist...
            
            %....make a new coefficient generator to serve those butterflies.
            %sCoeffArr(stageNum, groupNum) = xSignal;
            
            coeff= direct_coeff_gen_calc(fft_size, stageNum, larger_fft_size, fft_start_stage, butterflyNum-1);
            coeff = [imag(coeff), real(coeff)];
            coeff = coeff(1:(2^coeff_stage_step_arr(stageNum)):length(coeff)); %this shortens our coeff array so that it corresponds to what the coefficient generator expects.
            if(stageNum == 1)
               sCoeffSyncStage{groupNum} =  xSignal;
               %xDelay(sSyncTree{groupNum}, coeff_delay_arr(stageNum), ...
               %     strcat('coeff_sync_delay_', num2str(stageNum), 'x', num2str(groupNum)));
            else
                %we must now choose our sync driver based on who we got our
                %last sync from.  By doing this, we help keep the routing
                %problem simple.  This degenerates to the stock casper design
                %in the event that all the coefficient groups are
                %different.
                groupNum_last_stage = coeff_group_arr(stageNum-1, butterflyNum);
                sCoeffSyncStage{groupNum} = xDelay(sCoeffSync{stageNum-1}{groupNum_last_stage}, coeff_delay_arr(stageNum), ...
                    strcat('coeff_sync_delay_', num2str(stageNum), 'x', num2str(groupNum)));
            end
            
            bCoeffGenDual = xBlock(struct('source', str2func('coeff_gen_dual_draw'), ...
                'name',strcat('coeff_gen_stage_',num2str(stageNum), '_group_', num2str(groupNum))), ...
                {coeff, 'Block RAM', bit_width, coeff_stage_step_arr(stageNum), register_coeffs});
                
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
    sCoeffSyncStage = sCoeffSyncStage(~cellfun('isempty',sCoeffSyncStage));
    
    bBulkDelay = xBlock(struct('source',str2func('bulk_delay_draw'), ...
    'name', strcat('bulk_delay_', num2str(stageNum))), ...
    {length(sCoeffSyncStage),coeff_delay_arr(stageNum)}, sCoeffSyncStage, sCoeffSyncStage);
    
    sCoeffSync{stageNum} = sCoeffSyncStage;
    sCoeffArr{stageNum} = sCoeffArrStage;
    sCoeffStageConcated{stageNum} = xConcat(sCoeffByButterflyArrStage,strcat('concat_stage_', num2str(stageNum)));
    oCoeff{stageNum} = xOutport(strcat('coeff_stg',num2str(stageNum)));
    oCoeff{stageNum}.bind(sCoeffStageConcated{stageNum});
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end



total_fft_latency = sum(stage_delay_arr) + sum(coeff_delay_arr);

%here's our sync output.
sSyncDelay = xDelay(sSyncTree{1}, 1);
sSyncDelay = xDelay(sSyncDelay, 1);
sSyncDelay = xDelay(sSyncDelay, total_fft_latency-4);  %put it in the middle because it'll have a bit more hardware cost, want to make that location flexible.
sSyncDelay = xDelay(sSyncDelay, 1);
sSyncDelay = xDelay(sSyncDelay, 1);
oSync.bind(sSyncDelay);



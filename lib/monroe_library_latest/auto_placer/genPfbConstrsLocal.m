function [bramPlaceArr, dspPlaceArr] = genPfbConstrsLocal(pfbSize, nSimInputs, nTaps, inputBitWidth, pfbNum,  current_block)

%todo: handle  pfbSize > 10

if(pfbSize- nSimInputs > 10)
    error('vectorSize > 10 not yet supported!  sorry!')
    
end



vectorLen = pfbSize-nSimInputs;
hashCode = '_??????????/';


emptyPlacementSlot.instName = 'none';
emptyPlacementSlot.rlocGroup = 'none';
emptyPlacementSlot.rloc = 'none';
emptyPlacementSlot.rlocRange = 'none';
emptyPlacementSlot.rlocInst{1} = 'none';

delayBitsPerInputPair = inputBitWidth * (nTaps-1)*2;
numRamb36delay = (2^nSimInputs) * delayBitsPerInputPair / 36 ;

if(numRamb36delay - floor(numRamb36delay) > .5)
    numRamb18delay = 0;
    numRamb36delay = ceil(numRamb36delay);
elseif (numRamb36delay - floor(numRamb36delay) > 0)
    numRamb18delay = 1;
    numRamb36delay = floor(numRamb36delay);
else
    numRamb18delay = 0;
    numRamb36delay = floor(numRamb36delay);
end


bramConsumptionArray = ((0:(2^nSimInputs-1))* delayBitsPerInputPair /36);
bramConsumptionArray = [bramConsumptionArray, numRamb36delay]+1;


for(i=1:2)
    for(k=1:64)
        %             placeGridDspLocal(i,k).instName = 'empty';
        %             placeGridDspLocal(i,k).rlocGroup = 'none';
        % %             placeGridDspLocal(i,k).rloc = 'none';
        %             placeGridBRAMLocal(i,k).instName = 'empty';
        %             placeGridBRAMLocal(i,k).rlocGroup = 'none';
        %             placeGridBRAMLocal(i,k).rloc = 'none';
        placeGridDspLocal(i,k) = emptyPlacementSlot;
        placeGridBramLocal(i,k) = emptyPlacementSlot;
    end
end
numInputs = 2^(nSimInputs);
for(inputNum=0:(2^(nSimInputs-1))-1)
    
    
    %dsp placement
    rloc1 = ' ';
    rloc2 = ' ';
    rlocGroup = strcat('auto_pfb', num2str(pfbNum), '_in', num2str(inputNum));
        
    stageInitialVerticalPlaceLoc = getVerticalPlaceLoc(placeGridDspLocal, 1);
    for(i=0:(nTaps-1))
        dspPlaceLoc = getVerticalPlaceLoc(placeGridDspLocal, 1);
        
         
        % dsp_constrs = strcat(dsp_constrs,  buildAutoplaceHeader(current_block), 'filter_in', num2str(inputNum), '_stage', num2str(i), hashCode, 'dsp48e_mult', hashCode, 'dsp48e/dsp48e_inst" LOC = DSP48_X0Y',  num2str(dsp_col0_min_lvl+i), '/n', newline);
        % dsp_constrs = strcat(dsp_constrs,  buildAutoplaceHeader(current_block), 'filter_in', num2str(7-inputNum), '_stage', num2str(i), hashCode, 'dsp48e_mult', hashCode, 'dsp48e/dsp48e_inst" LOC = DSP48_X1Y',  num2str(dsp_col0_min_lvl+i), '/n', newline);
        instName1 = strcat(buildAutoplaceHeader(current_block), 'filter_in', num2str(inputNum), '_stage', num2str(1+i), hashCode, 'dsp48e_mult', hashCode, 'dsp48e/dsp48e_inst"');
        instName2 = strcat(buildAutoplaceHeader(current_block), 'filter_in', num2str((numInputs)-inputNum-1), '_stage', num2str(1+i), hashCode, 'dsp48e_mult', hashCode, 'dsp48e/dsp48e_inst"');
        placeGridDspLocal(1,dspPlaceLoc).instName = instName1;
        placeGridDspLocal(2, dspPlaceLoc).instName = instName2;
        placeGridDspLocal(1, dspPlaceLoc).rlocGroup = rlocGroup;
        placeGridDspLocal(2, dspPlaceLoc).rlocGroup = rlocGroup;
        
%         if(i==0)
%             placeGridDspLocal(1, dspPlaceLoc).rloc = strcat('X20Y10');
%             %rloc2 = strcat('X12Y', num2str(5*i));
%         end
        
        %placeGridDspLocal(1, dspPlaceLoc).rloc = rloc1;
        %placeGridDspLocal(2, dspPlaceLoc).rloc = rloc2;
    end
    
    
    
    
    
    %INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real1_955976e529/coeff_gen_1_6_5b14b22e07/coeff_ram_3/comp9.core_instance9/BU2/U0/blk_mem_generator/valid.cstr/ramloop[0].ram.r/v5_init.ram/TRUE_DP.SINGLE_PRIM18.TDP" LOC = RAMB36_X0Y4;
    
    %coeff bram placement
    coeffSearchStartPoint = getVerticalPlaceLoc(placeGridBramLocal, 1);
    for(i=1:(nTaps))
        bramPlaceLoc = getVerticalPlaceLoc(placeGridBramLocal, 1);
        instStr = strcat(buildAutoplaceHeader(current_block), 'coeff_gen_', num2str(inputNum), '_', num2str(numInputs-inputNum-1), hashCode, 'coeff_ram_', num2str(i), '/*', '/blk_mem_generator/*', '/TRUE_DP.SINGLE_PRIM18.TDP"');
        placeGridBramLocal(1, bramPlaceLoc).instName = instStr;
        placeGridBramLocal(1, bramPlaceLoc).rlocGroup= rlocGroup;
        
    end
    
    
    
    %INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real1_955976e529/
    %multi_delay_bram_fast_draw1_61594231c8/double_delay_bram_fast3_9ced2d73ab/dual_port_ram/comp29.core_instance29/BU2/U0/blk_mem_generator/valid.cstr/ramloop[0].ram.r/v5_init.ram/TRUE_DP.SINGLE_PRIM36.TDP" LOC = RAMB36_X1Y6;
    
    %delay bram placement
    for(i = bramConsumptionArray(inputNum+1):bramConsumptionArray(inputNum+2))
        bramPlaceLoc = getVerticalPlaceLoc(placeGridBramLocal, 2);

%         
%         instStr = strcat(buildAutoplaceHeader(current_block), 'multi_delay_bram_fast_0', hashCode, 'double_delay_bram_fast', num2str(i),  '_', hashCode, 'dual_port_ram/*/TRUE_DP.SINGLE_PRIM36.TDP"');
        
        %perform left-col bram packing
        if(vectorLen < 10)
            instStr = strcat(buildAutoplaceHeader(current_block), 'multi_delay_bram_fast_0', hashCode, 'double_delay_bram_fast', num2str(i), hashCode, 'dual_port_ram/*/TRUE_DP.SINGLE_PRIM18.TDP"');
            
            if(strcmp(placeGridBramLocal(1, bramPlaceLoc).instName, 'none'))
                placeGridBramLocal(1, bramPlaceLoc).instName = instStr;
                placeGridBramLocal(1, bramPlaceLoc).rlocGroup = rlocGroup;
            else
                placeGridBramLocal(2, bramPlaceLoc).instName = instStr;
                placeGridBramLocal(2, bramPlaceLoc).rlocGroup = rlocGroup;
            end
        else
            if(~mod(bramPlaceLoc, 2))
                placeGridBramLocal(2, bramPlaceLoc).instName = 'reserved_bram36 (suprise!)';
                bramPlaceLoc = bramPlaceLoc+1;
            end
            instStr = strcat(buildAutoplaceHeader(current_block), 'multi_delay_bram_fast_0', hashCode, 'double_delay_bram_fast', num2str(i), hashCode, 'dual_port_ram/*/TRUE_DP.SINGLE_PRIM36.TDP"');
            
            if(strcmp(placeGridBramLocal(1, bramPlaceLoc).instName, 'none'))
                placeGridBramLocal(1, bramPlaceLoc).instName = instStr;
                placeGridBramLocal(1, bramPlaceLoc).rlocGroup = rlocGroup;
                
                placeGridBramLocal(1, bramPlaceLoc+1).instName = 'reserved_bram36';
            else
                placeGridBramLocal(2, bramPlaceLoc).instName = instStr;
                placeGridBramLocal(2, bramPlaceLoc).rlocGroup = rlocGroup;
                placeGridBramLocal(2, bramPlaceLoc+1).instName = 'reserved_bram36';
            end
        end
    end
    
    
    
    
    %we now have a primitive placement finished.  Lets improve it by
    %shuffling coefficient brams into the right col in the event that
    %it is shorter than the left col
    
    
    
    
    
    
    kk= 1;
    bramIndex = coeffSearchStartPoint;
    %coeffSearchStartPoint
    while(kk< nTaps+1)
        if(strcmp(placeGridBramLocal(2, bramPlaceLoc).instName, 'none'))
            %then there is no bram in the right col at this location.
            %bump the coeff bram over there, and shuffle everything
            %down.
            placeGridBramLocal(2,bramIndex) = placeGridBramLocal(1,bramIndex);
            
            shuffleIndex = bramIndex;
            while(~strcmp(placeGridBramLocal(1,shuffleIndex+1).instName, 'none'))
                placeGridBramLocal(1,shuffleIndex) = placeGridBramLocal(1,shuffleIndex+1);
                shuffleIndex = shuffleIndex +1;
                %bramIndex = shuffleIndex +1;
            end
            placeGridBramLocal(1,shuffleIndex) = emptyPlacementSlot;
        else
            bramIndex = bramIndex + 1;
        end
        kk = kk+1;
    end
    
    
    
    bramConsumptionArray = floor(bramConsumptionArray);
    
    
    
    
    %now add all pfb fabric constraints
    
    %i_user_app1/fft/spec_4kw_4tap_pfb_x0\/pfb_fir_real1_3cd4ef1707\/filter_in15_stage3_dace4675d3\/delay2\/srl_delay.synth_reg_srl_inst\/partial_one.last_srl17e\/reg_array[6].fde_used.u2
    %i_user_app1/fft/spec_4kw_4tap_pfb_x0\/pfb_fir_real1_3cd4ef1707\/coeff_gen_4_11_5d8897e32f\/delay_a_addr2\/srl_delay.synth_reg_srl_inst\/partial_one.last_srl17e\/reg_array[3].fde_used.u2
    %i_user_app1/fft/spec_4kw_4tap_pfb_x0\/pfb_fir_real1_3cd4ef1707\/coeff_gen_1_14_d6fcb04f2b\/counter0\/comp1.core_instance1/U0\/i_synth\/i_baseblox.i_baseblox_counter\/the_addsub\/no_pipelining.the_addsub\/i_lut6.i_lut6_addsub\/i_q.i_simple.qreg\/fd\/output_6
    %i_user_app1/fft/spec_4kw_4tap_pfb_x0\/pfb_fir_real1_3cd4ef1707\/multi_delay_bram_fast_0_956cb39c38\/double_delay_bram_fast7_c8dbb7469d\/counter_limited_fast_fa7ebfd089\/delay\/srl_delay.synth_reg_srl_inst\/partial_one.last_srl17e\/reg_array[5].fde_used.u2
    
    stageFinalVerticalPlaceLoc = getVerticalPlaceLoc(placeGridDspLocal, 1);
    
    diff = stageFinalVerticalPlaceLoc - stageInitialVerticalPlaceLoc;
    placeGridDspLocal(1,stageInitialVerticalPlaceLoc).rlocInst{1} = strcat(buildAutoplaceHeader(current_block), 'filter_in', num2str(inputNum), '_stage?', hashCode, 'delay*"');
    placeGridDspLocal(1,stageInitialVerticalPlaceLoc).rlocInst{2}= strcat(buildAutoplaceHeader(current_block), 'coeff_gen_', num2str(inputNum), '_', num2str(2^nSimInputs-inputNum-1), hashCode, 'delay*"');
    placeGridDspLocal(1,stageInitialVerticalPlaceLoc).rlocInst{3} = strcat(buildAutoplaceHeader(current_block), 'coeff_gen_', num2str(inputNum), '_', num2str(2^nSimInputs-inputNum-1), hashCode, 'counter*"');
    
    rlocRange = strcat('X0Y0:X40Y', num2str(diff*5+10+10));
    placeGridDspLocal(1,stageInitialVerticalPlaceLoc).rlocRange = rlocRange; 
    
    
    kk=3;
    for(i = bramConsumptionArray(inputNum+1):bramConsumptionArray(inputNum+2))
        placeGridDspLocal(1,stageInitialVerticalPlaceLoc).rlocInst{kk} = strcat(buildAutoplaceHeader(current_block), 'multi_delay_bram_fast_0', hashCode, 'double_delay_bram_fast', num2str(i), hashCode, 'counter*"');
        kk=kk+1;
        %instStr = strcat(buildAutoplaceHeader(current_block), 'multi_delay_bram_fast_draw0', hashCode, 'double_delay_bram_fast', num2str(i),  '_', hashCode 'dual_port_ram/*/TRUE_DP.SINGLE_PRIM36.TDP"');
    end
    
    
    %rlocInstLen = length(placeGridDspLocal(1,stageInitialVerticalPlaceLoc).rlocInst);
    
    
    
    
    %rloc2 = strcat('X22Y', num2str(5*i+10));
    
    
    
    
    
    
    
    
    
    
    %all placement and packing for this stage is now finished.
    %since in some situations, the cols will grow at different rates,
    %we need to reserve any slots which have slack.  Otherwise they
    %will be filled in the next stage, placing related hardware very
    %far from each other.
    
    %this is a bad thing.
    
    %first, determine the place that has the highest placed object:
    dspMax = getVerticalPlaceLoc(placeGridDspLocal, 1);
    coeffMax = getVerticalPlaceLoc(placeGridBramLocal, 1);
    delayMax = getVerticalPlaceLoc(placeGridBramLocal, 2);
    
    [totalMax] = max([dspMax, coeffMax, delayMax]);
    
    for(i= dspMax:(totalMax-1))
        placeGridDspLocal(1,i).instName = 'reserved';
        placeGridDspLocal(2,i).instName = 'reserved';
    end
    
    for(i = coeffMax:(totalMax-1))
        placeGridBramLocal(1,i).instName = 'reserved';
    end
    
    
    if(floor(bramConsumptionArray(inputNum+1)) ~= bramConsumptionArray(inputNum+1))
        totalMax = totalMax -2; %if the delays were not the largest...
        %and the final bram is being shared with the next input
        %we want to allow that to be placed in the slot below the start
        %of the next input, so that we don't make our placement
        %delay-limited
    end
    for(i = delayMax:(totalMax-1))
        placeGridBramLocal(2,i).instName = 'reserved';
    end
end

bramPlaceArr = placeGridBramLocal;
dspPlaceArr = placeGridDspLocal;
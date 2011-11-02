function constrs = gen_abs_constrs_pfb_fir_real(pfbSize, nSimInputs, nTaps, coeffBitWidth, inputBitWidth, pfbConfig, current_block)


%first, determine how much DSP and slice we're going to take up...
%ram is measured in RAMB18s.
numDsp = nTaps;
current_block = 'currentBlockString\'
numRamb18Coeff = nTaps; %simplified by the fact that the maximum coeff bit width is 18.
%this also means that the coeffs will *always* be found on the right colmun
%because I don't need to worry about dynamically programming where they
%should be.  I want to minimize colmun space, and I think that will be
%DSP-limited now.  Everything else will conform.  In addition, I can make
%some free BRAMs on the left side, too... if I'm lucky!
newline = [char(10), char(13)];



%design will stack coeffs on left, delays on right.
numDelayBits = inputBitWidth * (nTaps-1);

numRamb36Delay = floor(numDelayBits/18);
numDelayBits = mod(numDelayBits,18);


if(numDelayBits > 9)
    numRamb36Delay = numRamb36Delay +1;
    numRamb18Delay=0;
elseif(numDelayBits>0)
    numRamb18Delay = 1;
else
    numRamb18Delay=0;
end



placeGridDSP = zeros(10, 64);
placeGridBRAM = zeros(7, 64);



%if pfbConfig is 'adc0', then we are targeting adc0 with this pfb.  if it
%is 'adc1', we are targeting 'adc1'.  if it is 'both', then we are
%targeting both.  n_sim_inputs=3 for adc0 or adc1, and n_sim_inputs=4 for 'both'.
%No matter what, you absolutely cannot put two pfbs into
%the same configuration, or have any placed pfb with another in the 'both'
%config.


%lets index brams like so:
%coeffBramArr(count) = [<bram18s_above_bottom_dsp>];
%delayBramArr(count) = [<bram18s_above_bottom_dsp>, <col0/col1>];
%and the bram36 it will be placed = floor(coeffBramArr(count)/2)
%also place by BEL for higher performance.

%set up the local placement.
hashCode = '_??????????/';
constrs = '';
%INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real1_955976e529/multi_delay_bram_fast_draw1_61594231c8/double_delay_bram_fast3_9ced2d73ab/dual_port_ram/comp29.core_instance29/BU2/U0/blk_mem_generator/valid.cstr/ramloop[0].ram.r/v5_init.ram/TRUE_DP.SINGLE_PRIM36.TDP" LOC = RAMB36_X1Y6;
%INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real1_955976e529/coeff_gen_1_6_5b14b22e07/coeff_ram_3/comp9.core_instance9/BU2/U0/blk_mem_generator/valid.cstr/ramloop[0].ram.r/v5_init.ram/TRUE_DP.SINGLE_PRIM18.TDP" LOC = RAMB36_X0Y4;
%INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real_a3983a29b7/filter_in6_stage3_99b132a0f6/dsp48e_mult_add_pcin_f23cd6695f/dsp48e/dsp48e_inst" LOC = DSP48_X1Y42;

dsp_col0_min_lvl = 32; %anything above here is free.
%do the taps that go above the middle...
if(strcmp(pfbConfig, 'adc0'))
    for(inputNum=0:(2^2-1))
        oddTap = mod(filterHeightTotal,2);
        if(numTaps > 2*numRamb36Delay+ numRamb18Delay+oddTap)
            filterHeight = numTaps;
            
        else
            filterHeight = 2*numRamb36Delay+ numRamb18Delay +oddTap;
        end
        
        
        %coefficient brams
        coeffPlaceArr = zeros(1,nTaps);
        coeffPlaceArr = mod((1:nTaps) + floor(nTaps/2),nTaps)';
        if(oddTaps && mod(inputNum,2))
            coeffPlaceArr = coeffPlaceArr + 1;
        end

        %dsp placement
        for(i=0:(nTaps-1))
           dsp_constrs = strcat(dsp_constrs,  buildAutoplaceHeader(current_block), 'filter_in', num2str(inputNum), '_stage', num2str(i), hashCode, 'dsp48e_mult', hashCode, 'dsp48e/dsp48e_inst" LOC = DSP48_X0Y',  num2str(dsp_col0_min_lvl+i), '/n', newline);
           dsp_constrs = strcat(dsp_constrs,  buildAutoplaceHeader(current_block), 'filter_in', num2str(7-inputNum), '_stage', num2str(i), hashCode, 'dsp48e_mult', hashCode, 'dsp48e/dsp48e_inst" LOC = DSP48_X1Y',  num2str(dsp_col0_min_lvl+i), '/n', newline);
           placeGridDSP(1, dsp_col0_min_lvl+i +1) = placeGridDSP(1, dsp_col0_min_lvl+i +1) +1;
           placeGridDSP(2, dsp_col0_min_lvl+i +1) = placeGridDSP(2, dsp_col0_min_lvl+i +1) +1; %policy will be to *increment* these values instead of assigning them to one.  This way, it's easy to detect a double placement.
           
        end
%INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real1_955976e529/coeff_gen_1_6_5b14b22e07/coeff_ram_3/comp9.core_instance9/BU2/U0/blk_mem_generator/valid.cstr/ramloop[0].ram.r/v5_init.ram/TRUE_DP.SINGLE_PRIM18.TDP" LOC = RAMB36_X0Y4;        
        %coeff bram placement
        for(i=0:(nTaps-1))
            bramPlaceLoc = coeffPlaceArr(i) +dsp_col0_min_lvl;
            xilinxBramPlaceLoc = floor(bramPlaceLoc/2);
            belTop = mod(bramPlaceLoc, 2);
           
            instStr = strcat(coeff,buildAutoplaceHeader(current_block), 'coeff_gen_', num2str(inputNum), '_', num2str(7-inputNum), hashCode, 'coeff_ram_', num2str(i), '/*', '/blk_mem_generator/*', '/TRUE_DP.SINGLE_PRIM18.TDP"', newline);
            
            if(belTop)
               bel = 'UPPER'; 
            else
                bel = 'LOWER';
            end
            
           coeff_constrs = strcat(coeff_constrs, instStr, ' LOC = RAMB36_X0Y', num2str(xilinxBramPlaceLoc ), ';', newline);
           coeff_constrs = strcat(coeff_constrs, instStr, ' BEL = ', bel, ';' , newline);
           
           placeGridBRAM(1, dsp_col0_min_lvl+i +1) = placeGridBRAM(1, coeffPlaceArr(i) +dsp_col0_min_lvl) +1;
        end
        
        
            
        
        
        bramPlaceArr = [ones(1,numRamb36Delay), (0:(numRamb36Delay-1)+oddTap)];
        
        for(i = 0:numRamb36Delay)
           bramPlaceLoc = bramPlaceArr(i)*2 +dsp_col0_min_lvl;
           xilinxBramPlaceLoc = ceil(bramPlaceLoc/2);
           %INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real1_955976e529/multi_delay_bram_fast_draw1_61594231c8/double_delay_bram_fast3_9ced2d73ab/dual_port_ram/comp29.core_instance29/BU2/U0/blk_mem_generator/valid.cstr/ramloop[0].ram.r/v5_init.ram/TRUE_DP.SINGLE_PRIM36.TDP" LOC = RAMB36_X1Y6;
           
           
            
        end
       
        
        
        
        
         dsp_col0_min_lvl = dsp_col0_min_lvl+ filterHeightTotal;
    end
end 
for(i=0:(2^n_sim_inputs))
    coeffPlaceArr = zeros(1,nTaps);
    coeffPlaceArr = mod((1:nTaps) + floor(nTaps/2),nTaps);
    if(oddTaps && mod(i,2))
        coeffPlaceArr = tap_locs + 1;
    end
    
    if(strcmp(pfbConfig, 'both') || strcmp(pfbConfig, 'adc0'))
        
    else
        
        
    end
    
end

end

%INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real1_955976e529/multi_delay_bram_fast_draw1_61594231c8/double_delay_bram_fast3_9ced2d73ab/dual_port_ram/comp29.core_instance29/BU2/U0/blk_mem_generator/valid.cstr/ramloop[0].ram.r/v5_init.ram/TRUE_DP.SINGLE_PRIM36.TDP" LOC = RAMB36_X1Y6;
%INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real1_955976e529/coeff_gen_1_6_5b14b22e07/coeff_ram_3/comp9.core_instance9/BU2/U0/blk_mem_generator/valid.cstr/ramloop[0].ram.r/v5_init.ram/TRUE_DP.SINGLE_PRIM18.TDP" LOC = RAMB36_X0Y4;
%INST "spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_XSG_core_config/spec_2pol_4kw_6tap_pfb_x0/pfb_fir_real_a3983a29b7/filter_in6_stage3_99b132a0f6/dsp48e_mult_add_pcin_f23cd6695f/dsp48e/dsp48e_inst" LOC = DSP48_X1Y42;


function [res, header2_stripped]= buildAutoplaceHeader(curBlock)
[modelName,remain] = strtok(curBlock, '/');
strTemp = modelName;
header2_stripped = remain(2:(length(remain)));
header2 = remain;

while(~isempty(remain))
    [strTemp,remain] = strtok(strTemp, '/');
    header2 = strcat(header2, remain, '_??????????/');
end


res = strcat('INST "', modelName, '_XSG_core_config/', modelName, '_XSG_core_config/', modelName, '_x0', header2);



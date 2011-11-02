function coeff_gen_dual_draw(coeff, memory_type, bit_width, step_rate, register_output, mode2)
% This is a generated function based on subsystem:
%     coeff_gen_plan/coeff_gen_dual
% Though there are limitations about the generated script,
% the main purpose of this utility is to make learning
% Sysgen Script easier.
%
% To test it, run the following commands from MATLAB console:
% cfg.source = str2func('coeff_gen_dual');
% cfg.toplevel = 'coeff_gen_plan/coeff_gen_dual';
% args = {my_coeff, my_bit_width, my_register_output, my_memory_type};
% xBlock(cfg, args);
%
% You can edit coeff_gen_dual.m to debug your script.
%
% You can also replace the MaskInitialization code with the
% following commands so the subsystem will be generated
% according to the values of mask parameters.
% cfg.source = str2func('coeff_gen_dual');
% cfg.toplevel = gcb;
% args = {coeff, bit_width, register_output, memory_type};
% xBlock(cfg, args);
%
% To configure the xBlock call in debug mode, in which mode,
% autolayout will be performed every time a block is added,
% run the following commands:
% cfg.source = str2func('coeff_gen_dual');
% cfg.toplevel = gcb;
% cfg.debug = 1;
% args = {coeff, bit_width, register_output, memory_type};
% xBlock(cfg, args);
%
% To make the xBlock smart so it won't re-generate the
% subsystem if neither the arguments nor the scripts are
% changes, use as the following:
% cfg.source = str2func('coeff_gen_dual');
% cfg.toplevel = gcb;
% cfg.depend = {'coeff_gen_dual'};
% args = {coeff, bit_width, register_output, memory_type};
% xBlock(cfg, args);
% %
% % See also xBlock, xInport, xOutport, xSignal, xlsub2script.
% xBlock;
% coeff= (1:128)/128;
% memory_type = 'Block RAM';
% bit_width = 18;
% step_rate = 1;
% register_output = 0;
% mode2=0;


if( ~exist('mode2'))
    mode2 = 0;
end


%% inports
iSync = xInport('sync');
%% outports
oCoeff = xOutport('coeff');
%% diagram

counterBits = log2(length(coeff))-1+step_rate + mode2;


sConstBool_0 = xConstBool(0, 'Constant');
sConstVal_0 = xConstVal(0, 'Signed', bit_width, bit_width-1, 'Constant3');
xlsub2_Counter_out1 = xSignal;


xlsub2_Reinterpret_out1 = xSignal;
xlsub2_Reinterpret1_out1 = xSignal;

sCoeffsConcated = xConcat({xlsub2_Reinterpret_out1, xlsub2_Reinterpret1_out1}, 'Concat2');
% block: coeff_gen_plan/coeff_gen_dual/Constant5
%load_val_older = (length(coeff)/2)-2-register_output;
%load_val_old = (2^counterBits) - 2 - register_output;
load_val = 2 + register_output;


sCountPreSlice = xSignal;

% block: coeff_gen_plan/coeff_gen_dual/Counter
xlsub2_Counter = xBlock(struct('source', 'Counter', 'name', 'Counter'), ...
    struct('n_bits', counterBits, 'start_count', load_val, 'rst', 'on' ), ...
    {iSync}, ...
    {sCountPreSlice});



blockTemp = xBlock(struct('source', 'Slice', 'name', 'count_slice'), struct('nbits', log2(length(coeff))-1, 'boolean_output','off', 'mode', 'Upper Bit Location + Width', 'base1', 'MSB of Input', 'base0', 'MSB of Input', 'bit1', 0), {sCountPreSlice}, {xlsub2_Counter_out1});

if(mode2)
    load_val = 2 + register_output;
    
    sCounterTop = xSliceBool(xlsub2_Counter_out1, 'upper', 0, 'slice_count_top1');
    sCounterTop2 = xSliceBool(xlsub2_Counter_out1, 'upper', -1, 'slice_count_top2');
    sCounterRest = xSlice(xlsub2_Counter_out1, counterBits-2, 'lower', 0, 'slice_count_bottom');
    
    sCounterTopNext = xSignal;
    sCounterTop2Next = xSignal;
    blockTemp = xBlock('Logical', struct('logical_function', 'XOR'), {sCounterTop, sCounterTop2}, {sCounterTop2Next});
    blockTemp = xBlock('Inverter', struct('latency', 0), {sCounterTop, sCounterTopNext}, {sCounterTopNext});
    
    sAddrImag = xConcat({sCounterTopNext, sCounterTop2Next, sCounterRest}, 'concat_counter_imag');
    sAddrReal = sCountPreSlice;
else
    
    sConstBool_0_2 = xConstBool(0, 'Constant4');
    sConstBool_1 = xConstBool(1, 'Constant1');
    sAddrReal = xConcat({sConstBool_1, xlsub2_Counter_out1}, 'Concat');
    sAddrImag = xConcat({sConstBool_0_2, xlsub2_Counter_out1}, 'Concat1');
    
end


% block: coeff_gen_plan/coeff_gen_dual/Delay

sCoeff = xDelay(sCoeffsConcated, register_output, 'Delay');
oCoeff.bind(sCoeff);

% block: coeff_gen_plan/coeff_gen_dual/Dual Port RAM4
xlsub2_Dual_Port_RAM4_out1 = xSignal('xlsub2_Dual_Port_RAM4_out1');
xlsub2_Dual_Port_RAM4_out2 = xSignal('xlsub2_Dual_Port_RAM4_out2');
xlsub2_Dual_Port_RAM4 = xBlock(struct('source', 'Dual Port RAM', 'name', 'Dual Port RAM4'), ...
    struct('depth', length(coeff), ...
    'initVector', coeff, ...
    'latency', 2, ...
    'distributed_mem', memory_type, ...
    'write_mode_A', 'No Read On Write', ...
    'write_mode_B', 'No Read On Write'));
if(strcmp(memory_type,'Block RAM'))
    xlsub2_Dual_Port_RAM4.bindPort({sAddrReal, sConstVal_0, sConstBool_0, sAddrImag, sConstVal_0, sConstBool_0}, ...
        {xlsub2_Dual_Port_RAM4_out1, xlsub2_Dual_Port_RAM4_out2});
else
    xlsub2_Dual_Port_RAM4.bindPort({sAddrReal, sConstVal_0, sConstBool_0, sAddrImag}, ...
        {xlsub2_Dual_Port_RAM4_out1, xlsub2_Dual_Port_RAM4_out2});
end
% block: coeff_gen_plan/coeff_gen_dual/Reinterpret
xlsub2_Reinterpret = xBlock(struct('source', 'Reinterpret', 'name', 'Reinterpret'), ...
    struct('force_arith_type', 'on', ...
    'force_bin_pt', 'on'), ...
    {xlsub2_Dual_Port_RAM4_out1}, ...
    {xlsub2_Reinterpret_out1});

% block: coeff_gen_plan/coeff_gen_dual/Reinterpret1
xlsub2_Reinterpret1 = xBlock(struct('source', 'Reinterpret', 'name', 'Reinterpret1'), ...
    struct('force_arith_type', 'on', ...
    'force_bin_pt', 'on'), ...
    {xlsub2_Dual_Port_RAM4_out2}, ...
    {xlsub2_Reinterpret1_out1});




%
end


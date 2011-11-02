function single_bit_invert_draw(bitWidth, bit_to_invert, out_bin_pt)
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
% % %
% % % See also xBlock, xInport, xOutport, xSignal, xlsub2script.
% xBlock;
% bitWidth = 7;
% bit_to_invert = 2;

iDin = xInport('in');

oDout = xOutport('out');

if(bit_to_invert < 5)
    
   sDelayed = xDelay(iDin, 2^bit_to_invert, 'delay');
   oDout.bind(sDelayed);
else
    invertBits = bitWidth-bit_to_invert;
       sToInvert = xSlice(iDin, invertBits, 'upper', 0, 'sliceInvert');
   sLower = xSlice(iDin, bit_to_invert , 'lower', 0, 'sliceLower');
   sInverted = xSignal;
   sConst1 = xConstVal(1, 'unsigned', 1, 0, 'const_1');
   blockTemp = xBlock('AddSub', struct('mode', 'Subtraction', 'latency', 0, 'precision', 'User Defined', 'arith_type', 'Unsigned','n_bits', invertBits, 'bin_pt', 0,  'quantization', 'Truncate', 'overflow', 'Wrap', 'hw_selection', 'Fabric', 'use_behavioral_HDL', 'on'), { sToInvert, sConst1}, {sInverted});
    sOut = xConcat({sInverted, sLower}, 'concat');
     oDout.bind(sOut);
end
% if(bit_to_invert==bitWidth-1)
%    sToInvert = xSlice(iDin, 1, 'upper', 0, 'sliceInvert');
%    sLower = xSlice(iDin, bitWidth-1 , 'lower', 0, 'sliceLower');
%    sInverted = xSignal;
%    
%    bInvert = xBlock('Inverter', {}, {sToInvert}, {sInverted});
%    
%    sOut = xConcat({sInverted, sLower}, 'concat');
%    oDout.bind(sOut);
% else
%     
%    sUpper = xSlice(iDin, bitWidth-bit_to_invert-1, 'upper', 0, 'sliceUpper');
%    sToInvert = xSlice(iDin, 1, 'lower', bit_to_invert, 'sliceInvert');
%    sLower = xSlice(iDin, bit_to_invert , 'lower', 0, 'sliceLower');
%    sInverted = xSignal;
%    
%    bInvert = xBlock('Inverter', {}, {sToInvert}, {sInverted});
%    
%    sOut = xConcat({sUpper, sInverted, sLower}, 'concat');
%    oDout.bind(sOut);
% end
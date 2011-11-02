
%draws a dual pfb coefficient generator.  The coeff_rev output will drive
%another filter:  for a 16 simultaneous input PFB (numbered 0 to 15), this
%coefficient generator, configured to drive input 1, would drive input 1 on
%the coeff line, and input 14 on the coeff_rev line.


%matched pairs for a 16 simultaneous input filter:
%{00,08}
%{01,09}
%{02,10}
%{03,11}
%{04,12}
%{05,13}
%{06,14}
%{07,15}


function pfb_coeff_gen_dual_draw(pfb_size,n_sim_inputs,n_taps,input_num,n_bits,window_fn,mem_type,bram_latency,bin_width)
%initial definitions
% 
% xBlock;
% % %comment these lines out, and un-comment the declaration at the top in order to make this function configurable.
% n_bits=18;
% bin_pt=17;
% pfb_size=14;
% n_taps=2;
% window_fn = 'hamming';
% bram_latency = 3;
% n_sim_inputs = 4;
% input_num = 3;
% bin_width = 1;
% mem_type = 'Block RAM';
% pathToBlock = 'PATH_pfb_coeff_gen_draw';
% %end comment block
% 
vector_len = pfb_size - n_sim_inputs;
%input_num = input_num;
bin_pt = n_bits-1;
%xBlock; %training mode

%draw the sync first.
iSync=xInport('sync');
oSyncOut = xOutport('sync_out');


sSyncOut = xDelay(iSync, 0, 'delay_sync');
oSyncOut.bind(sSyncOut);


%draw other outputs
oCoeff = xOutport('coeff');
oCoeffRev = xOutport('coeff_rev');


%draw our two constants, which we will eventually use on each bram.
%one for the write enable, one for the data-in port.
%we want these BRAMs to never change, so lets set their write-enables to
%'0'

sConst_0_Bool = xConstBool(0,'const_we');
sConst_0_Val = xConstVal(0,'Signed', n_bits, bin_pt, 'const_din');



    
%draw the counters.
%they will count in oppisite directions, taking advantage of the symmetry
%inherent in a PFB.
sCountUp = xSignal;
sCountDown = xSignal;
bCountUp = xBlock(struct('source', 'Counter', 'name', 'Counter0'), struct('cnt_type', 'Free Running', 'operation', 'Up', 'start_count',...
    6+bram_latency, 'cnt_by_val', 1, 'arith_type', 'Unsigned', 'n_bits', vector_len, 'bin_pt', 0, 'load_pin', 'off',...
    'rst', 'on', 'en', 'off', 'period', 1, 'explicit_period', 'on', 'implementation', 'Fabric'), ...
        {iSync}, {sCountUp});
    
bCountDown = xBlock(struct('source', 'Counter', 'name', 'Counter1'), struct('cnt_type', 'Free Running', 'operation', 'Down', 'start_count', (2^vector_len)-1-bram_latency-6, ...
    'cnt_by_val', 1, 'arith_type', 'Unsigned', 'n_bits', vector_len, 'bin_pt', 0, 'load_pin', 'off',...
    'rst', 'on', 'en', 'off', 'period', 1, 'explicit_period', 'on', 'implementation', 'Fabric'), ...
        {iSync}, {sCountDown});
    

    %get our scope for these variables at the top level.
sCramUp = {xSignal, xSignal};
sCramDown = {xSignal, xSignal};

for i=1:n_taps  %do the stuff we have to do for every single bram
    
    %draw the delays on the addressing lines.  I will be using DSP48Es as
    %both multipliers and adders, but I need to delay the input data and
    %coefficients by one clock cycle per stage.  While I must delay the
    %data no matter what, it's alot cheaper to delay a 10-bit counter than
    %an 18-bit coefficient.  So the delay comes here.
    
    aDelayLen = n_taps-i;
    bDelayLen = i-1;
    
    if(aDelayLen > 0)
        aDelayLen = aDelayLen-1;  %these take advantage of the DSP-piplelining optimization
    end
    
    if(bDelayLen > 0)
        bDelayLen = bDelayLen-1;
    end
    
    blockName = strcat('delay_a_addr', num2str(i));
    sDelayA = xDelay(sCountUp, aDelayLen, blockName);
    
    blockName = strcat('delay_b_addr', num2str(i));
    sDelayB = xDelay(sCountDown, bDelayLen , blockName);
    
    %We will use this coefficient generation string to fill the BRAM.
    %taken almost verbatim from the original coefficient generator.
    strRomVals = strcat('pfb_coeff_gen_calc(' , num2str(pfb_size) , ',' , num2str(n_taps) , ',''' , 'hamming' , ''',' , num2str(n_sim_inputs) ...
                                    , ',' , num2str(input_num) , ',' , num2str(bin_width) , ',' , num2str(i) , ')');
    
     romVals = pfb_coeff_gen_calc_ml(pfb_size, n_taps, window_fn , n_sim_inputs, input_num, bin_width ,i) ;
     %romVals= [0,0];
    
    sROM_A = xSignal;
    sROM_B = xSignal;
    
    %draw the rom.  UUUGH CARPAL TUNNEL
    %looks like matlab dosen't like putting these objects directly into
    %arrays.  Must make a single element and load it into an array in a
    %separate operation.
    xBlock(struct('source', 'Dual Port RAM', 'name', strcat('coeff_ram_', num2str(i))), struct('depth', 2^vector_len, 'initVector', romVals, ...
            'distributed_mem', 'Block RAM', 'latency', bram_latency, 'write_mode_A', 'No Read On Write', ...
            'write_mode_B', 'No Read On Write', 'optimize', 'Area') ...
            , {sDelayA, sConst_0_Val, sConst_0_Bool, sDelayB, sConst_0_Val, sConst_0_Bool}, {sROM_A, sROM_B});
    
  %  {sDelayA, sConst_0_Bool, sConst_0_Val, sDelayB, sConst_0_Bool, sConst_0_Val}
  %   class({sDelayA, sConst_0_Bool, sConst_0_Val, sDelayB, sConst_0_Bool, sConst_0_Val})   
 

     
    
   sCramUp{i} = sROM_A;
   sCramDown{n_taps-i+1} = sROM_B;
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
%    
%    bReinterpretA = xBlock('Reinterpret', struct('force_arith_type', 'on', 'arith_type', 'Unsigned', 'force_bin_pt', 'on', 'bin_pt', 0) ...
%            , {sROM_A}, {sReinterpretA});
%    bReinterpretB = xBlock('Reinterpret', struct('force_arith_type', 'on', 'arith_type', 'Unsigned', 'force_bin_pt', 'on', 'bin_pt', 0) ...
%            , {sROM_B}, {sReinterpretB});
%      strFieldNameA= strcat('in',num2str(i-1));
%      strFieldNameB= strcat('in',num2str(n_taps-i));
%      sCramAin.(strFieldNameA)=sReinterpretA;
%      sCramBin.(strFieldNameB)=sReinterpretB;
%    

end

%all the ROMs are now drawn.  We just need to draw the concats and hook
%them up and we are done here.
sCramA_out = xCram(sCramUp, 'cram_upper');
sCramB_out = xCram(sCramDown, 'cram_lower');

oCoeff.bind(sCramA_out);
oCoeffRev.bind(sCramB_out);



function half_band_init_xblock(filter_coeffs)



%% inports
xlsub2_In1 = xInport('In1');
xlsub2_In2 = xInport('In2');
xlsub2_In3 = xInport('In3');
xlsub2_In4 = xInport('In4');
xlsub2_In5 = xInport('In5');
xlsub2_In6 = xInport('In6');
xlsub2_In7 = xInport('In7');
xlsub2_In8 = xInport('In8');
xlsub2_In9 = xInport('In9');
xlsub2_In10 = xInport('In10');
xlsub2_In11 = xInport('In11');
xlsub2_In12 = xInport('In12');

%% outports
xlsub2_Out1 = xOutport('Out1');
xlsub2_Out2 = xOutport('Out2');
xlsub2_Out3 = xOutport('Out3');
xlsub2_Out4 = xOutport('Out4');
xlsub2_Out5 = xOutport('Out5');
xlsub2_Out6 = xOutport('Out6');
xlsub2_Out7 = xOutport('Out7');
xlsub2_Out8 = xOutport('Out8');

%% diagram

% block: half_band_xblock/Subsystem/adder_tree
xlsub2_parallel_fir_out1 = xSignal;
xlsub2_parallel_fir_out2 = xSignal;
xlsub2_parallel_fir1_out2 = xSignal;
xlsub2_parallel_fir2_out2 = xSignal;
xlsub2_parallel_fir3_out2 = xSignal;
xlsub2_adder_tree_sub = xBlock(struct('source',str2func('adder_tree_init_xblock_lib'), 'name', 'adder_tree'), ...
                           {4,1,'off','off'}, ...
                           {xlsub2_parallel_fir_out1, xlsub2_parallel_fir_out2, xlsub2_parallel_fir1_out2, xlsub2_parallel_fir2_out2, xlsub2_parallel_fir3_out2}, ...
                           {xlsub2_Out1, xlsub2_Out2});

% block: half_band_xblock/Subsystem/adder_tree1
xlsub2_parallel_fir1_out1 = xSignal;
xlsub2_parallel_fir_out3 = xSignal;
xlsub2_parallel_fir1_out3 = xSignal;
xlsub2_parallel_fir2_out3 = xSignal;
xlsub2_parallel_fir3_out3 = xSignal;
xlsub2_adder_tree1_sub = xBlock(struct('source', str2func('adder_tree_init_xblock_lib'), 'name', 'adder_tree1'), ...
                            {4,1,'off','off'}, ...
                            {xlsub2_parallel_fir1_out1, xlsub2_parallel_fir_out3, xlsub2_parallel_fir1_out3, xlsub2_parallel_fir2_out3, xlsub2_parallel_fir3_out3}, ...
                            {xlsub2_Out3, xlsub2_Out4});

% block: half_band_xblock/Subsystem/adder_tree3
xlsub2_parallel_fir2_out1 = xSignal;
xlsub2_parallel_fir_out4 = xSignal;
xlsub2_parallel_fir1_out4 = xSignal;
xlsub2_parallel_fir2_out4 = xSignal;
xlsub2_parallel_fir3_out4 = xSignal;
xlsub2_adder_tree3_sub = xBlock(struct('source',str2func('adder_tree_init_xblock_lib'), 'name', 'adder_tree3'), ...
                            {4,1,'off','off'}, ...
                            {xlsub2_parallel_fir2_out1, xlsub2_parallel_fir_out4, xlsub2_parallel_fir1_out4, xlsub2_parallel_fir2_out4, xlsub2_parallel_fir3_out4}, ...
                            {xlsub2_Out5, xlsub2_Out6});

% block: half_band_xblock/Subsystem/adder_tree4
xlsub2_parallel_fir3_out1 = xSignal;
xlsub2_parallel_fir_out5 = xSignal;
xlsub2_parallel_fir1_out5 = xSignal;
xlsub2_parallel_fir2_out5 = xSignal;
xlsub2_parallel_fir3_out5 = xSignal;
xlsub2_adder_tree4_sub = xBlock(struct('source', str2func('adder_tree_init_xblock_lib'), 'name', 'adder_tree4'), ...
                            {4,1,'off','off'}, ...
                            {xlsub2_parallel_fir3_out1, xlsub2_parallel_fir_out5, xlsub2_parallel_fir1_out5, xlsub2_parallel_fir2_out5, xlsub2_parallel_fir3_out5}, ...
                            {xlsub2_Out7, xlsub2_Out8});

% block: half_band_xblock/Subsystem/parallel_fir
xlsub2_parallel_fir_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir'), ...
                             {filter_coeffs}, ...
                             {xlsub2_In1, xlsub2_In2, xlsub2_In3}, ...
                             {xlsub2_parallel_fir_out1, xlsub2_parallel_fir_out2, xlsub2_parallel_fir_out3, xlsub2_parallel_fir_out4, xlsub2_parallel_fir_out5});

% block: half_band_xblock/Subsystem/parallel_fir1
xlsub2_parallel_fir1_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir1'), ...
                              {filter_coeffs}, ...
                              {xlsub2_In4, xlsub2_In5, xlsub2_In6}, ...
                              {xlsub2_parallel_fir1_out1, xlsub2_parallel_fir1_out2, xlsub2_parallel_fir1_out3, xlsub2_parallel_fir1_out4, xlsub2_parallel_fir1_out5});

% block: half_band_xblock/Subsystem/parallel_fir2
xlsub2_parallel_fir2_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir2'), ...
                              {filter_coeffs}, ...
                              {xlsub2_In7, xlsub2_In8, xlsub2_In9}, ...
                              {xlsub2_parallel_fir2_out1, xlsub2_parallel_fir2_out2, xlsub2_parallel_fir2_out3, xlsub2_parallel_fir2_out4, xlsub2_parallel_fir2_out5});

% block: half_band_xblock/Subsystem/parallel_fir3
xlsub2_parallel_fir3_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir3'), ...
                              {filter_coeffs}, ...
                              {xlsub2_In10, xlsub2_In11, xlsub2_In12}, ...
                              {xlsub2_parallel_fir3_out1, xlsub2_parallel_fir3_out2, xlsub2_parallel_fir3_out3, xlsub2_parallel_fir3_out4, xlsub2_parallel_fir3_out5});






end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://casper.berkeley.edu                                                %      
%   Copyright (C) 2011 Hong Chen, Mark Wagner                                 %
%                                                                             %
%   This program is free software; you can redistribute it and/or modify      %
%   it under the terms of the GNU General Public License as published by      %
%   the Free Software Foundation; either version 2 of the License, or         %
%   (at your option) any later version.                                       %
%                                                                             %
%   This program is distributed in the hope that it will be useful,           %
%   but WITHOUT ANY WARRANTY; without even the implied warranty of            %
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             %
%   GNU General Public License for more details.                              %
%                                                                             %
%   You should have received a copy of the GNU General Public License along   %
%   with this program; if not, write to the Free Software Foundation, Inc.,   %
%   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.               %
%                                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function half_band_init_xblock(blk, filter_coeffs)
%% not very parameterized...



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
xlsub2_adder_tree_sub = xBlock(struct('source',str2func('adder_tree_lib_init_xblock'), 'name', 'adder_tree'), ...
                           {[blk, '/adder_tree'], 'n_inputs',4,...
                             'latency', 1, ...
                             'first_stage_hdl', 'off'} , ... 
                           {xlsub2_parallel_fir_out1, xlsub2_parallel_fir_out2, xlsub2_parallel_fir1_out2, xlsub2_parallel_fir2_out2, xlsub2_parallel_fir3_out2}, ...
                           {xlsub2_Out1, xlsub2_Out2});

% block: half_band_xblock/Subsystem/adder_tree1
xlsub2_parallel_fir1_out1 = xSignal;
xlsub2_parallel_fir_out3 = xSignal;
xlsub2_parallel_fir1_out3 = xSignal;
xlsub2_parallel_fir2_out3 = xSignal;
xlsub2_parallel_fir3_out3 = xSignal;
xlsub2_adder_tree1_sub = xBlock(struct('source', str2func('adder_tree_lib_init_xblock'), 'name', 'adder_tree1'), ...
                            {[blk, '/adder_tree1'],'n_inputs',4,...
                             'latency',1, ...
                             'first_stage_hdl', 'off'} , ... 
                            {xlsub2_parallel_fir1_out1, xlsub2_parallel_fir_out3, xlsub2_parallel_fir1_out3, xlsub2_parallel_fir2_out3, xlsub2_parallel_fir3_out3}, ...
                            {xlsub2_Out3, xlsub2_Out4});

% block: half_band_xblock/Subsystem/adder_tree3
xlsub2_parallel_fir2_out1 = xSignal;
xlsub2_parallel_fir_out4 = xSignal;
xlsub2_parallel_fir1_out4 = xSignal;
xlsub2_parallel_fir2_out4 = xSignal;
xlsub2_parallel_fir3_out4 = xSignal;
xlsub2_adder_tree3_sub = xBlock(struct('source',str2func('adder_tree_lib_init_xblock'), 'name', 'adder_tree3'), ...
                            {[blk, '/adder_tree3'],'n_inputs',4,...
                             'latency', 1, ...
                             'first_stage_hdl', 'off'} , ... 
                            {xlsub2_parallel_fir2_out1, xlsub2_parallel_fir_out4, xlsub2_parallel_fir1_out4, xlsub2_parallel_fir2_out4, xlsub2_parallel_fir3_out4}, ...
                            {xlsub2_Out5, xlsub2_Out6});

% block: half_band_xblock/Subsystem/adder_tree4
xlsub2_parallel_fir3_out1 = xSignal;
xlsub2_parallel_fir_out5 = xSignal;
xlsub2_parallel_fir1_out5 = xSignal;
xlsub2_parallel_fir2_out5 = xSignal;
xlsub2_parallel_fir3_out5 = xSignal;
xlsub2_adder_tree4_sub = xBlock(struct('source', str2func('adder_tree_lib_init_xblock'), 'name', 'adder_tree4'), ...
                            {[blk,'/adder_tree4'], ...
                             'n_inputs',4,...
                             'latency', 1, ...
                             'first_stage_hdl', 'off'} , ... 
                            {xlsub2_parallel_fir3_out1, xlsub2_parallel_fir_out5, xlsub2_parallel_fir1_out5, xlsub2_parallel_fir2_out5, xlsub2_parallel_fir3_out5}, ...
                            {xlsub2_Out7, xlsub2_Out8});

% block: half_band_xblock/Subsystem/parallel_fir
xlsub2_parallel_fir_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir'), ...
                             {[blk, '/parallel_fir'], filter_coeffs}, ...
                             {xlsub2_In1, xlsub2_In2, xlsub2_In3}, ...
                             {xlsub2_parallel_fir_out1, xlsub2_parallel_fir_out2, xlsub2_parallel_fir_out3, xlsub2_parallel_fir_out4, xlsub2_parallel_fir_out5});

% block: half_band_xblock/Subsystem/parallel_fir1
xlsub2_parallel_fir1_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir1'), ...
                              {[blk, '/parallel_fir1'],filter_coeffs}, ...
                              {xlsub2_In4, xlsub2_In5, xlsub2_In6}, ...
                              {xlsub2_parallel_fir1_out1, xlsub2_parallel_fir1_out2, xlsub2_parallel_fir1_out3, xlsub2_parallel_fir1_out4, xlsub2_parallel_fir1_out5});

% block: half_band_xblock/Subsystem/parallel_fir2
xlsub2_parallel_fir2_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir2'), ...
                              {[blk, '/parallel_fir2'],filter_coeffs}, ...
                              {xlsub2_In7, xlsub2_In8, xlsub2_In9}, ...
                              {xlsub2_parallel_fir2_out1, xlsub2_parallel_fir2_out2, xlsub2_parallel_fir2_out3, xlsub2_parallel_fir2_out4, xlsub2_parallel_fir2_out5});

% block: half_band_xblock/Subsystem/parallel_fir3
xlsub2_parallel_fir3_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir3'), ...
                              {[blk, '/parallel_fir3'],filter_coeffs}, ...
                              {xlsub2_In10, xlsub2_In11, xlsub2_In12}, ...
                              {xlsub2_parallel_fir3_out1, xlsub2_parallel_fir3_out2, xlsub2_parallel_fir3_out3, xlsub2_parallel_fir3_out4, xlsub2_parallel_fir3_out5});



if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
end


end


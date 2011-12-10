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
function half_band_init_xblock(blk, varargin)
%% not very parameterized...

defaults = {'ExplicitCoefficient', 'on', ...
    'filter_coeffs', [0   -0.0228         0    0.2754    0.5000    0.2754         0   -0.0228], ...
    'n_taps', 8, ...
    'window', 'hamming', ...
    };


ExplicitCoefficient = get_var('ExplicitCoefficient', 'defaults', defaults, varargin{:});
if strcmp(ExplicitCoefficient, 'on')
    disp('Explicit Coefficient on');
    filter_coeffs = get_var('filter_coeffs', 'defaults', defaults, varargin{:});
    % validate coefficients
    check = parallel_fir_coefficient_check(filter_coeffs);
    if check
        return;
    end
else
    disp('Explicit Coefficient off');
    n_taps = get_var('n_taps', 'defaults', defaults, varargin{:});  % number of taps, or the order of the FIR filter
    window = get_var('window', 'defaults', defaults, varargin{:});
    
    if mod(n_taps, 4)
        disp('The number of taps must be a multiple of 4');
        return;
    end
    d = fdesign.halfband('Type', 'Lowpass', 'N', n_taps);
    f = design(d, 'window', 'Window', window);
    filter_coeffs = f.Numerator(2:end);
end

% disp(filter_coeffs);


%% inports
sync_in = xInport('sync_in');
cin_0 = xInport('cin_0');
cin_1 = xInport('cin_1');
cin_2 = xInport('cin_2');
cin_3 = xInport('cin_3');
cin_4 = xInport('cin_4');
cin_5 = xInport('cin_5');
cin_6 = xInport('cin_6');
cin_7 = xInport('cin_7');

%% outports
sync_out = xOutport('sync_out');
out1_real = xOutport('out1_real');
out1_imag = xOutport('out1_imag');
out2_real = xOutport('out2_real');
out2_imag = xOutport('out2_imag');

%% diagram

% xSignals
dec_fir_sync_out = xSignal('dec_fir_sync_out');
real_v0_out = xSignal('real_v0_out');
real_v1_out = xSignal('real_v1_out');
real_v2_out = xSignal('real_v2_out');
real_v3_out = xSignal('real_v3_out');
real_v4_out = xSignal('real_v4_out');
real_v5_out = xSignal('real_v5_out');
real_v6_out = xSignal('real_v6_out');
real_v7_out = xSignal('real_v7_out');
imag_v0_out = xSignal('imag_v0_out');
imag_v1_out = xSignal('imag_v1_out');
imag_v2_out = xSignal('imag_v2_out');
imag_v3_out = xSignal('imag_v3_out');
imag_v4_out = xSignal('imag_v4_out');
imag_v5_out = xSignal('imag_v5_out');
imag_v6_out = xSignal('imag_v6_out');
imag_v7_out = xSignal('imag_v7_out');

% block: half_band_xblock/Subsystem/adder_tree
xlsub2_adder_tree_sub = xBlock(struct('source',str2func('adder_tree_lib_init_xblock'), 'name', 'adder_tree'), ...
                           {[blk, '/adder_tree'], 'n_inputs',4,...
                             'latency', 1, ...
                             'first_stage_hdl', 'off'} , ... 
                           {dec_fir_sync_out, real_v0_out, real_v1_out, real_v2_out, real_v3_out}, ...
                           {sync_out, out1_real});

% block: half_band_xblock/Subsystem/adder_tree1
xlsub2_adder_tree1_sub = xBlock(struct('source', str2func('adder_tree_lib_init_xblock'), 'name', 'adder_tree1'), ...
                            {[blk, '/adder_tree1'],'n_inputs',4,...
                             'latency',1, ...
                             'first_stage_hdl', 'off'} , ... 
                            {dec_fir_sync_out, imag_v0_out, imag_v1_out, imag_v2_out, imag_v3_out}, ...
                            {[], out1_imag});

% block: half_band_xblock/Subsystem/adder_tree3
xlsub2_adder_tree3_sub = xBlock(struct('source',str2func('adder_tree_lib_init_xblock'), 'name', 'adder_tree3'), ...
                            {[blk, '/adder_tree3'],'n_inputs',4,...
                             'latency', 1, ...
                             'first_stage_hdl', 'off'} , ... 
                            {dec_fir_sync_out, real_v4_out, real_v5_out, real_v6_out, real_v7_out}, ...
                            {[], out2_real});

% block: half_band_xblock/Subsystem/adder_tree4
xlsub2_adder_tree4_sub = xBlock(struct('source', str2func('adder_tree_lib_init_xblock'), 'name', 'adder_tree4'), ...
                            {[blk,'/adder_tree4'], ...
                             'n_inputs',4,...
                             'latency', 1, ...
                             'first_stage_hdl', 'off'} , ... 
                            {dec_fir_sync_out, imag_v4_out, imag_v5_out, imag_v6_out, imag_v7_out}, ...
                            {[], out2_imag});

% block: half_band_xblock/Subsystem/parallel_fir
xlsub2_parallel_fir_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir'), ...
                             {[blk, '/parallel_fir'], filter_coeffs}, ...
                             {sync_in, cin_0, cin_4}, ...
                             {dec_fir_sync_out, real_v0_out, imag_v0_out, real_v4_out, imag_v4_out});

% block: half_band_xblock/Subsystem/parallel_fir1
xlsub2_parallel_fir1_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir1'), ...
                              {[blk, '/parallel_fir1'],filter_coeffs}, ...
                              {sync_in, cin_1, cin_5}, ...
                              {[], real_v1_out, imag_v1_out, real_v5_out, imag_v5_out});

% block: half_band_xblock/Subsystem/parallel_fir2
xlsub2_parallel_fir2_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir2'), ...
                              {[blk, '/parallel_fir2'],filter_coeffs}, ...
                              {sync_in, cin_2, cin_6}, ...
                              {[], real_v2_out, imag_v2_out, real_v6_out, imag_v6_out});

% block: half_band_xblock/Subsystem/parallel_fir3
xlsub2_parallel_fir3_sub = xBlock(struct('source', str2func('parallel_fir_init_xblock'), 'name', 'parallel_fir3'), ...
                              {[blk, '/parallel_fir3'],filter_coeffs}, ...
                              {sync_in, cin_3, cin_7}, ...
                              {[], real_v3_out, imag_v3_out, real_v7_out, imag_v7_out});



if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
end


end


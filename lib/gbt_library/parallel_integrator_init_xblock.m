%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://casper.berkeley.edu                                                %      
%   Copyright (C) 2011    Hong Chen                                           %
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
% n_inputs: Number of parallel input streams
% n_bits:   Bit width of input/output data
% bin_pt:   Position of binary point of input/output data
% add_latency:  Add/Sub latency
% reduced_crtc_path: Whether to use the reduced critical path structure;
%                       this designed is abandoned since we don't want the
%                       add_latency to be zero, so it's forced to be 'off'
function parallel_integrator_init_xblock(blk, varargin)

defaults = {'n_inputs', 4, ...
            'n_bits', 18, ...
            'bin_pt', 16, ...
            'add_latency', 2, ...
            'reduced_crtc_path', 'off'};
            

n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
n_bits = get_var('n_bits', 'defaults', defaults, varargin{:});
bin_pt = get_var('bin_pt', 'defaults', defaults, varargin{:});
add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
reduced_crtc_path = get_var('reduced_crtc_path', 'defaults', defaults, varargin{:});
reduced_crtc_path = 'off';

in = cell(1,n_inputs);
for i = 1:n_inputs,
    in{i} = xInport(['in', num2str(i)]);
end
sync = xInport('sync');

out = cell(1, n_inputs);
for i = 1:n_inputs,
    out{i} = xOutport(['out',num2str(i)]);
end

Im = cell(1,n_inputs);
for i = 1:n_inputs,
    Im{i} = xSignal(['Im',num2str(i)]);
    if strcmp(reduced_crtc_path,'off')      %%%% This is where the reduced pipelined critical path trick
        xlsub2_acc{i} = xBlock(struct('source','xbsIndex/Accumulator','name',['Accumulator',num2str(i)]),...
                            struct('n_bits', n_bits, ...
                                    'rst', 'off'), ...
                                    {in{i}},{Im{i}});
    else
        xlsub2_acc{i} = xBlock(struct('source',@my_accumulator_init_xblock ,'name',['Accumulator',num2str(i)]),...
                                   {n_bits,bin_pt,  reduced_crtc_path}, ...
                                    {in{i}},{Im{i}});
    end
end

dIm = cell(1,n_inputs-1);
xlsub3_Delay = cell(1,n_inputs-1);
for i = 2:n_inputs,
    dIm{i-1} = xSignal(['dIm', num2str(i)]);
    xlsub3_Delay{i-1} = xBlock(struct('source', 'Delay', 'name', ['Delay',num2str(i)]), ...
                          struct('latency', 1), ...
                          {Im{i}}, ...
                          {dIm{i-1}});

end

xlsub3_addertree = cell(1,n_inputs);
for i=1:n_inputs,
    xlsub3_addertree{i} = xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', ['adder_tree', num2str(i)]), ...
                                 {[blk,'/','adder_tree', num2str(i)], n_inputs, add_latency, 'Round  (unbiased: +/- Inf)', 'Wrap', 'Behavioral'}, ...
                                 {sync,Im{1:i}, dIm{i:end}}, ...
                                 {{},out{i}});
end



if ~isempty(blk)
    if ~strcmp(blk(1), '/')
        clean_blocks(blk);
        fmtstr = sprintf('Add Latency = %d\nBitwidth = %d bin_pt = %d', add_latency, n_bits, bin_pt);
        set_param(blk, 'AttributesFormatString', fmtstr);
    end
end

end



function my_accumulator_init_xblock(n_bits, bin_pt, reduced_crtc_path)

inport = xInport('in');
outport = xOutport('out');

add_latency = 1;  % hardcoded add latency;

if strcmp(reduced_crtc_path, 'off')         
    accumulator = xBlock(struct('source','xbsIndex/Accumulator','name','accumulator'), ...
                         struct('operation','Add', ...
                                'n_bits',n_bits, ...
                                'overflow', 'Wrap', ...
                                'scale', 1, ...
                                'use_behavioral_HDL', 'on', ...
                                'implementation','DSP48'), ...
                                {inport}, ...
                                {outport});
                                
                                
else
    feedback_signal = xSignal('feedback');
    adder_out = xSignal('adder_out');
    adder_blk = xBlock(struct('source','AddSub', 'name', 'adder_blk1'), ...
                             struct('mode', 'Addition', ...
                                    'latency', add_latency, 'precision', 'User Defined', ...
                                    'arith_type', 'Signed  (2''s comp)', ...
                                    'n_bits', n_bits, 'bin_pt', bin_pt, ...
                                    'quantization', 'Round  (unbiased: +/- Inf)',...
                                    'overflow', 'Wrap', ...
                                    'use_behavioral_HDL', 'on'), ...
                                    {feedback_signal, inport}, ...
                                      {feedback_signal});   
    % delay_blk = xBlock(struct('source','Delay','name', 'delay'), ...
    %                           struct('latency', 1), ...   
    %                           {adder_out}, ...
    %                           {feedback_signal});

    outport.bind(feedback_signal);
end





end


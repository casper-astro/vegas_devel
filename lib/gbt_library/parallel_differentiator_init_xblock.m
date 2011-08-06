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
function parallel_differentiator_init_xblock(blk, n_inputs,n_bits, bin_pt, diff_length, latency)

if n_inputs ==1
    inport = xInport('in');
    outport = xOutport('out');
    
    d_in = xSignal('d_in');
    delay_blk = xBlock(struct('source', 'Delay', 'name', 'Delay'), ...
                          struct('latency', diff_length), ...
                          {inport}, ...
                          {d_in});
                      
    sub_out = xSignal('sub_out');
    sub_blk = xBlock(struct('source','AddSub', 'name', 'sub_blk1'), ...
                         struct('mode', 'Subtraction', 'latency', latency, 'precision', 'User Defined', ...
                                'arith_type', 'Signed  (2''s comp)', ...
                                'n_bits', n_bits, 'bin_pt', bin_pt, ...
                                'quantization', 'Truncate',...
                                'overflow', 'Wrap', ...
                                'use_behavioral_HDL', 'on'), ...
                                {inport, d_in}, ...
                                {sub_out});   
    outport.bind(sub_out);
elseif diff_length < n_inputs
    
    inports = cell(1,n_inputs);
    outports = cell(1,n_inputs);
    for i =1:n_inputs
        inports{i} = xInport(['in',num2str(i)]);
        outports{i} = xOutport(['out',num2str(i)]);
    end
    
    delay_blks = cell(1,diff_length);
    d_ins = cell(1,diff_length);
    for i =1:diff_length
        d_ins{i} = xSignal(['d_in',num2str(n_inputs-i+1)]);
        delay_blks{i} = xBlock(struct('source', 'Delay', 'name', ['Delay',num2str(n_inputs-i+1)]), ...
                              struct('latency', 1), ...
                              {inports{n_inputs-i+1}}, ...
                              {d_ins{i}});
    end
    
    sub_blks = cell(1,n_inputs);
    for i =1:diff_length
        sub_blks{i} = xBlock(struct('source','AddSub', 'name', ['sub_blk', num2str(i)]), ...
                         struct('mode', 'Subtraction', 'latency', latency, 'precision', 'User Defined', ...
                                'arith_type', 'Signed  (2''s comp)', ...
                                'n_bits', n_bits, 'bin_pt', bin_pt, ...
                                'quantization', 'Truncate',...
                                'overflow', 'Wrap', ...
                                'use_behavioral_HDL', 'on'), ...
                                {inports{i}, d_ins{i}}, ...
                                {outports{i}}); 
    end
    
    for i = diff_length+1: n_inputs
            sub_blks{i} = xBlock(struct('source','AddSub', 'name',[ 'sub_blk',num2str(i)]), ...
                     struct('mode', 'Subtraction', 'latency', latency, 'precision', 'User Defined', ...
                            'arith_type', 'Signed  (2''s comp)', ...
                            'n_bits', n_bits, 'bin_pt', bin_pt, ...
                            'quantization', 'Truncate',...
                            'overflow', 'Wrap', ...
                            'use_behavioral_HDL', 'on'), ...
                            {inports{i}, inports{i-diff_length}}, ...
                            {outports{i}}); 
    end
else
    disp('Not supported yet');
end




if ~isempty(blk)
    if ~strcmp(blk(1), '/')
        fmtstr=sprintf('Differentiation length = %d\n', diff_length);
        set_param(blk,'AttributesFormatString', fmtstr);
        clean_blocks(blk);
    end
end

end
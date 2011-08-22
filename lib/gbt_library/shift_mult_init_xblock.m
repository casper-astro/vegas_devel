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
function shift_mult_init_xblock(blk, const,add_latency,n_bits,bin_pt)

inport = xInport('in');
outport = xOutport('out');

const_binary = dec2bin(const);
len = length(const_binary);
len1 = length(find(const_binary=='1'));  % the actually number of shift blocks

shifts = cell(1,len1);
shift_out = cell(1,len1);
k=1;
for i = 1:len
    if strcmp(const_binary(len-i+1) ,'1')
        shift_out{k} = xSignal(['shift_out',num2str(i)]);
        shifts{k} = xBlock(struct('source','Shift','name', ['shift',num2str(i)]), ...
                            struct('shift_dir', 'Left', ...
                                    'shift_bits', i-1, ...
                                    'latency', 0, ...
                                    'Precision','User Defined', ...
                                    'arith_type', 'Signed  (2''s comp)', ...
                                    'n_bits', n_bits+(i-1), ...
                                    'bin_pt', bin_pt), ...
                                    {inport}, ...
                                    {shift_out{k}});
                                
        k=k+1;
    end
end

if len1==1
    outport.bind(shift_out{1});
    
else
    xlsub3_addertree = xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', 'adder_tree'), ...
                             {[blk,'/adder_tree'], len1, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                             [{inport},shift_out], ...
                             {{},outport});

end


if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
end
end
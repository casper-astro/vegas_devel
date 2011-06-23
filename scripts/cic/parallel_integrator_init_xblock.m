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
function parallel_integrator_init_xblock(n_inputs,output_bitwidth, add_latency)

in = cell(1,n_inputs);
for i = 1:n_inputs,
    in{i} = xInport(['in', num2str(i)]);
end
enable = xInport('en');
sync = xInport('sync');

out = cell(1, n_inputs);
for i = 1:n_inputs,
    out{i} = xOutport(['out',num2str(i)]);
end

Im = cell(1,n_inputs);
for i = 1:n_inputs,
    Im{i} = xSignal(['Im',num2str(i)]);
    xlsub2_acc{i} = xBlock(struct('source','xbsIndex/Accumulator','name',['Accumulator',num2str(i)]),...
                        struct('n_bits', output_bitwidth, ...
                                'en', 'on', ...
                                'rst', 'off'), ...
                                {in{i}, enable},{Im{i}});
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
                                 {n_inputs, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                                 {sync,Im{1:i}, dIm{i:end}}, ...
                                 {{},out{i}});
end


end
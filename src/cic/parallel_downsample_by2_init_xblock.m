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
function parallel_downsample_by2_init_xblock(n_inputs)
% for now only supports even n_inputs
% number of outports is twice of number of inports
%    in the case of odd n_inputs, the two numbers should be the same

if mod(n_inputs,2) ==1
    disp('odd n_inputs is not supported for now');
    return
end

inports = cell(1,n_inputs);
for i=1:n_inputs
    inports{i} = xInport(['inport',num2str(i)]);
end
sync = xInport('sync');

outports = cell(1,2*n_inputs);
for i = 1:2*n_inputs,
    outports{i} = xOutport(['outport',num2str(i)]);
end
sync_out = xOutport('sync_out');

Delay = xBlock(struct('source', 'Delay', 'name', 'Delay'), ...
                          struct('latency', 2), ...
                          {sync}, ...
                          {sync_out});

delay_out = cell(1,n_inputs);
delay_blk = cell(1,2*n_inputs);
for i=1:n_inputs
    delay_out{i} = xSignal(['delay_out',num2str(i)]);
    delay_blk{i} = xBlock(struct('source', 'Delay', 'name', ['Delay',num2str(i)]), ...
                          struct('latency', 1), ...
                          {inports{i}}, ...
                          {delay_out{i}});
end

downsample_blk = cell(1,2*n_inputs);
downsample_out = cell(1,n_inputs);
for i = 1:n_inputs
    downsample_out{i} = xSignal(['downsample_out',num2str(i)]);
    downsample_blk{i} = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', ['Down_sample',num2str(i)]), ...
                               struct('sample_ratio',2, ...
                                        'sample_phase','Last Value of Frame  (most efficient)', ...
                                      'latency', 1), ...
                          {inports{i}}, ...
                          {downsample_out{i}});
end


% delay first
for i = n_inputs+1:2*n_inputs
    downsample_blk{i} = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', ['Down_sample',num2str(i)]), ...
                               struct('sample_ratio',2, ...
                                        'sample_phase','Last Value of Frame  (most efficient)', ...
                                      'latency', 1), ...
                          {delay_out{i-n_inputs}}, ...
                          {outports{i}});    
end

% downsample first
for i=n_inputs+1:2*n_inputs
    delay_blk{i} = xBlock(struct('source', 'Delay', 'name', ['Delay',num2str(i)]), ...
                          struct('latency', 1), ...
                          {downsample_out{i-n_inputs}}, ...
                          {outports{i-n_inputs}});
end

end
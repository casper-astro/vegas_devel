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
% dec_rate: decimation rate
% n_inputs: number of parallel input streams

function parallel_downconverter_init_xblock(blk, varargin)

defaults = {'n_inputs', 4, 'dec_rate', 3};

n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
dec_rate = get_var('dec_rate', 'defaults', defaults, varargin{:});



sync_in = xInport('sync_in');
inports = cell(1,n_inputs);

sync_out = xOutport('sync_out');
outports = cell(1,n_inputs);
for i =1:n_inputs   
    inports{i} = xInport(['In',num2str(i)]);
    outports{i} = xOutport(['Out',num2str(i)]);
end

if gcd(n_inputs, dec_rate) ~= 1
    % when their greatest common factor is not 1
    % this implementation is a very rough one
    % should be working but not efficient
    factor = gcd(n_inputs, dec_rate);
    for i =1:n_inputs
        if mod(i, factor) ~= 1
            xBlock(struct('source', 'Terminator', 'name', ['terminator', num2str(i)]),...
                [], ...
                {inports{i}}, {});
        end
    end
    
    %% finding the amount of delays
    % not the most elegant way, but it works for now
    max_delay = ceil((n_inputs-1)*dec_rate/n_inputs)-1;
    delay_values = cell(1,n_inputs);
    in2out_map =cell(1,n_inputs);
    counter = 1;
    for j=0:dec_rate-1   % columns
        for i =1:factor:n_inputs   % rows
            test = n_inputs*j+i;
            if mod(test-1, dec_rate) == 0
                test
                delay_values{counter} = max_delay - j;
                in2out_map{counter} = [i, j];
                [delay_values{counter}, in2out_map{counter}]
                counter = counter + 1;
                %continue;
            end
        end
    end
    
    delay_values
    in2out_map{:}


    delay_blks = cell(1,n_inputs);
    delay_outs = cell(1,n_inputs);
    downsample_blks = cell(1,n_inputs);

    for i =1:n_inputs
        delay_outs{i} = xSignal(['dO',num2str(i)]);
        delay_blks{i} = xBlock(struct('source','Delay','name', ['delay',num2str(i)]), ...
                                  struct('latency', delay_values{i}), ...   
                                  {inports{in2out_map{i}(1)}}, ...  % take the row
                                  {delay_outs{i}});
        downsample_blks{i} = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', ['Down_sample',num2str(i)]), ...
                                   struct('sample_ratio',dec_rate, ...
                                            'sample_phase','Last Value of Frame  (most efficient)', ...
                                          'latency', 1), ...
                              {delay_outs{i}}, ...
                              {outports{i}});
    end

    % take care of sync pulse
    sync_delay = xBlock(struct('source','Delay','name', 'sync_delay'), ...
                                  struct('latency', 1), ...   
                                  {sync_in}, ...
                                  {sync_out});
    
else
    %% finding the amount of delays
    % not the most elegant way, but it works for now
    max_delay = ceil((n_inputs-1)*dec_rate/n_inputs)-1;
    delay_values = cell(1,n_inputs);
    in2out_map =cell(1,n_inputs);
    for i =1:n_inputs  % rows
        for j=0:dec_rate-1  % columns
            test = n_inputs*j+i;
            if mod(test-1,dec_rate) == 0
                delay_values{i} = max_delay - j;
                in2out_map{i} = ceil(test/dec_rate);
                %continue;
            end
        end
    end

    delay_values
    in2out_map


    delay_blks = cell(1,n_inputs);
    delay_outs = cell(1,n_inputs);
    downsample_blks = cell(1,n_inputs);

    for i =1:n_inputs
        delay_outs{i} = xSignal(['dO',num2str(i)]);
        delay_blks{i} = xBlock(struct('source','Delay','name', ['delay',num2str(i)]), ...
                                  struct('latency', delay_values{i}), ...   
                                  {inports{i}}, ...
                                  {delay_outs{i}});
        downsample_blks{i} = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', ['Down_sample',num2str(i)]), ...
                                   struct('sample_ratio',dec_rate, ...
                                            'sample_phase','Last Value of Frame  (most efficient)', ...
                                          'latency', 1), ...
                              {delay_outs{i}}, ...
                              {outports{in2out_map{i}}});
    end

    % take care of sync pulse
    sync_delay = xBlock(struct('source','Delay','name', 'sync_delay'), ...
                                  struct('latency', 1), ...   
                                  {sync_in}, ...
                                  {sync_out});

end

if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr = sprintf('Decimation Rate = %d', dec_rate);
    set_param(blk, 'AttributesFormatString', fmtstr);
end
end
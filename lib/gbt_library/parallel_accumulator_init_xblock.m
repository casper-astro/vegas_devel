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
% n_inputs: Number of parallel Input streams
% len: accumulate length 
% add_latency:  Add latency
function parallel_accumulator_init_xblock(blk, varargin)

defaults = {'n_inputs', 2, ...
            'len', 2, ...
            'add_latency', 1, ...
            'delay_len', 1};
        
n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
len = get_var('len', 'defaults', defaults, varargin{:});
add_latency= get_var('add_latency', 'defaults', defaults, varargin{:});
delay_len= get_var('delay_len', 'defaults', defaults, varargin{:});

sync_in = xInport('sync_in');
inports = cell(1,n_inputs);

sync_out = xOutport('sync_out');
outports = cell(1,n_inputs);
for i =1:n_inputs
    inports{i} = xInport(['in',num2str(i)]);
    outports{i} = xOutport(['out',num2str(i)]);
end

if len ==1
    sync_out.bind(sync_in);
    for i =1:n_inputs
        outports{i}.bind(inports{i});
    end
    return;
end

if delay_len <1
    delay_len = 1;
end


adder_tree_blks = cell(1,n_inputs);
adder_tree_inports = cell(n_inputs,len);

if len > n_inputs
    delay_blks = cell(int32(len/n_inputs),n_inputs);
    delay_outs = cell(int32(len/n_inputs),n_inputs);
    for i = 1:n_inputs
        delay_outs{1,i} = xSignal(['d1_',num2str(i)]);
        delay_blks{1,i} = xBlock(struct('source','Delay','name', ['delay1_',num2str(i)]), ...
                          struct('latency', delay_len), ...   
                          {inports{i}}, ...
                          {delay_outs{1,i}});
    end
    for i =2: ceil(len/n_inputs)-1
        for j=1:n_inputs
            delay_outs{i,j} = xSignal(['d',num2str(i),'_',num2str(j)]);
            delay_blks{i,j} = xBlock(struct('source','Delay','name', ['delay',num2str(i),'_',num2str(j)]), ...
                              struct('latency', delay_len), ...   
                              {delay_outs{i-1,j}}, ...
                              {delay_outs{i,j}});
        end
    end
    i=ceil(len/n_inputs);
    for j=1:len-1-n_inputs*(i-1)
        delay_outs{i,n_inputs-j+1} = xSignal(['d',num2str(i),'_',num2str(n_inputs-j+1)]);
        delay_blks{i,n_inputs-j+1} = xBlock(struct('source','Delay','name', ['delay',num2str(i),'_',num2str(n_inputs-j+1)]), ...
                  struct('latency', delay_len), ...   
                  {delay_outs{i-1,n_inputs-j+1}}, ...
                  {delay_outs{i,n_inputs-j+1}});
    end
    

else
    delay_blks = cell(1,len-1);
    delay_outs = cell(1,n_inputs);
    for i =1:len-1
        delay_outs{n_inputs-i+1} = xSignal(['d',num2str(n_inputs-i+1)]);
        delay_blks{i} = xBlock(struct('source','Delay','name', ['delay',num2str(n_inputs-i+1)]), ...
                          struct('latency', delay_len), ...   
                          {inports{n_inputs-i+1}}, ...
                          {delay_outs{n_inputs-i+1}});
    end
end




adder_tree_sync_in = cell(1,len+1);
adder_tree_sync_in{1} = sync_in;
for i =1:n_inputs
    adder_tree_inports{i,1} = inports{i};
    for j = 2:len
        if i-j+1> 0
            adder_tree_inports{i,j} = inports{i-j+1};
        else
            adder_tree_inports{i,j} = delay_outs{ceil((-(i-j))/n_inputs),n_inputs-mod(-(i-j+1),n_inputs)};
        end
    end

    adder_tree_sync_in{i+1} = xSignal(['adder_tree_sync_in',num2str(i+1)]);
    adder_tree_sync_in{i+1} =xSignal(['adder_tree_sync_in',num2str(i)]);
    adder_tree_blks{i} = xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', ['adder_tree',num2str(i)]), ...
                     {[blk,'/','adder_tree',num2str(i)], len, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                     {adder_tree_sync_in{i},adder_tree_inports{i,:}}, ...
                     {adder_tree_sync_in{i+1},outports{i}});
end

sync_out.bind(adder_tree_sync_in{n_inputs+1});

    
    
    
if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr = sprintf('Accumulation Length = %d\nAdd Latency = %d\nDelay Length =%d', len, add_latency, delay_len);
    set_param(blk, 'AttributesFormatString', fmtstr);
end
end
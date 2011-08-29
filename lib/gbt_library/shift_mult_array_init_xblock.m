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
function shift_mult_array_init_xblock(blk, const_array, add_latency,n_bits,bin_pt,delay_max)

len = length(const_array);

inport = xInport('in');
outport = cell(1,len);
for i =1:len
    outport{i} = xOutport(['out',num2str(i)]);
end
sync = xInport('sync');
sync_out = xOutport('sync_out');

const_bin_array = cell(1,len);
max_num_bits = 0;  % maximum length of binary string
num_bits = cell(1,len);
for i=1:len
    const_bin_array{i} = dec2bin(const_array(i));
    num_bits{i}  = length(const_bin_array{i});
    if  num_bits{i}> max_num_bits
        max_num_bits = num_bits{i};
    end
end


combined_bin = zeros(1,max_num_bits);
for i =1:max_num_bits
    for j = 1:len
        if combined_bin(1,i) == 1
            continue
        end
        temp_string = const_bin_array{j};
        temp_length = length(temp_string);
        if  (temp_length-i+1)>0
            if strcmp(temp_string(temp_length-i+1),'1')
                combined_bin(1,i) = 1;
            end
        end
        [i,j]
        temp_string
        combined_bin
    end
end

combined_bin
all_shifts = cell(1,max_num_bits);
all_shift_out = cell(1,max_num_bits);
for i =1:max_num_bits
     all_shift_out{i} = xSignal(['shift_out',num2str(i)]);
     if combined_bin(i) ==1 && i>1
        all_shifts{i} = xBlock(struct('source','Shift','name', ['shift',num2str(i)]), ...
                            struct('shift_dir', 'Left', ...
                                    'shift_bits', i-1, ...
                                    'latency', 0, ...
                                    'Precision','User Defined', ...
                                    'arith_type', 'Signed  (2''s comp)', ...
                                    'n_bits', n_bits+(i-1), ...
                                    'bin_pt', bin_pt), ...
                                    {inport}, ...
                                    {all_shift_out{i}});                

     else
         all_shift_out{i}.bind(inport);
     end
end




adder_trees_inputs = cell(1,len);
adder_trees_out = cell(1,len);
delays_amount = cell(1,len);
xlsub3_addertree = cell(1,len);
zero_established = 0;
max_delay = 0;
outs = cell(1,len);
replicated = zeros(1,len);
for i =1:len
    
    outs{i} = xSignal(['out',num2str(i)]);
    adder_trees_out{i} = xSignal(['adder_trees_out',num2str(i)]);
    temp_len = length(find(const_bin_array{i} == '1'));
    delays_amount{i} = ceil(log2(temp_len))*add_latency;
    if delays_amount{i} > max_delay
        max_delay = delays_amount{i};
    end
    

    % replication check
    temp_const_array = const_array(1:i-1);
    rep_check = find(temp_const_array == const_array(i));
    if rep_check
        replicated(1,i) = 1;
        outs{i}=outs{rep_check(1)};
        continue;
    end
    if const_array(i) == 0
        if zero_established 
            adder_trees_out{i}.bind(zero_out);
        else
            zero_out = xSignal('zero_out');
            zero_established = 1;
            xlsub3_zero = xBlock(struct('source', 'Constant','name','zero'), ...
                                    struct('arith_type', 'signed', ...
                                           'const', 0, ...
                                           'n_bits', n_bits, ...
                                           'bin_pt', bin_pt), ...
                                           {}, ...
                                           {zero_out});
            adder_trees_out{i}.bind(zero_out);
        end
        continue;
    end

    
    adder_trees_inputs{i} = cell(1,temp_len);
    temp_array = const_bin_array{i};
    
    temp_array
    temp_len
    
    if temp_len ==1
        ind = find(temp_array == '1');
        adder_trees_out{i}.bind(all_shift_out{num_bits{i}-ind+1});
        continue;
    end
    
    
    k = 1;
    for j = 1:num_bits{i}
        if strcmp(temp_array(num_bits{i}-j+1),'1')
            adder_trees_inputs{i}{k} = all_shift_out{j};
            k = k+1;
        end
    end
    adder_trees_inputs{i}
    xlsub3_addertree{i} = xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', ['adder_tree',num2str(i)]), ...
                             {[blk, '/adder_tree',num2str(i)], temp_len, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                             [{inport},adder_trees_inputs{i}], ...
                             {{},adder_trees_out{i}});
end


% consider the input parameter delay_max
if max_delay < delay_max
    max_delay = delay_max;
end


% add delay blocks and connect to outports
xlsub3_delays = cell(1,len);
for i = 1:len
    if replicated(1,i) 
        continue;
    end
    if delays_amount{i} < max_delay
        xlsub3_delays{i} = xBlock(struct('source','Delay','name',['delay',num2str(i)]), ...
                                    struct('latency', max_delay - delays_amount{i}), ...
                                    {adder_trees_out{i}}, ...
                                    {outs{i}});
        
    else
        outs{i}.bind(adder_trees_out{i});
    end
end

for i =1:len
    outport{i}.bind(outs{i});
end


% take care of sync
sync_delay =  xBlock(struct('source','Delay','name','sync_delay'), ...
                                    struct('latency', max_delay), ...
                                    {sync}, ...
                                    {sync_out});

if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
end

end
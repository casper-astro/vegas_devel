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
function cic_stage_dec2_recursive_init_xblock(blk, varargin)

defaults = {'n_stages', 3, ...
            'n_inputs', 1,...
            'add_latency', 1, ...
            'delay_len', 1, ...
            'input_bits', 18 ...
            'bin_pt', 16, ...
            'input_clk_period', 1, ...
            'input_period', 2};

n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
n_stages = get_var('n_stages', 'defaults', defaults, varargin{:});
input_bits = get_var('input_bits', 'defaults', defaults, varargin{:});
bin_pt = get_var('bin_pt', 'defaults', defaults, varargin{:});
delay_len = get_var('delay_len', 'defaults', defaults, varargin{:});
add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
input_clk_period = get_var('input_clk_period', 'defaults', defaults, varargin{:});
input_period = get_var('input_period', 'defaults', defaults, varargin{:});



coeffs = zeros(1,n_stages+1);
for i = 1:n_stages+1
    coeffs(1,i) = nchoosek(n_stages,i-1);
end
 
n_bits = ceil(n_stages*log2(2*delay_len) + input_bits);




if n_inputs ==1
    
    inport = xInport('in');
    outport = xOutport('out');
    sync_in = xInport('sync');
    sync_out = xOutport('sync_out');
    
    acc_ins = cell(1,n_stages+1);
    acc_ins{1} = inport;
    accumulators = cell(1,n_stages);
    comb_ins = cell(1,n_stages+1);
    comb_ins{1} = xSignal('comb_in1');
    combs_delay = cell(1,n_stages);
    combs_sub = cell(1,n_stages);
    comb_sub_ins = cell(1,n_stages);
    
    for i = 1:n_stages
        acc_ins{i+1} = xSignal(['acc_in',num2str(i)]);
        accumulators{i} = xBlock(struct('source','xbsIndex/Accumulator','name',['accumulator',num2str(i)]), ...
                         struct('operation','Add', ...
                                'n_bits',n_bits, ...
                                'overflow', 'Wrap', ...
                                'scale', 1, ...
                                'use_behavioral_HDL', 'on', ...
                                'rst', 'off', ...
                                'implementation','DSP48'), ...
                                acc_ins(i), ...
                                acc_ins(i+1));
    end
    
    downsampler = xBlock(struct('source', str2func('downsample_init_xblock'), 'name', 'Down_sample1'), ...
                               {[blk,'/','Down_sample1'], 'dec_rate', 2,'input_clk_period',input_clk_period, 'input_period', input_period}, ...
                          [acc_ins(n_stages+1), {sync_in}], ...
                          [comb_ins(1),{sync_out}]); 
                      
    for i = 1:n_stages
        comb_ins{i+1} = xSignal(['comb_in',num2str(i)]);
        comb_sub_ins{i} = xSignal(['comb_sub_in',num2str(i)]);
        combs_delay{i} = xBlock(struct('source','Delay','name',['comb_delay',num2str(i)]), ...
                             struct('latency', delay_len*2*(input_period/input_clk_period)), ...
                             comb_ins(i), ...
                             comb_sub_ins(i));
        combs_sub{i} = xBlock(struct('source', 'AddSub', 'name', ['comb_sub_ins', num2str(i)]), ...
                                struct('mode', 'Subtraction', ...
                                       'latency', add_latency, ...
                                       'precision', 'User Defined', ...
                                       'arith_type','Signed  (2''s comp)',...
                                       'n_bits', n_bits, ...
                                       'bin_pt', bin_pt, ...
                                       'quantization', 'Truncate', ...
                                       'overflow', 'Wrap', ...
                                       'use_behavioral_HDL', 'on'), ...
                                  [comb_ins(i),comb_sub_ins(i)], ...
                                  comb_ins(i+1));
                                 
    end
                                                              
    outport.bind(comb_ins{n_stages+1});
    
else
   
    inports = cell(1,n_inputs);
    outports = cell(1,n_inputs/2);
    
    for i = 1:n_inputs
        inports{i} = xInport(['in',num2str(i)]);
    end
    for i =1:n_inputs/2
        outports{i} = xOutport(['out',num2str(i)])
    end
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');
    
    % add parallel integrator
    int_ins = cell(1,n_stages+1);
    int_blks = cell(1,n_stages);
    int_ins{1} = inports;
    for i =1:n_stages
        int_ins{i+1} = cell(1,n_inputs);
        for j =1:n_inputs
            int_ins{i+1}{j} = xSignal(['int_ins',num2str(i),'_',num2str(j)]);
        end
        int_blks{i} = xBlock(struct('source',str2func('parallel_integrator_init_xblock'),...
                                    'name',['int_',num2str(i)]), ...
                              {strcat(blk,['/int_',num2str(i)]), ...
                                'n_inputs', n_inputs, 'n_bits', n_bits, 'bin_pt', bin_pt, ...
                                'add_latency', add_latency}, ...
                               [{sync_in}, int_ins{i}], ...
                               [{[]},int_ins{i+1}]);
    end
    
    % add terminators
    for i=1:2:n_inputs
        xBlock(struct('source','Terminator','name',['terminator',num2str(i)]), ...
                [], ...
                int_ins{n_stages+1}(i), ...
                {[]});
    end
    
    % add down sampler
    % not really downsampler in this case, becasue we just thrown half of
    % the outputs of the parallel integrator
    % and add comb section
    comb_ins = cell(1,n_stages+1);
    comb_blks = cell(1,n_stages);
    comb_ins{1} = int_ins{n_stages+1}(1:2:end);
    for i =1:n_stages
        comb_ins{i+1} = cell(1,n_inputs/2);
        for j=1:n_inputs/2
            comb_ins{i+1}{j} = xSignal(['comb_ins',num2str(i),'_',num2str(j)]);
        end
        comb_blks{i} = xBlock(struct('source',str2func('parallel_differentiator_init_xblock'), ...
                                     'name',['comb_',num2str(i)]), ...
                              {strcat(blk,['/comb_',num2str(i)]), ...
                               'n_inputs', n_inputs/2,'n_bits', n_bits, 'bin_pt', bin_pt, ...
                               'diff_length', 1, 'latency', add_latency}, ...
                               [{sync_in}, comb_ins{i}], ...   % the sync ports are kind of awkward
                               [{[]}, comb_ins{i+1}]);
    end
    
    
    for i =1:n_inputs/2
        disp(comb_ins{n_stages+1});
        outports{i}.bind(comb_ins{n_stages+1}{i});
    end
    sync_out.bind(sync_in);
end

if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('recursive\n%d_%din\ndec_rate = 2',input_bits, bin_pt);
    set_param(blk,'AttributesFormatString',fmtstr);
end

end


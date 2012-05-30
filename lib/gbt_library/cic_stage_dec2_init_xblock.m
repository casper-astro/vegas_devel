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
% A single stage of CIC with decimation rate of 2
% n_stages:  order of CIC filter
% n_inputs:  number of parallel inputs streams, 
%            only suppports n_inputs == 2^n
%            when n_inputs > 1, output number is divide by 2
% add_latency: Add latency
% n_bits:    bitwidth of input/output data
% bin_pt:    binary point
% input_clk_period: the actual sampling period of the input signal
% input_period: the supposed sampling period of the input signal
%            has to be a multiple of input_clk_period
function cic_stage_dec2_init_xblock(blk, varargin)

defaults = {'n_stages', 3, ...
            'n_inputs', 1,...
            'add_latency', 1, ...
            'n_bits', 18, ...
            'bin_pt', 16, ...
            'input_clk_period', 1, ...
            'input_period', 2};


n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
n_stages = get_var('n_stages', 'defaults', defaults, varargin{:});
n_bits = get_var('n_bits', 'defaults', defaults, varargin{:});
bin_pt = get_var('bin_pt', 'defaults', defaults, varargin{:});
add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
input_clk_period = get_var('input_clk_period', 'defaults', defaults, varargin{:});
input_period = get_var('input_period', 'defaults', defaults, varargin{:});


if mod(log2(n_inputs),1) || mod(input_period/input_clk_period,1)
    disp('Only supports n_inputs == 2^n && input_period must be a multiple of input_clk_period');
    return;
end

coeffs = zeros(1,n_stages+1);
for i = 1:n_stages+1
    coeffs(1,i) = nchoosek(n_stages,i-1);
end
 

if n_inputs==1  % only one input
    
   
    % non-polyphase structure
    inport = xInport('in');
    outport = xOutport('out');
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');
    
    delay_blks = cell(n_stages,1);
    delay_inports = cell(n_stages+1);
    delay_outports = cell(n_stages,2);
    adder_tree_blks = cell(n_stages,1);
    adder_tree_sync_in = cell(n_stages+1);
    delay_inports{1} = inport;
    adder_tree_sync_in{1} = inport;  % fake sync singal
    for i = 1:n_stages
        adder_tree_sync_in{i+1}= xSignal(['adder_tree_sync_in',num2str(i+1)]);
        delay_inports{i+1} = xSignal(['delay_in',num2str(i),'_1']);
        delay_outports{i,1} = delay_inports{i};
        delay_outports{i,2} = xSignal(['delay_out',num2str(i),'_',num2str(2)]);
        delay_blks{i,1} = xBlock(struct('source','Delay','name', ['delay',num2str(i),'_',num2str(1)]), ...
                          struct('latency', input_period/input_clk_period), ...   
                          delay_outports(i,1), ...
                          delay_outports(i,2));
        adder_tree_blks{i} =  xBlock(struct('source', str2func('adder_tree_init_xblock'), ...
                                            'name', ['adder_tree',num2str(i)]), ...
                     {[blk, '/', 'adder_tree',num2str(i)],...
                     'n_inputs', 2, ....
                     'add_latency', add_latency*(input_period/input_clk_period), ...
                     'quantization', 'Round  (unbiased: +/- Inf)', ...
                     'overflow', 'Saturate', ...
                     'mode', 'Behavioral'}, ...
                     [adder_tree_sync_in(i),delay_outports(i,:)], ...
                     {adder_tree_sync_in{i+1},delay_inports{i+1}});
    end
    
    
    downsampler = xBlock(struct('source', str2func('downsample_init_xblock'), 'name', 'Down_sample1'), ...
                               {[blk,'/','Down_sample1'], 'dec_rate', 2,'input_clk_period',input_clk_period, 'input_period', input_period}, ...
                          [delay_inports(n_stages+1), {sync_in}], ...
                          {outport,sync_out}); 
   



else % more than one inputs

    if ~(input_period == input_clk_period)
        disp('This mode hasn''t been implemented yet');
        return;
    end

    if mod(n_inputs,2)~=0
        disp('only supports even number of inputs at this mode');
        return;
    end
    inports= cell(1,n_inputs);
    outports = cell(1,n_inputs/2);
    for i =1:n_inputs
        inports{i} = xInport(['in',num2str(i)]);
    end
    for i=1:n_inputs/2
        outports{i} = xOutport(['out',num2str(i)]);
    end
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');

    parallel_adder_blks = cell(1,n_stages);
    parallel_adder_ins = cell(n_stages+1,n_inputs);
    for i = 1:n_inputs
        parallel_adder_ins{1,i} = inports{i};
    end
    parallel_adder_sync_ins = cell(1,n_stages+1);
    parallel_adder_sync_ins{1} = sync_in; 
    for i =1:n_stages
        for j = 1:n_inputs
            parallel_adder_ins{i+1,j} = xSignal(['pa_i',num2str(i+1),'_',num2str(j)]);
        end
        parallel_adder_sync_ins{i+1} = xSignal(['sync_in',num2str(i+1)]);
        parallel_adder_blks{i} = xBlock(struct('source',str2func('parallel_accumulator_init_xblock'), ...
                                                'name',['p_adder',num2str(i)]), ...
                                        {[blk,'/','p_adder',num2str(i)],'n_inputs', n_inputs,'len', 2,'add_latency', add_latency}, ...
                                        {sync_in,parallel_adder_ins{i,:}}, ...
                                        {parallel_adder_sync_ins{i+1},parallel_adder_ins{i+1,:}});



    end

    for i =1:n_inputs/2
        outports{i}.bind(parallel_adder_ins{n_stages+1,i*2});
    end
    sync_out.bind(parallel_adder_sync_ins{n_stages+1});



end


if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('nonrecursive\n%d_%d_in\nclock(actualy/effective): %d/%d',n_bits, bin_pt,input_clk_period, input_period);
    set_param(blk,'AttributesFormatString',fmtstr);
end


end
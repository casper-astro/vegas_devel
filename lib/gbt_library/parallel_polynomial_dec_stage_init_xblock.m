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
% n_stages:  order of CIC filter
% dec_rate:  Decimation rate
% n_inputs:  number of parallel inputs streams
% polyphase: whether to use polyphase structure within stages
% add_latency: Add latency
% dec2_halfout: When decimation rate is 2, the number of parallel output
%               stream can be just half of n_inputs
% n_bits:    bitwidth of input/output data
% bin_pt:    binary point
% input_clk_rate: the sampling rate of the input signal
% recursive:  Whether to implement this stage using recursive structure
%             Notice that when using this structure, changing the add_latency
%             needs extra attention
% reduced_crtc_path:   Whether to use the reduced critical path structure;
%                       this designed is abandoned since we don't want the
%                       add_latency to be zero, so it's forced to be 'off'
function parallel_polynomial_dec_stage_init_xblock(blk, varargin)

defaults = {'n_stages', 3, ...
            'dec_rate', 3, ...
            'n_inputs', 1,...
            'polyphase', 'off',...
            'add_latency', 1, ...
            'dec2_halfout', 'on', ...
            'n_bits', 18, ...
            'bin_pt', 16, ...
            'reduced_crtc_path', 'off', ...
            'recursive', 1, ...
            'input_clk_rate', 4};


n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
n_stages = get_var('n_stages', 'defaults', defaults, varargin{:});
n_bits = get_var('n_bits', 'defaults', defaults, varargin{:});
bin_pt = get_var('bin_pt', 'defaults', defaults, varargin{:});
add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
dec_rate = get_var('dec_rate', 'defaults', defaults, varargin{:});
polyphase = get_var('polyphase', 'defaults', defaults, varargin{:});
recursive = get_var('recursive', 'defaults', defaults, varargin{:});
dec2_halfout = get_var('dec2_halfout', 'defaults', defaults, varargin{:});
input_clk_rate = get_var('input_clk_rate', 'defaults', defaults, varargin{:});
reduced_crtc_path = get_var('reduced_crtc_path', 'defaults', defaults, varargin{:});   
reduced_crtc_path = 'off';
        


% actually should be able to support any integer
if ~isprime(dec_rate) && dec_rate~=1
    disp('Only supports decimation rate that is a prime number or 1');
    return;
end

if dec_rate==2 
    %parallel_polynomial_dec2_stage_init_xblock(m,n_inputs,polyphase,add_latency,0, n_bits, bin_pt,strcmp(dec2_halfout,'on'));

    %parallel_polynomial_dec2_stage_init_xblock(m,n_inputs,polyphase,add_latency,0, n_bits, bin_pt,dec2_halfout, recursive);
    if recursive
        func_handler = @parallel_dec2_stage_recursive_init_xblock;
        func_handler(blk, n_stages,n_inputs,add_latency,0, n_bits, bin_pt,dec2_halfout, reduced_crtc_path, input_clk_rate); 
    else
        func_handler = @parallel_dec2_stage_nonrecursive_init_xblock;
        func_handler(blk, n_stages,n_inputs,polyphase,add_latency,0, n_bits, bin_pt,dec2_halfout, input_clk_rate);
    end

else
    
    if recursive
        func_handler = @parallel_decN_stage_recursive_init_xblock;
        func_handler(blk,n_stages,dec_rate, n_inputs,add_latency, n_bits, bin_pt,reduced_crtc_path, input_clk_rate);
    else
        func_handler = @parallel_decN_stage_nonrecursive_init_xblock;
        func_handler(blk, n_stages,dec_rate,n_inputs,polyphase,add_latency,0, n_bits, bin_pt, input_clk_rate); 
    end
end


disp('new stage added');

end





function parallel_dec2_stage_nonrecursive_init_xblock(blk, n_stages,n_inputs,polyphase,add_latency,skip, n_bits, bin_pt,dec2_halfout, input_clk_rate)

coeffs = zeros(1,n_stages+1);
for i = 1:n_stages+1
    coeffs(1,i) = nchoosek(n_stages,i-1);
end
 

if n_inputs==1 && strcmp(polyphase,'off') % only one input
    
   
    %parallel_polynomial_decN_stage_init_xblock(n_stages,2,n_inputs,polyphase,add_latency,skip, n_bits, bin_pt)
    parallel_decN_stage_nonrecursive_init_xblock(blk, n_stages, 2,n_inputs,polyphase,add_latency,skip, n_bits, bin_pt, input_clk_rate)


elseif n_inputs ==1 && strcmp(polyphase,'on')
    % polyphase structure
    inport = xInport('in');
    sync_in = xInport('sync_in');
    outport = xOutport('out');
    sync_out = xOutport('sync_out');

    first_delay_out = xSignal('delay1');
    first_delay = xBlock(struct('source','Delay','name', 'first_delay'), ...
                              struct('latency', 1), ...
                              {inport}, ...
                              {first_delay_out});


    ds_out0 =xSignal('downsample_out0');
    ds_out1 = xSignal('downsample_out1');
    downsampling0 = xBlock(struct('source', str2func('downsample_init_xblock'), 'name', 'Down_sample0'), ...
                               {[blk, '/', 'Down_sample0'],'dec_rate',2, 'input_clk_rate',input_clk_rate}, ...
                          {inport, sync_in}, ...
                          {ds_out0,sync_out});  
    sync_poly = xSignal('sync_poly');
    downsampling1 = xBlock(struct('source', str2func('downsample_init_xblock'), 'name', 'Down_sample1'), ...
                               {[blk, '/', 'Down_sample1'],'dec_rate',2,'input_clk_rate',input_clk_rate}, ...
                          {first_delay_out,sync_in}, ...
                          {ds_out1,sync_poly}); 

    poly_out0 = xSignal('poly_out0');
    poly_out1 = xSignal('poly_out1');

    delay_max = find_delay_max(coeffs,add_latency);
    polynomial_blk0 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'), ...
                                    'name','polynomial0'), ...
                            {[blk,'/','/polynomial0'], coeffs(1:2:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                            {ds_out0, sync_poly}, ...
                            {poly_out0, sync_out});      

    polynomial_blk1 = xBlock(struct('source', str2func('polynomial_shift_mult_transpose_init_xblock'), ...
                                    'name','polynomial1'), ...
                            {[blk,'/','/polynomial1'], coeffs(2:2:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                            {ds_out1, sync_poly}, ...
                            {poly_out1,[]});                              

    final_adder = xBlock(struct('source','AddSub', 'name', 'final_adder'), ...
                             struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                    'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                    {poly_out0,poly_out1}, ...
                                    {outport});  

elseif n_inputs ==2 && strcmp(polyphase,'on')
    inport0  = xInport('in1');
    inport1 = xInport('in2');   
    outport1 = xOutport('out1');

    poly_out1_even = xSignal('poly_ou1_even');
    poly_out1_odd = xSignal('poly_out1_odd');
    poly_out0_odd = xSignal('poly_out0_odd');
    poly_out0_even = xSignal('poly_out0_even');
    sync_poly = xSignal('sync_poly');
    polynomial_blk0 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'), ...
                                    'name','polynomial0'), ...
                            {[blk,'/polynomial0'], coeffs, add_latency,n_bits, bin_pt,'on',0}, ...
                            {inport0, inport0}, ...
                            {poly_out0_odd,poly_out0_even,[]}); 
    polynomial_blk1= xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'), ...
                                    'name','polynomial1'), ...
                            {[blk,'/polynomial1'], coeffs, add_latency,n_bits, bin_pt,'on',0}, ...
                            {inport1, inport1}, ...
                            {poly_out1_odd,poly_out1_even,[]}); 

    final_adder1 = xBlock(struct('source','AddSub', 'name', 'final_adder1'), ...
                             struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                    'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                    {poly_out0_even,poly_out1_odd}, ...
                                    {outport1});  


elseif strcmp(dec2_halfout,'on')


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
        outports{i}.bind(parallel_adder_ins{n_stages+1,i*2-skip});
    end
    sync_out.bind(parallel_adder_sync_ins{n_stages+1});


else  % dec2_halfout 'off', n_outputs = n_inputs
    
    inports= cell(1,n_inputs);
    outports = cell(1,n_inputs);
    for i =1:n_inputs
        inports{i} = xInport(['in',num2str(i)]);
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
    parallel_adder_sync_ins{1} = sync_in;  % fake sync signal
    for i =1:n_stages
        for j = 1:n_inputs
            parallel_adder_ins{i+1,j} = xSignal(['pa_i',num2str(i+1),'_',num2str(j)]);
        end
        parallel_adder_sync_ins{i+1} = xSignal(['sync_in',num2str(i+1)]);
        parallel_adder_blks{i} = xBlock(struct('source',str2func('parallel_accumulator_init_xblock'), ...
                                                'name',['p_adder',num2str(i)]), ...
                                        {[blk,'/','p_adder',num2str(i)], 'n_inputs', n_inputs,'len', 2,'add_latency', add_latency}, ...
                                        {sync_in,parallel_adder_ins{i,:}}, ...
                                        {parallel_adder_sync_ins{i+1},parallel_adder_ins{i+1,:}});



    end
    
    decimator = xBlock(struct('source',str2func('parallel_downsampler_init_xblock'), ...
                                'name', 'decimator_2'), ...
                        {[blk, '/', 'decimator_',num2str(2)], 'dec_rate', 2,'n_inputs', n_inputs,'input_clk_rate',input_clk_rate}, ...
                        {parallel_adder_ins{n_stages+1,:}, sync_in}, ...
                        [outports, {sync_out}]);

    


end


if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('nonrecursive\n%d_in\npolyphase:%s',n_bits,polyphase);
    set_param(blk,'AttributesFormatString',fmtstr);
end


end


function parallel_dec2_stage_recursive_init_xblock(blk,n_stages,n_inputs,add_latency,skip, input_bits, bin_pt,dec2_halfout, reduced_crtc_path, input_clk_rate)

coeffs = zeros(1,n_stages+1);
for i = 1:n_stages+1
    coeffs(1,i) = nchoosek(n_stages,i-1);
end
 
n_bits = ceil(n_stages*log2(2*1) + input_bits);


if n_inputs ==1 && strcmp(reduced_crtc_path, 'off')
    
    inport = xInport('in');
    outport = xOutport('out');
    
    if add_latency >0
        on_off_setting = 'on';
    else
        on_off_setting = 'off';
    end
    cic_blk = xBlock(struct('source','xrbsDSP_r4/CIC Filter','name','cic_blk'), ...
                     {'input_bitwidth', input_bits,  ...
                      'input_binpt', bin_pt, ...
                      'filter_type', 'Decimator', ...
                      'rate_change', 2, ...
                      'stages', n_stages, ...
                      'm', 1, ...
                      'pipeline', on_off_setting}, ...
                      {inport}, ...
                      {outport});
    disp('what''s going on');
                                                              
elseif n_inputs ==1 && strcmp(reduced_crtc_path, 'on')
    
    inport = xInport('in');
    outport = xOutport('out');
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');
    
    % draw integrator section
    int_ins = cell(1,n_stages+1);
    int_blks = cell(1,n_stages);
    int_ins{1} = inport;
    for i =1:n_stages
        int_ins{i+1} = xSignal(['int_in',num2str(i+1)]);
        int_blks{i} = xBlock(struct('source',@my_accumulator_init_xblock, 'name', ['integrator_',num2str(i)]), ...
                            {n_bits,bin_pt, reduced_crtc_path}, ...
                            int_ins(i), ...
                            int_ins(i+1));
    end
    
    % add down sampler
    ds_out = xSignal('ds_out');
    downsampler = xBlock(struct('source', str2func('downsample_init_xblock'), 'name', 'Down_sample1'), ...
                          {[blk,'/','Down_sample1'],'dec_rate', 2,'input_clk_rate',input_clk_rate},...
                      [int_ins(n_stages+1),{sync_in}], ...
                      {ds_out,sync_out});
                  
    % add comb section
    comb_ins = cell(1,n_stages+1);
    comb_blks = cell(1,n_stages+1);
    comb_ins{1} = ds_out;
    for i = 1:n_stages
        comb_ins{i+1} = xSignal(['comb_in',num2str(i+1)]);
        comb_blks{i} = xBlock(struct('source',str2func('parallel_differentiator_init_xblock'), ...
                                    'name',['comb_',num2str(i)]), ...
                              {strcat(blk,['/comb_',num2str(i)]), ...
                               'n_inputs', 1,'n_bits',n_bits, 'bin_pt',bin_pt, 'diff_length',1, 'latency',add_latency}, ...
                               {ds_out, comb_ins{i}}, ...
                               {[], comb_ins{i+1}});
    end
    
    outport.bind(comb_ins{n_stages+1});
    
elseif strcmp(dec2_halfout,'on') 
   
    inports = cell(1,n_inputs);
    outports = cell(1,n_inputs/2);
    
    for i = 1:n_inputs
        inports{i} = xInport(['in',num2str(i)]);
        if mod(i,2) == skip
            outports{int32(i/2)}=xOutport(['out',num2str(int32(i/2))]);
        end
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
                                'add_latency', add_latency, 'reduced_crtc_path', reduced_crtc_path}, ...
                               [{sync_in},int_ins{i}], ...
                               [{[]},int_ins{i+1}]);
    end
    
    % add terminators
    for i=1+mod(skip+1,2):2:n_inputs
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
    comb_ins{1} = int_ins{n_stages+1}(1+skip:2:end);
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
        outports{i}.bind(comb_ins{n_stages+1}{i})
    end
    
    
    
else
    inports = cell(1,n_inputs);
    outports = cell(1,n_inputs);
    
    for i = 1:n_inputs
        inports{i} = xInport(['in',num2str(i)]);
        outports{i} = xOutport(['out', num2str(i)]);
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
                                'add_latency', add_latency, 'reduced_crtc_path', reduced_crtc_path}, ...
                               [{sync_in},int_ins{i}], ...
                               [{[]},int_ins{i+1}]);
    end

    
    % add down sampler
    ds_outs=cell(1,n_inputs);
    for i =1:n_inputs
        ds_outs{i}=xSignal(['ds_out',num2str(i)]);
    end
    downsampler=xBlock(struct('source',str2func('parallel_downsampler_init_xblock'), ...
                                'name','downsample_2'),...
                        {[blk, '/', 'downsample_2'], 'dec_rate', 2, 'n_inputs', n_inputs, 'input_clk_rate',input_clk_rate}, ...
                        [int_ins{n_stages+1},{sync_in}],...
                        [ds_outs, {sync_out}]);
    
    
    
    
    % and add comb section
    comb_ins = cell(1,n_stages+1);
    comb_blks = cell(1,n_stages);
    comb_ins{1} = ds_outs;
    for i =1:n_stages
        comb_ins{i+1} = cell(1,n_inputs);
        for j=1:n_inputs
            comb_ins{i+1}{j} = xSignal(['comb_ins',num2str(i),'_',num2str(j)]);
        end
        comb_blks{i} = xBlock(struct('source',str2func('parallel_differentiator_init_xblock'), ...
                                    'name',['comb_',num2str(i)]), ...
                              {strcat(blk,['/comb_',num2str(i)]), ...
                               'n_inputs', n_inputs, 'n_bits', n_bits, 'bin_pt', bin_pt, ...
                               'diff_length', 1, 'latency', add_latency}, ...
                               [comb_ins{i}(1), comb_ins{i}], ...   % the sync ports are kind of awkward
                               [{[]}, comb_ins{i+1}]);
    end
    
    
    for i =1:n_inputs
        outports{i}.bind(comb_ins{n_stages+1}{i})
    end
    
end

if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('recursive\n%d_in\nreduced critical path structure%s',input_bits, reduced_crtc_path);
    set_param(blk,'AttributesFormatString',fmtstr);
end

end

function parallel_decN_stage_nonrecursive_init_xblock(blk, m,dec_rate,n_inputs,polyphase,add_latency,skip, n_bits, bin_pt, input_clk_rate)

coeffs = cic_coefficient_generator(m,dec_rate);

if skip == 0
    skip_setting = 'Last Value of Frame  (most efficient)';
else
end

output_bits = ceil(m*log2(dec_rate) + n_bits);   % m = n_stages = order of CIC filter
 

if n_inputs==1 && strcmp(polyphase,'off') % only one input
    
    % non-polyphase structure
    inport = xInport('in');
    outport = xOutport('out');
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');
    
    delay_blks = cell(m,dec_rate-1);
    delay_inports = cell(m+1);
    delay_outports = cell(m,dec_rate);
    adder_tree_blks = cell(m,1);
    adder_tree_sync_in = cell(m+1);
    delay_inports{1} = inport;
    adder_tree_sync_in{1} = inport;  % fake sync singal
    for i = 1:m
        adder_tree_sync_in{i+1}= xSignal(['adder_tree_sync_in',num2str(i+1)]);
        delay_inports{i+1} = xSignal(['delay_in',num2str(i),'_1']);
        delay_outports{i,1} = delay_inports{i};
        for j = 1:dec_rate-1
            delay_outports{i,j+1} = xSignal(['delay_out',num2str(i),'_',num2str(j+1)]);
            delay_blks{i,j} = xBlock(struct('source','Delay','name', ['delay',num2str(i),'_',num2str(j)]), ...
                              struct('latency', 1), ...   
                              {delay_outports{i,j}}, ...
                              {delay_outports{i,j+1}});
        end
        adder_tree_blks{i} =  xBlock(struct('source', str2func('adder_tree_init_xblock'), ...
                                            'name', ['adder_tree',num2str(i)]), ...
                     {[blk, '/', 'adder_tree',num2str(i)],...
                     'n_inputs', dec_rate, ....
                     'add_latency', add_latency, ...
                     'quantization', 'Round  (unbiased: +/- Inf)', ...
                     'overflow', 'Saturate', ...
                     'mode', 'Behavioral'}, ...
                     [{adder_tree_sync_in{i}},{delay_outports{i,:}}], ...
                     {adder_tree_sync_in{i+1},delay_inports{i+1}});
    end
    
    if dec_rate == 1
        outport.bind(delay_inports{m+1});
    else
        downsampler = xBlock(struct('source', str2func('downsample_init_xblock'), 'name', 'Down_sample1'), ...
                                   {[blk,'/','Down_sample1'], 'dec_rate', dec_rate,'input_clk_rate',input_clk_rate}, ...
                              [delay_inports(m+1), {sync_in}], ...
                              {outport,sync_out}); 
    end


elseif n_inputs ==1 && strcmp(polyphase,'on')
     % polyphase structure
     
    
    inport = xInport('in');
    outport = xOutport('out');
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');
    
    delay_blks = cell(1,dec_rate-1);
    delay_ins = cell(1,dec_rate);
    delay_ins{1} = inport;
    for i =1:dec_rate-1
        delay_ins{i+1} = xSignal(['delay',num2str(i+1)]);
        delay_blks{i} = xBlock(struct('source','Delay','name',['delay',num2str(i)]), ...
                              struct('latency', 1), ...
                              {delay_ins{i}}, ...
                              {delay_ins{i+1}});
    end
                          
    ds_out = cell(1,dec_rate);
    downsampling = cell(1,dec_rate);
    for i =1:dec_rate
        ds_out{i} = xSignal(['downsample_out',num2str(i)]);
        if dec_rate ==1
            ds_out{1}.bind(delay_ins{1});
        else
            downsampling{i} = xBlock(struct('source', str2func('downsample_init_xblock'), 'name', ['Down_sample',num2str(i)]), ...
                                      {[blk,'/','Down_sample',num2str(i)], 'dec_rate', dec_rate,'input_clk_rate',input_clk_rate}, ...
                                     [delay_ins(i),{sync_in}], ...
                                     [ds_out(i),{sync_out}]);
        end
    end

    
    poly_out = cell(1,dec_rate);
    polynomial_blk = cell(1,dec_rate);
    sync_poly = xSignal('sync_poly');
    poly_out{1} = xSignal('poly_out1');
    delay_max = find_delay_max(coeffs,add_latency);
    polynomial_blk{1} = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),...
                                      'name','polynomial0'), ...
                            {[blk, '/polynomial0'], coeffs(1:dec_rate:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                            {ds_out{1}, inport}, ...
                            {poly_out{1}, []});  
    for i =2:dec_rate
        poly_out{i}= xSignal(['poly_out',num2str(i)]);
        polynomial_blk{i} = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),...
                                            'name',['polynomial',num2str(i)]), ...
                            {[blk, '/','polynomial',num2str(i)], coeffs(i:dec_rate:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                            {ds_out{i}, inport}, ...
                            {poly_out{i}, []}); 
    end
                        
    adder_tree_sync_out = xSignal('adder_tree_sync_out');
    final_adder_tree =xBlock(struct('source', str2func('adder_tree_init_xblock'), ...
                                    'name', 'adder_tree'), ...
                     {[blk, '/adder_tree'], ...
                     'n_inputs', dec_rate, ...
                     'add_latency', add_latency, ...
                     'quantization', 'Round  (unbiased: +/- Inf)',...
                     'overflow', 'Saturate',...
                     'mode', 'Behavioral'}, ...
                     [{inport},poly_out], ...
                     {[],outport});  
   
else
    
    
    
    inports= cell(1,n_inputs);
    outports = cell(1,n_inputs);
    for i =1:n_inputs
        inports{i} = xInport(['in',num2str(i)]);
        outports{i} = xOutport(['out',num2str(i)]);
    end
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');
    
    parallel_adder_blks = cell(1,m);
    parallel_adder_ins = cell(m+1,n_inputs);
    for i = 1:n_inputs
        parallel_adder_ins{1,i} = inports{i};
    end
    parallel_adder_sync_ins = cell(1,m+1);
    parallel_adder_sync_ins{1} = sync_in;
    for i =1:m
        for j = 1:n_inputs
            parallel_adder_ins{i+1,j} = xSignal(['pa_i',num2str(i+1),'_',num2str(j)]);
        end
        parallel_adder_sync_ins{i+1} = xSignal(['sync_in',num2str(i+1)]);
        parallel_adder_blks{i} = xBlock(struct('source',str2func('parallel_accumulator_init_xblock'), ...
                                                'name',['p_adder',num2str(i)]), ...
                                        {[blk, '/', 'p_adder',num2str(i)], 'n_inputs', n_inputs, ...
                                        'len', dec_rate, 'add_latency', add_latency}, ...
                                        {parallel_adder_sync_ins{i},parallel_adder_ins{i,:}}, ...
                                        {parallel_adder_sync_ins{i+1},parallel_adder_ins{i+1,:}});
                                        
                                        
        
    end
    
    decimator = xBlock(struct('source',str2func('parallel_downsampler_init_xblock'), ...
                                'name', ['decimator_',num2str(dec_rate)]), ...
                        {[blk, '/', 'decimator_',num2str(dec_rate)], 'dec_rate', dec_rate,'n_inputs', n_inputs,'input_clk_rate',input_clk_rate}, ...
                        {parallel_adder_ins{m+1,:},sync_in}, ...
                        [outports,{sync_out}]);
    
    
    
end

if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('nonrecursive\n%d_in\npolyphase:%s',n_bits,polyphase);
    set_param(blk,'AttributesFormatString',fmtstr);
end

end

function parallel_decN_stage_recursive_init_xblock(blk,n_stages,rate_change, n_inputs,add_latency, input_bits, bin_pt, reduced_crtc_path, input_clk_rate)
% rate_change is just dec_rate, as the xilinx CIC block use this
coeffs = zeros(1,n_stages+1);

for i = 1:n_stages+1
    coeffs(1,i) = nchoosek(n_stages,i-1);
end

n_bits = ceil(n_stages*log2(rate_change*1) + input_bits);
 
if n_inputs ==1 && strcmp(reduced_crtc_path, 'off')
    
    inport = xInport('in');
    outport = xOutport('out');
    
    on_off = {'off','on'};
    cic_blk = xBlock(struct('source','xrbsDSP_r4/CIC Filter','name','cic_blk'), ...
                     {'input_bitwidth', input_bits,  ...
                      'input_binpt', bin_pt, ...
                      'filter_type', 'Decimator', ...
                      'rate_change', rate_change, ...
                      'stages', n_stages, ...
                      'm', 1, ...
                      'pipeline', on_off{(add_latency>0)+1}}, ...
                      {inport}, ...
                      {outport});
                                                              
elseif n_inputs ==1 && strcmp(reduced_crtc_path, 'on')
    
    inport = xInport('in');
    outport = xOutport('out');
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');
    
    % draw integrator section
    int_ins = cell(1,n_stages+1);
    int_blks = cell(1,n_stages);
    int_ins{1} = inport;
    for i =1:n_stages
        int_ins{i+1} = xSignal(['int_in',num2str(i+1)]);
        int_blks{i} = xBlock(struct('srouce',@my_accumulator_init_xblock, 'name', ['integrator_',num2str(i)]), ...
                            {n_bits,bin_pt, reduced_crtc_path}, ...
                            int_ins(i), ...
                            int_ins(i+1));
    end
    
    % add down sampler
    ds_out = xSignal('ds_out');
    downsampler = xBlock(struct('source', str2func('downsample_init_xblock'), 'name', 'Down_sample1'), ...
                           {[blk,'/Down_sample1'], 'dec_rate', dec_rate,'input_clk_rate',input_clk_rate}, ...
                      [int_ins(n_stages+1),{sync_in}], ...
                      {ds_out,sync_out});
                  
    % add comb section
    comb_ins = cell(1,n_stages+1);
    comb_blks = cell(1,n_stages+1);
    comb_ins{1} = ds_out;
    for i = 1:n_stages
        comb_ins{i+1} = xSignal(['comb_in',num2str(i+1)]);
        comb_blks{i} = xBlock(struct('source',str2func('parallel_differentiator_init_xblock'), ...
                                    'name',['comb_',num2str(i)]), ...
                              {strcat(blk,['/comb_',num2str(i)]), ...
                               'n_inputs', 1,'n_bits', n_bits, 'bin_pt', bin_pt, 'diff_length', 1, 'latency', add_latency}, ...
                               {[], comb_ins{i}}, ...
                               {[], comb_ins{i+1}});   % the sync ports are kind of awkward
    end
    
    outport.bind(comb_ins{n_stages+1});
    
else
  
    inports = cell(1,n_inputs);
    outports = cell(1,n_inputs);
    
    for i =1:n_inputs
        inports{i} = xInport(['in',num2str(i)]);
        outports{i} = xOutport(['out',num2str(i)]);
    end
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');
    
    % add parallel integrator
    int_ins = cell(1,n_stages+1);
    int_blks = cell(1,n_stages);
    int_ins{1} = inports;
    int_sync_ins = cell(1,n_stages+1);
    int_sync_ins{1} = sync_in;
    for i =1:n_stages
        int_ins{i+1} = cell(1,n_inputs);
        for j =1:n_inputs
            int_ins{i+1}{j} = xSignal(['int_ins',num2str(i),'_',num2str(j)]);
        end
        int_sync_ins{i+1} = xSignal(['int_sync_ins', num2str(i)]);
        int_blks{i} = xBlock(struct('source',str2func('parallel_integrator_init_xblock'),...
                                    'name',['int_',num2str(i)]), ...
                              {strcat(blk,['/int_',num2str(i)]), ...
                                'n_inputs', n_inputs, ...
                                'n_bits', n_bits, ...
                                'bin_pt', bin_pt, ...
                                'add_latency', add_latency, ...
                                'reduced_crtc_path', reduced_crtc_path}, ...
                               [{sync_in}, int_ins{i}], ...
                               [int_sync_ins(i+1),int_ins{i+1}]);
    end
    
    % add down sampler
    ds_outs=cell(1,n_inputs);
    for i =1:n_inputs
        ds_outs{i}=xSignal(['ds_out',num2str(i)]);
    end
    ds_sync_out = xSignal('ds_sync_out');
    downsampler=xBlock(struct('source',str2func('parallel_downsampler_init_xblock'), ...
                                'name',['downsample_',num2str(rate_change)]),...
                        {[blk, '/', 'downsample_',num2str(rate_change)], 'dec_rate', rate_change, 'n_inputs', n_inputs, 'input_clk_rate',input_clk_rate}, ...
                        [int_ins{n_stages+1},{sync_in}],...
                        [ds_outs,{ds_sync_out}]);
    
    
    
    
    % and add comb section
    comb_ins = cell(1,n_stages+1);
    comb_blks = cell(1,n_stages);
    comb_ins{1} = ds_outs;
    comb_sync_ins = cell(1, n_stages+1);
    comb_sync_ins{1} = ds_sync_out;
    for i =1:n_stages
        %disp('a comb!');
        comb_ins{i+1} = cell(1,n_inputs);
        for j=1:n_inputs
            comb_ins{i+1}{j} = xSignal(['comb_ins',num2str(i),'_',num2str(j)]);
        end
        comb_sync_ins{i+1} = xSignal(['comb_sync_in', num2str(i+1)]);
        comb_blks{i} = xBlock(struct('source',str2func('parallel_differentiator_init_xblock'), ...
                                    'name',['comb_',num2str(i)]), ...
                              {strcat(blk,['/comb_',num2str(i)]), ...
                               'n_inputs', n_inputs, 'n_bits', n_bits, 'bin_pt', bin_pt, ...
                               'diff_length', 1, 'latency', add_latency}, ...
                               [comb_sync_ins(i), comb_ins{i}], ...
                               [comb_sync_ins(i+1), comb_ins{i+1}]);     % the sync ports are kind of awkward
    end
    
    for i =1:n_inputs
         outports{i}.bind(comb_ins{n_stages+1}{i});
    end
    sync_out.bind(comb_sync_ins{n_stages+1});
    
    %disp('hey');
    
    
end

if ~isempty(blk) && ~strcmp(blk(1),'/')  
    clean_blocks(blk);
    fmtstr=sprintf('recursive\n %d_in\n reducded critical path: %s',input_bits,  reduced_crtc_path);
    set_param(blk,'AttributesFormatString',fmtstr);
end

%disp('test');
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


function delay_max = find_delay_max(coefficients,add_latency)

len = length(coefficients);

const_bin_array = cell(1,len);
for i = 1:len
    const_bin_array{i} = dec2bin(coefficients(i));
end

delay_max = 0;
for i=1:len
    
    temp_len = length(find(const_bin_array{i} == '1'));
    temp_delay = ceil(log2(temp_len))*add_latency;
    if temp_delay > delay_max
        delay_max = temp_delay;
    end
end

end
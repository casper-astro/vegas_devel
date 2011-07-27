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
function parallel_polynomial_decN_stage_init_xblock(m,n,n_inputs,polyphase,add_latency,skip, n_bits, bin_pt, recursive)

coeffs = cic_coefficient_generator(m,n);

if skip == 0
    skip_setting = 'Last Value of Frame  (most efficient)';
else
end
 

if n_inputs==1 && strcmp(polyphase,'off') % only one input
    
    % non-polyphase structure
    sync =xInport('sync');
    sync_out=xOutport('sync_out');
    inport = xInport('in');
    outport = xOutport('out');
    
    delay_blks = cell(m,n-1);
    delay_inports = cell(m+1);
    delay_outports = cell(m,n);
    adder_tree_blks = cell(m,1);
    adder_tree_sync_in = cell(m+1);
    delay_inports{1} = inport;
    adder_tree_sync_in{1} = sync;
    for i = 1:m
        adder_tree_sync_in{i+1}= xSignal(['adder_tree_sync_in',num2str(i+1)]);
        delay_inports{i+1} = xSignal(['delay_in',num2str(i),'_1']);
        delay_outports{i,1} = delay_inports{i};
        for j = 1:n-1
            delay_outports{i,j+1} = xSignal(['delay_out',num2str(i),'_',num2str(j+1)]);
            delay_blks{i,j} = xBlock(struct('source','Delay','name', ['delay',num2str(i),'_',num2str(j)]), ...
                              struct('latency', 1), ...   
                              {delay_outports{i,j}}, ...
                              {delay_outports{i,j+1}});
        end
        adder_tree_blks{i} =  xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', ['adder_tree',num2str(i)]), ...
                     {n, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                     [{adder_tree_sync_in{i}},{delay_outports{i,:}}], ...
                     {adder_tree_sync_in{i+1},delay_inports{i+1}});
    end
    
    downsampler = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', 'Down_sample1'), ...
                               struct('sample_ratio',n, ...
                                      'sample_phase',skip_setting, ...
                                      'latency', 1), ...
                          {delay_inports{m+1}}, ...
                          {outport}); 
    final_delay = xBlock(struct('source','Delay','name', 'final_delay'), ...
                              struct('latency', 1), ...   
                              {adder_tree_sync_in{m+1}}, ...
                              {sync_out});


elseif n_inputs ==1 && strcmp(polyphase,'on')
    
     % polyphase structure
    sync =xInport('sync');
    sync_out=xOutport('sync_out');
    inport = xInport('in');
    outport = xOutport('out');
    
    delay_blks = cell(1,n-1);
    delay_ins = cell(1,n);
    delay_ins{1} = inport;
    for i =1:n-1
        delay_ins{i+1} = xSignal(['delay',num2str(i+1)]);
        delay_blks{i} = xBlock(struct('source','Delay','name',['delay',num2str(i)]), ...
                              struct('latency', 1), ...
                              {delay_ins{i}}, ...
                              {delay_ins{i+1}});
    end
                          
    ds_out = cell(1,n);
    downsampling = cell(1,n);
    for i =1:n
        ds_out{i} = xSignal(['downsample_out',num2str(i)]);
        downsampling{i} = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', ['Down_sample',num2str(i)]), ...
                               struct('sample_ratio',n, ...
                                        'sample_phase','Last Value of Frame  (most efficient)', ...
                                      'latency', 1), ...
                                 {delay_ins{i}}, ...
                                 {ds_out{i}});  
    end
    
    
    poly_out = cell(1,n);
    polynomial_blk = cell(1,n);
    sync_poly = xSignal('sync_poly');
    poly_out{1} = xSignal('poly_out1');
    delay_max = find_delay_max(coeffs,add_latency);
    polynomial_blk{1} = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial0'), ...
                            {coeffs(1:n:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                            {ds_out{1}, sync}, ...
                            {poly_out{1}, sync_poly});  
    for i =2:n
        poly_out{i}= xSignal(['poly_out',num2str(i)]);
        polynomial_blk{i} = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name',['polynomial',num2str(i)]), ...
                            {coeffs(i:n:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                            {ds_out{i}, sync}, ...
                            {poly_out{i}, []}); 
    end
                        
    adder_tree_sync_out = xSignal('adder_tree_sync_out');
    final_adder_tree =xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', 'adder_tree'), ...
                     {n, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                     [{sync_poly},poly_out], ...
                     {adder_tree_sync_out,outport});  
    % take care of sync
    sync_delay = xBlock(struct('source','Delay','name', 'sync_delay'), ...
                              struct('latency', add_latency + 1), ...   % compensate the downsample delay
                              {adder_tree_sync_out}, ...
                              {sync_out});
   
else
    
    
    
    inports= cell(1,n_inputs);
    outports = cell(1,n_inputs);
    sync_in = xInport('sync_in');
    sync_out = xOutport('sync_out');
    for i =1:n_inputs
        inports{i} = xInport(['in',num2str(i)]);
        outports{i} = xOutport(['out',num2str(i)]);
    end
    
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
        parallel_adder_blks{i} = xBlock(struct('source',str2func('parallel_adder_init_xblock'),'name',['p_adder',num2str(i)]), ...
                                        {n_inputs,n,add_latency}, ...
                                        {parallel_adder_sync_ins{i},parallel_adder_ins{i,:}}, ...
                                        {parallel_adder_sync_ins{i+1},parallel_adder_ins{i+1,:}});
                                        
                                        
        
    end
    
    decimator = xBlock(struct('source',str2func('parallel_filter_init_xblock'), 'name', ['decimator_',num2str(n)]), ...
                        {n, n_inputs}, ...
                        {parallel_adder_sync_ins{m+1},parallel_adder_ins{m+1,:}}, ...
                        [{sync_out}, outports]);
    
    
    
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
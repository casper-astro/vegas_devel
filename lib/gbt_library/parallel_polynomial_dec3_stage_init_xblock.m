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
function parallel_polynomial_dec3_stage_init_xblock(m,n_inputs,polyphase,add_latency,skip, n_bits, bin_pt, recursive)

coeffs = cic_coefficient_generator(m,3);

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
    
    delay_blks = cell(m,2);
    delay_inports = cell(m+1);
    delay_outports = cell(m,2);
    adder_tree_blks = cell(m,1);
    adder_tree_sync_in = cell(m+1);
    delay_inports{1} = inport;
    adder_tree_sync_in{1} = sync;
    for i = 1:m
        adder_tree_sync_in{i+1}= xSignal(['adder_tree_sync_in',num2str(i+1)]);
        delay_inports{i+1} = xSignal(['delay_in',num2str(i),'_1']);
        delay_outports{i,1} = xSignal(['delay_out',num2str(i),'_1']);
        delay_outports{i,2} = xSignal(['delay_out',num2str(i),'_2']);
        delay_blks{i,1} = xBlock(struct('source','Delay','name', ['delay',num2str(i),'_1']), ...
                              struct('latency', 1), ...   
                              {delay_inports{i}}, ...
                              {delay_outports{i,1}});
        delay_blks{i,2} = xBlock(struct('source','Delay','name', ['delay',num2str(i),'_2']), ...
                              struct('latency', 1), ...   
                              {delay_outports{i,1}}, ...
                              {delay_outports{i,2}});
        adder_tree_blks{i} =  xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', ['adder_tree',num2str(i)]), ...
                     {3, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                     {adder_tree_sync_in{i},delay_inports{i},delay_outports{i,1},delay_outports{i,2}}, ...
                     {adder_tree_sync_in{i+1},delay_inports{i+1}});
    end
    
    downsampler = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', 'Down_sample1'), ...
                               struct('sample_ratio',3, ...
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
    
    first_delay_out = xSignal('delay1');
    first_delay = xBlock(struct('source','Delay','name', 'first_delay'), ...
                              struct('latency', 1), ...
                              {inport}, ...
                              {first_delay_out});
    second_delay_out = xSignal('delay2');
    second__delay = xBlock(struct('source','Delay','name', 'second_delay'), ...
                              struct('latency', 1), ...
                              {first_delay_out}, ...
                              {second_delay_out});                         
                          
    ds_out0 =xSignal('downsample_out0');
    ds_out1 = xSignal('downsample_out1');
    ds_out2 = xSignal('downsample_out2');
    downsampling0 = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', 'Down_sample0'), ...
                               struct('sample_ratio',3, ...
                                        'sample_phase','Last Value of Frame  (most efficient)', ...
                                      'latency', 1), ...
                          {inport}, ...
                          {ds_out0});  
    downsampling1 = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', 'Down_sample1'), ...
                               struct('sample_ratio',3, ...
                                        'sample_phase','Last Value of Frame  (most efficient)', ...
                                      'latency', 1), ...
                          {first_delay_out}, ...
                          {ds_out1});  
    downsampling2 = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', 'Down_sample2'), ...
                               struct('sample_ratio',3, ...
                                        'sample_phase','Last Value of Frame  (most efficient)', ...
                                      'latency', 1), ...
                          {second_delay_out}, ...
                          {ds_out2}); 
    
    delay_max = find_delay_max(coeffs,add_latency);                  
    poly_out0 = xSignal('poly_out0');
    poly_out1 = xSignal('poly_out1');
    poly_out2 = xSignal('poly_out2');
    sync_poly = xSignal('sync_poly');
    polynomial_blk0 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial0'), ...
                            {coeffs(1:3:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                            {ds_out0, sync}, ...
                            {poly_out0, sync_poly});      
                        
    polynomial_blk1 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial1'), ...
                            {coeffs(2:3:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                            {ds_out1, sync}, ...
                            {poly_out1,[]});    
                        
    polynomial_blk2 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial2'), ...
                            {coeffs(3:3:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                            {ds_out2, sync}, ...
                            {poly_out2,[]}); 
                        
    adder_tree_sync_out = xSignal('adder_tree_sync_out');
    final_adder_tree =xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', 'adder_tree'), ...
                     {3, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                     {sync_poly,poly_out0, poly_out1,poly_out2}, ...
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
                                        {n_inputs,3,add_latency}, ...
                                        {parallel_adder_sync_ins{i},parallel_adder_ins{i,:}}, ...
                                        {parallel_adder_sync_ins{i+1},parallel_adder_ins{i+1,:}});
                                        
                                        
        
    end
    
    decimator = xBlock(struct('source',str2func('parallel_filter_init_xblock'), 'name', 'decimator_3'), ...
                        {3, n_inputs}, ...
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
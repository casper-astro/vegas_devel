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
function parallel_polynomial_dec2_stage_init_xblock(m,n_inputs,polyphase,add_latency,skip, n_bits, bin_pt)

coeffs = zeros(1,m+1);
for i = 1:m+1
    coeffs(1,i) = nchoosek(m,i-1);
end
 

if n_inputs==1 && strcmp(polyphase,'off') % only one input
    
    % non-polyphase structure
    inport = xInport('in');
    outport = xOutport('out');
    sync =xInport('sync');
    sync_out=xOutport('sync_out');
    
    polynomial_blk = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial'), ...
                            {coeffs, add_latency,n_bits, bin_pt,'off'}, ...
                            {inport, sync}, ...
                            {outport, sync_out});


elseif n_inputs ==1 && strcmp(polyphase,'on')
    % polyphase structure
    inport = xInport('in');
    outport = xOutport('out');
    sync =xInport('sync');
    sync_out=xOutport('sync_out');
    
    first_delay_out = xSignal('delay1');
    first_delay = xBlock(struct('source','Delay','name', 'first_delay'), ...
                              struct('latency', 1), ...
                              {inport}, ...
                              {first_delay_out});
                          
                          
    ds_out0 =xSignal('downsample_out0');
    ds_out1 = xSignal('downsample_out1');
    downsampling0 = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', 'Down_sample0'), ...
                               struct('sample_ratio',2, ...
                                        'sample_phase','Last Value of Frame  (most efficient)', ...
                                      'latency', 1), ...
                          {inport}, ...
                          {ds_out0});  
    downsampling1 = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', 'Down_sample1'), ...
                               struct('sample_ratio',2, ...
                                        'sample_phase','Last Value of Frame  (most efficient)', ...
                                      'latency', 1), ...
                          {first_delay_out}, ...
                          {ds_out1}); 
                      
    poly_out0 = xSignal('poly_out0');
    poly_out1 = xSignal('poly_out1');
    sync_poly = xSignal('sync_poly');
    polynomial_blk0 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial0'), ...
                            {coeffs(1:2:end), add_latency,n_bits, bin_pt,'off'}, ...
                            {ds_out0, sync}, ...
                            {poly_out0, sync_poly});      
                        
    polynomial_blk1 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial1'), ...
                            {coeffs(2:2:end), add_latency,n_bits, bin_pt,'off'}, ...
                            {ds_out1, sync}, ...
                            {poly_out1,[]});                              

    final_adder = xBlock(struct('source','AddSub', 'name', 'final_adder'), ...
                             struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                    'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                    {poly_out0,poly_out1}, ...
                                    {outport});  
    % take care of sync
    sync_delay = xBlock(struct('source','Delay','name', 'sync_delay'), ...
                              struct('latency', add_latency + 1), ...   % compensate the downsample delay
                              {sync_poly}, ...
                              {sync_out});
elseif n_inputs ==2
    inport0  = xInport('in1');
    inport1 = xInport('in2');
    sync = xInport('sync');
    
    outport0 = xOutport('out0');
    outport1 = xOutport('out1');
    sync_out = xOutport('sync_out');    
    
    poly_out1_even = xSignal('poly_ou1_even');
    poly_out1_odd = xSignal('poly_out1_odd');
    poly_out0_odd = xSignal('poly_out0_odd');
    poly_out0_even = xSignal('poly_out0_even');
    sync_poly = xSignal('sync_poly');
    polynomial_blk0 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial0'), ...
                            {coeffs, add_latency,n_bits, bin_pt,'on'}, ...
                            {inport0, sync}, ...
                            {poly_out0_odd,poly_out0_even,sync_poly}); 
    polynomial_blk1= xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial1'), ...
                            {coeffs, add_latency,n_bits, bin_pt,'on'}, ...
                            {inport1, sync}, ...
                            {poly_out1_odd,poly_out1_even,[]}); 
                        
    d_poly_out1_even = xSignal('d_poly_out1_even');                 
    delay_1 = xBlock(struct('source','Delay','name', 'delay_1'), ...
                              struct('latency', 1), ...  
                              {poly_out1_even}, ...
                              {d_poly_out1_even});
                          
    final_adder1 = xBlock(struct('source','AddSub', 'name', 'final_adder1'), ...
                             struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                    'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                    {poly_out0_even,poly_out1_odd}, ...
                                    {outport1});  
    final_adder0 = xBlock(struct('source','AddSub', 'name', 'final_adder0'), ...
                             struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                    'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                    {poly_out0_odd,d_poly_out1_even}, ...
                                    {outport0});  
    % take care of sync
    sync_delay = xBlock(struct('source','Delay','name', 'sync_delay'), ...
                              struct('latency', add_latency ), ... 
                              {sync_poly}, ...
                              {sync_out});
                          
                          
elseif n_inputs >m  % think about this later
            
    inports = cell(1,n_inputs);
    for i = 1:n_inputs,
        inports{i} = xInport(['in',num2str(i)]);
    end
    sync = xInport('sync');

    outports = cell(1,n_inputs);
    for i=1+mod(skip,2):2:n_inputs,
        outports{i} = xOutport(['out',num2str(i)]);
    end
    sync_out = xOutport('sync_out');   
    
    
    
    % do the multiplications
    odd_coeffs = coeffs(1:2:end);
    even_coeffs = coeffs(2:2:end);
    mult_cxi_out = cell(1,n_inputs);
    mult_cxi_blks = cell(1,n_inputs);
    sync_mult = xSignal('sync_mult');
    for i =1:n_inputs
        if mod(i,2) == skip  % it means y_i(n) will be skipped, thus take the even coefficients
            mult_cxi_out{i} = cell(1,m+1);
            for j=2:2:m+1
                mult_cxi_out{i}{j} = xSignal(['mult_cxi_out',num2str(i),'_',num2str(j)]);
            end
            mult_cxi_blks{i} = xBlock(struct('source',str2func('shift_mult_array_init_xblock'), 'name', ['mult_cxi_blk',num2str(i)]), ...
                      {even_coeffs, add_latency, n_bits, bin_pt}, ...
                      {inports{i},sync}, ...
                      {mult_cxi_out{i}{2:2:end},[]}); 
        else  % take the odd coefficients
            mult_cxi_out{i} = cell(1,m+1);
            for j=1:2:m+1
                mult_cxi_out{i}{j} = xSignal(['mult_cxi_out',num2str(i),'_',num2str(j)]);
            end
            if i ==1
                mult_cxi_blks{i} = xBlock(struct('source',str2func('shift_mult_array_init_xblock'), 'name', ['mult_cxi_blk',num2str(i)]), ...
                          {odd_coeffs, add_latency, n_bits, bin_pt}, ...
                          {inports{i},sync}, ...
                          {mult_cxi_out{i}{1:2:end},sync_mult});
            else
                mult_cxi_blks{i} = xBlock(struct('source',str2func('shift_mult_array_init_xblock'), 'name', ['mult_cxi_blk',num2str(i)]), ...
                          {odd_coeffs, add_latency, n_bits, bin_pt}, ...
                          {inports{i},sync}, ...
                          {mult_cxi_out{i}{1:2:end},[]}); 
            end
        end
    end
    
    
    % add delays
    d_mult_cxi_out = cell(1,n_inputs);
    delay_blks = cell(1,m-1);
    for i = (n_inputs - m +1) : n_inputs
        d_mult_cxi_out{i} = cell(1,m+1);
        delay_blks{i} = cell(1,m+1);
        k = i-(n_inputs - m +1);
        for j=m+1-k:(m+1)
            if mod(i,2) == skip   % coefficient take the even
                if mod(j,2) == 1 
                    continue;
                else
                    d_mult_cxi_out{i}{j} = xSignal(['d_mult_cxi_out',num2str(i),'_',num2str(j)]);
                    delay_blks{k+1}{j} = xBlock(struct('source','Delay','name', ['delay',num2str(k+1),'_',num2str(j)]), ...
                                                  struct('latency', add_latency), ...
                                                  {mult_cxi_out{i}{j}}, ...
                                                  {d_mult_cxi_out{i}{j}});

                end
            else  % coefficient take the odd
                if mod(j,2) == 0
                    continue;
                else
                    d_mult_cxi_out{i}{j} = xSignal(['d_mult_cxi_out',num2str(i),'_',num2str(j)]);
                    delay_blks{k+1}{j} = xBlock(struct('source','Delay','name', ['delay',num2str(k+1),'_',num2str(j)]), ...
                                                  struct('latency', add_latency), ...
                                                  {mult_cxi_out{i}{j}}, ...
                                                  {d_mult_cxi_out{i}{j}});                    
                end
            end
        end
    end
        
    % organize the matrix
    mat = cell(n_inputs,m+1);
    ref_mat = cell(n_inputs,m+1); % displayable, for debugging
    
    % top right triangle 
    % (correspond to the bottom right triangle in the
    % old matrix)
    for i = 1+mod(skip,2):2:m
        for j = 1+i:m+1
            k = j-(1+i);
            mat{i,j} = d_mult_cxi_out{n_inputs-k}{j};
            ref_mat{i,j} = [num2str(n_inputs-k),'_',num2str(j)];
        end
    end
    
    
    % middle parallelogram
    for i= 1+mod(skip,2):2:(n_inputs-m)
        for j = 1:2:m+1
            mat{i+j-1,j} = mult_cxi_out{i}{j};
            ref_mat{i+j-1,j} = [num2str(i),'_',num2str(j)];
        end
    end
    
    
    % bottom triangle
    for i = (n_inputs - m+1):m
        k = i - (n_inputs - m+1);
        for j = 1:(m-k)
            if mod(i+j-1,2) == skip
                continue;
            else
                mat{i+j-1,j} = mult_cxi_out{i}{j};
                ref_mat{i+j-1,j} = [num2str(i),'_',num2str(j)];
            end
        end
    end

    % adder_trees for all the outputs
    adder_trees_blks = cell(1,n_inputs);
    adder_trees_blks{1+mod(skip,2)} = xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', ['adder_tree',num2str(1+mod(skip,2))]), ...
                     {m+1, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                     {sync_mult,mat{1+mod(skip,2),:}}, ...
                     {sync_out,outports{1+mod(skip,2)}});
    for i= 1+mod(skip,2)+2:2:n_inputs
        adder_trees_blks{i} = xBlock(struct('source', str2func('adder_tree_init_xblock'),'name', ['adder_tree',num2str(i)]), ...
                             {m+1, add_latency, 'Round  (unbiased: +/- Inf)', 'Saturate', 'Behavioral'}, ...
                             {sync_mult,mat{i,:}}, ...
                             {[],outports{i}});
    end
    
    ref_mat
end


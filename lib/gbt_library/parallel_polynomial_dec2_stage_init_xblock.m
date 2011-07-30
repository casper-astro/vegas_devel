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
function parallel_polynomial_dec2_stage_init_xblock(m,n_inputs,polyphase,add_latency,skip, n_bits, bin_pt,dec2_halfout, recursive)

coeffs = zeros(1,m+1);
for i = 1:m+1
    coeffs(1,i) = nchoosek(m,i-1);
end
 

if recursive
    if reduced_pipeline
    else
    end
else
    if n_inputs==1 && strcmp(polyphase,'off') % only one input

        % non-polyphase structure
        sync =xInport('sync');
        sync_out=xOutport('sync_out');
        inport = xInport('in');
        outport = xOutport('out');

        poly_out = xSignal('poly_out');
        polynomial_blk = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial'), ...
                                {coeffs, add_latency,n_bits, bin_pt,'off',0}, ...
                                {inport, sync}, ...
                                {poly_out, sync_out});

        downsampling0 = xBlock(struct('source', 'xbsBasic_r4/Down Sample', 'name', 'Down_sample0'), ...
                                   struct('sample_ratio',2, ...
                                            'sample_phase','Last Value of Frame  (most efficient)', ...
                                          'latency', 1), ...
                              {poly_out}, ...
                              {outport}); 

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

        delay_max = find_delay_max(coeffs,add_latency);
        polynomial_blk0 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial0'), ...
                                {coeffs(1:2:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
                                {ds_out0, sync}, ...
                                {poly_out0, sync_poly});      

        polynomial_blk1 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial1'), ...
                                {coeffs(2:2:end), add_latency,n_bits, bin_pt,'off',delay_max}, ...
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
    elseif n_inputs ==2 && strcmp(polyphase,'on')
        sync = xInport('sync');
        sync_out = xOutport('sync_out'); 
        inport0  = xInport('in1');
        inport1 = xInport('in2');   
        outport1 = xOutport('out1');

        poly_out1_even = xSignal('poly_ou1_even');
        poly_out1_odd = xSignal('poly_out1_odd');
        poly_out0_odd = xSignal('poly_out0_odd');
        poly_out0_even = xSignal('poly_out0_even');
        sync_poly = xSignal('sync_poly');
        polynomial_blk0 = xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial0'), ...
                                {coeffs, add_latency,n_bits, bin_pt,'on',0}, ...
                                {inport0, sync}, ...
                                {poly_out0_odd,poly_out0_even,sync_poly}); 
        polynomial_blk1= xBlock(struct('source',str2func('polynomial_shift_mult_transpose_init_xblock'),'name','polynomial1'), ...
                                {coeffs, add_latency,n_bits, bin_pt,'on',0}, ...
                                {inport1, sync}, ...
                                {poly_out1_odd,poly_out1_even,[]}); 

        final_adder1 = xBlock(struct('source','AddSub', 'name', 'final_adder1'), ...
                                 struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                        'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                        {poly_out0_even,poly_out1_odd}, ...
                                        {outport1});  
        % take care of sync
        sync_delay = xBlock(struct('source','Delay','name', 'sync_delay'), ...
                                  struct('latency', add_latency ), ... 
                                  {sync_poly}, ...
                                  {sync_out});

    elseif strcmp(dec2_halfout,'on')


        if mod(n_inputs,2)~=0
            disp('only supports even number of inputs at this mode');
            return;
        end
        inports= cell(1,n_inputs);
        outports = cell(1,n_inputs/2);
        sync_in = xInport('sync_in');
        sync_out = xOutport('sync_out');
        for i =1:n_inputs
            inports{i} = xInport(['in',num2str(i)]);
        end
        for i=1:n_inputs/2
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
                                            {n_inputs,2,add_latency}, ...
                                            {parallel_adder_sync_ins{i},parallel_adder_ins{i,:}}, ...
                                            {parallel_adder_sync_ins{i+1},parallel_adder_ins{i+1,:}});



        end

        sync_out.bind(parallel_adder_sync_ins{m+1});

        for i =1:n_inputs/2
            outports{i}.bind(parallel_adder_ins{m+1,i*2-skip});
        end


    else  % dec2_halfout not true, n_outputs = n_inputs
        disp('to be implemented');


    end
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
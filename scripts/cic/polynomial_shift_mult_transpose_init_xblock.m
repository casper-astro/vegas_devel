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
function polynomial_shift_mult_transpose_init_xblock(coeffs, add_latency, n_bits, bin_pt,oddeven)

if strcmp(oddeven,'off')
    inport = xInport('in');
    sync =xInport('sync');
    outport = xOutport('out');
    sync_out=xOutport('sync_out');

    len = length(coeffs);

    sync_mult = xSignal('sync_mult');
    mult_outs = cell(1,len);
    for j = 1:len
        mult_outs{j} = xSignal(['mult_out', num2str(j)]);
    end

    % shift multiplication with coefficients
    mult_blk = xBlock(struct('source',str2func('shift_mult_array_init_xblock'), 'name', 'shift_mult_array'), ...
                      {coeffs(end:-1:1), add_latency, n_bits, bin_pt}, ...
                      {inport,sync}, ...
                      [mult_outs,{sync_mult}]); 

    adder_blks = cell(1,len-1);
    delay_blks = cell(1,len-1);
    delay_ins = cell(1,len);
    delay_outs = cell(1,len-1);
    delay_ins{1} = mult_outs{1};
    for j=1:len-1
        delay_outs{j} = xSignal(['delay_out',num2str(j)]);
        delay_ins{j+1} = xSignal(['delay_in',num2str(j+1)]);
        delay_blks{j} =  xBlock(struct('source','Delay','name', ['delay',num2str(j)]), ...
                              struct('latency', 1), ...
                              {delay_ins{j}}, ...
                              {delay_outs{j}});
        adder_blks{j} = xBlock(struct('source','AddSub', 'name', ['adder_blk',num2str(j)]), ...
                                 struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                        'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                        {delay_outs{j},mult_outs{j+1}}, ...
                                        {delay_ins{j+1}});                            
    end

    outport.bind(delay_ins{len});
    % take care of sync
    sync_delay1 = xBlock(struct('source','Delay','name', 'sync_delay1'), ...
                              struct('latency', add_latency), ...
                              {sync_mult}, ...
                              {sync_out});

else
    inport = xInport('in');
    sync =xInport('sync');
    outport_odd = xOutport('out_odd');
    outport_even = xOutport('out_even');
    sync_out=xOutport('sync_out');


    len = length(coeffs);

    sync_mult = xSignal('sync_mult');
    mult_outs = cell(1,len);
    for j = 1:len
        mult_outs{j} = xSignal(['mult_out', num2str(j)]);
    end

    % shift multiplication with coefficients
    mult_blk = xBlock(struct('source',str2func('shift_mult_array_init_xblock'), 'name', 'shift_mult_array'), ...
                      {coeffs(end:-1:1), add_latency, n_bits, bin_pt}, ...
                      {inport,sync}, ...
                      [mult_outs,{sync_mult}]); 
                  
    % sort out blocks
    % pay attention to the flipping
    mult_odd_outs = fliplr({mult_outs{1:2:end}});
    odd_len = length(mult_odd_outs);
    mult_even_outs = fliplr({mult_outs{2:2:end}});
    even_len = length(mult_even_outs);
    
    odd_adder_blks = cell(1,odd_len-1);
    odd_delay_blks = cell(1,odd_len-1);
    odd_delay_ins = cell(1,odd_len);
    odd_delay_outs = cell(1,odd_len-1);
    odd_delay_ins{1} = mult_odd_outs{1};
    for j=1:odd_len-1
        odd_delay_outs{j} = xSignal(['odd_delay_out',num2str(j)]);
        odd_delay_ins{j+1} = xSignal(['odd_delay_in',num2str(j+1)]);
        odd_delay_blks{j} =  xBlock(struct('source','Delay','name', ['odd_delay',num2str(j)]), ...
                              struct('latency', 1), ...
                              {odd_delay_ins{j}}, ...
                              {odd_delay_outs{j}});
        odd_adder_blks{j} = xBlock(struct('source','AddSub', 'name', ['odd_adder_blk',num2str(j)]), ...
                                 struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                        'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                        {odd_delay_outs{j},mult_odd_outs{j+1}}, ...
                                        {odd_delay_ins{j+1}});                            
    end
    
    even_adder_blks = cell(1,even_len-1);
    even_delay_blks = cell(1,even_len-1);
    even_delay_ins = cell(1,even_len);
    even_delay_outs = cell(1,even_len-1);
    even_delay_ins{1} = mult_even_outs{1};
    for j=1:even_len-1
        even_delay_outs{j} = xSignal(['even_delay_out',num2str(j)]);
        even_delay_ins{j+1} = xSignal(['even_delay_in',num2str(j+1)]);
        even_delay_blks{j} =  xBlock(struct('source','Delay','name', ['even_delay',num2str(j)]), ...
                              struct('latency', 1), ...
                              {even_delay_ins{j}}, ...
                              {even_delay_outs{j}});
        even_adder_blks{j} = xBlock(struct('source','AddSub', 'name', ['even_adder_blk',num2str(j)]), ...
                                 struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                        'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                        {even_delay_outs{j},mult_even_outs{j+1}}, ...
                                        {even_delay_ins{j+1}});                            
    end

    outport_odd.bind(odd_delay_ins{odd_len});    
    outport_even.bind(even_delay_ins{even_len}); 
    
    % take care of sync
    sync_delay1 = xBlock(struct('source','Delay','name', 'sync_delay1'), ...
                              struct('latency', add_latency), ...
                              {sync_mult}, ...
                              {sync_out});
end
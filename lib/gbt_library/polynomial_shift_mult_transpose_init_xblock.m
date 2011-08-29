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
function polynomial_shift_mult_transpose_init_xblock(blk, coeffs, add_latency, n_bits, bin_pt,oddeven,delay_max)

if add_latency ~= 1
    disp('only supports add_latency == 1 at this moment');
    add_latency = 1;
end

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
                      {[blk,'/shift_mult_array'], coeffs(end:-1:1), add_latency, n_bits, bin_pt,delay_max}, ...
                      {inport,sync}, ...
                      [mult_outs,{sync_mult}]); 

    adder_blks = cell(1,len-1);
    adder_ins = cell(1,len);
    adder_ins{1} = xSignal('adder_ins1');
    delay_blk = xBlock(struct('source','Delay','name', 'delay0'), ...
                              struct('latency', 1), ...
                              {mult_outs{1}}, ...
                              {adder_ins{1}});
    for j=1:len-1
        adder_ins{j+1} = xSignal(['adder_ins',num2str(j+1)]);
        adder_blks{j} = xBlock(struct('source','AddSub', 'name', ['adder_blk',num2str(j)]), ...
                                 struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                        'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                        {adder_ins{j},mult_outs{j+1}}, ...
                                        {adder_ins{j+1}});                            
    end
    

    
    outport.bind(adder_ins{len});                       
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
                      {[blk, '/shift_mult_array'], coeffs(end:-1:1), add_latency, n_bits, bin_pt,delay_max}, ...
                      {inport,sync}, ...
                      [mult_outs,{sync_mult}]); 
                  
    % sort out blocks
    % pay attention to the flipping
    mult_odd_outs = fliplr({mult_outs{1:2:end}});
    odd_len = length(mult_odd_outs);
    mult_even_outs = fliplr({mult_outs{2:2:end}});
    even_len = length(mult_even_outs);
    
    odd_adder_blks = cell(1,odd_len-1);
    odd_adder_ins = cell(1,odd_len);
    odd_adder_ins{1} = xSignal('odd_adder_ins1');
    odd_delay_blk = xBlock(struct('source','Delay','name', 'odd_delay'), ...
                              struct('latency', 1), ...
                              {mult_odd_outs{1}}, ...
                              {odd_adder_ins{1}});
    for j=1:odd_len-1
        odd_adder_ins{j+1} = xSignal(['odd_adder_in',num2str(j+1)]);
        odd_adder_blks{j} = xBlock(struct('source','AddSub', 'name', ['odd_adder_blk',num2str(j)]), ...
                                 struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                        'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                        {odd_adder_ins{j},mult_odd_outs{j+1}}, ...
                                        {odd_adder_ins{j+1}});                            
    end
    
    even_adder_blks = cell(1,even_len-1);
    even_adder_ins = cell(1,even_len);
    even_adder_ins{1} = xSignal('even_adder_ins1');
    even_delay_blk = xBlock(struct('source','Delay','name', 'even_delay'), ...
                              struct('latency', 1), ...
                              {mult_even_outs{1}}, ...
                              {even_adder_ins{1}});
    for j=1:even_len-1
        even_adder_ins{j+1} = xSignal(['even_adder_in',num2str(j+1)]);
        even_adder_blks{j} = xBlock(struct('source','AddSub', 'name', ['even_adder_blk',num2str(j)]), ...
                                 struct('mode', 'Addition', 'latency', add_latency, 'precision', 'Full', ...
                                        'use_behavioral_HDL', 'off', 'hw_selection', 'Fabric'), ...
                                        {even_adder_ins{j},mult_even_outs{j+1}}, ...
                                        {even_adder_ins{j+1}});                            
    end


    outport_odd.bind(odd_adder_ins{odd_len});
    outport_even.bind(even_adder_ins{even_len});
    
    
    % take care of sync
    sync_delay1 = xBlock(struct('source','Delay','name', 'sync_delay1'), ...
                              struct('latency', add_latency), ...
                              {sync_mult}, ...
                              {sync_out});


end

if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
end
end
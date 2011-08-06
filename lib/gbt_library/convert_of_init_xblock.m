%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://casper.berkeley.edu                                                %      
%   Copyright (C) 2011 Suraj Gowda                                            %
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
function convert_of_init_xblock(bit_width_i, binary_point_i, bit_width_o, binary_point_o, latency, ...
	overflow, quantization)

%% convert din to dout 
din = xInport('din');
dout = xOutport('dout');
convert = xBlock(struct('source', 'Convert', 'name', 'convert'), ...
                        struct('n_bits', bit_width_o, 'bin_pt', binary_point_o, 'latency', latency, ...
                               'pipeline', 'on', 'quantization', quantization, 'overflow', overflow), {din}, {dout});


%% compute overflow signal
of = xOutport('of');

wb_lost = max((bit_width_i - binary_point_i) - (bit_width_o - binary_point_o),0);

%for case where no overflow issues
if wb_lost == 0,
	xBlock( struct('source', 'Constant', 'name', 'no_of'), ...
			struct('arith_type', 'Boolean', 'const','0', 'explicit_period', 'on', 'period', '1' ), ...
			{}, {of} );
else
	inverted_bits = {};
	verted_bits = {};
	for k = 1:wb_lost+1
		k_th_bit = xSignal;
		k_th_bit_neg = xSignal;
		
		% slice off the (k-1)th bit (MSB ref)
		xBlock( struct('source', 'Slice', 'name', ['slice_', num2str(k)]), ...
				struct('boolean_output', 'on', 'bit1', -(k-1)), {din}, {k_th_bit} );	
		
		% connect the kth bit to all 1's directly
		verted_bits{k} = k_th_bit;
		
		% negate k_th_bit
		xBlock( struct('source', 'Inverter', 'name', ['inv_', num2str(k)]), [], {k_th_bit}, {k_th_bit_neg} );
		
		% connect kth bit negated to all 0's
		inverted_bits{k} = k_th_bit_neg;
	end
	
	
	all_0s_out1 = xSignal;
	all_1s_out1 = xSignal;
	all_0s = xBlock(struct('source', 'Logical', 'name', 'all_0s'), ...
						   struct('logical_function', 'NAND', 'latency', latency, 'inputs', wb_lost+1), ...
						   inverted_bits, {all_0s_out1});
						   
	all_1s = xBlock(struct('source', 'Logical', 'name', 'all_1s'), ...
						   struct('logical_function', 'NAND', 'latency', latency, 'inputs', wb_lost+1), ...
						   verted_bits, {all_1s_out1});
	
	% AND both detectors together
	and = xBlock(struct('source', 'Logical', 'name', 'and'), [], {all_0s_out1, all_1s_out1}, {of});

end


end


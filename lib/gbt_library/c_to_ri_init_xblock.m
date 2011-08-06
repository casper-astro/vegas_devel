%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://casper.berkeley.edu                                                %      
%   Copyright (C) 2011                Hong Chen                               %
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
function c_to_ri_init_xblock(n_bits, bin_pt)
%% inports
c = xInport('c');

%% outports
re = xOutport('re');
im = xOutport('im');

%% diagram

% block: untitled1/c_to_ri/force_im
slice_im_out1 = xSignal;
force_im = xBlock(struct('source', 'Reinterpret', 'name', 'force_im'), ...
                         struct('force_arith_type', 'on', ...
                                'arith_type', 'Signed  (2''s comp)', ...
                                'force_bin_pt', 'on', ...
                                'bin_pt', bin_pt), ...
                         {slice_im_out1}, ...
                         {im});

% block: untitled1/c_to_ri/force_re
slice_re_out1 = xSignal;
force_re = xBlock(struct('source', 'Reinterpret', 'name', 'force_re'), ...
                         struct('force_arith_type', 'on', ...
                                'arith_type', 'Signed  (2''s comp)', ...
                                'force_bin_pt', 'on', ...
                                'bin_pt', bin_pt), ...
                         {slice_re_out1}, ...
                         {re});

% block: untitled1/c_to_ri/slice_im
slice_im = xBlock(struct('source', 'Slice', 'name', 'slice_im'), ...
                         struct('nbits', n_bits, ...
                                'mode', 'Lower Bit Location + Width'), ...
                         {c}, ...
                         {slice_im_out1});

% block: untitled1/c_to_ri/slice_re
slice_re = xBlock(struct('source', 'Slice', 'name', 'slice_re'), ...
                         struct('nbits', n_bits), ...
                         {c}, ...
                         {slice_re_out1});



end


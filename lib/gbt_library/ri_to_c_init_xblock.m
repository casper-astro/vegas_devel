%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://casper.berkeley.edu                                                %      
%   Copyright (C) 2011 Suraj Gowda    Hong Chen                               %
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
function ri_to_c_init_xblock()



%% inports
xlsub3_re = xInport('re');
xlsub3_im = xInport('im');

%% outports
xlsub3_c = xOutport('c');

%% diagram

% block: untitled/butterfly_direct/ri_to_c01/concat
xlsub3_force_re_out1 = xSignal;
xlsub3_force_im_out1 = xSignal;
xlsub3_concat = xBlock(struct('source', 'Concat', 'name', 'concat'), ...
                       [], ...
                       {xlsub3_force_re_out1, xlsub3_force_im_out1}, ...
                       {xlsub3_c});

% block: untitled/butterfly_direct/ri_to_c01/force_im
xlsub3_force_im = xBlock(struct('source', 'Reinterpret', 'name', 'force_im'), ...
                         struct('force_arith_type', 'on', ...
                                'force_bin_pt', 'on'), ...
                         {xlsub3_im}, ...
                         {xlsub3_force_im_out1});

% block: untitled/butterfly_direct/ri_to_c01/force_re
xlsub3_force_re = xBlock(struct('source', 'Reinterpret', 'name', 'force_re'), ...
                         struct('force_arith_type', 'on', ...
                                'force_bin_pt', 'on'), ...
                         {xlsub3_re}, ...
                         {xlsub3_force_re_out1});



end


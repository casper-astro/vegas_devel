%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://casper.berkeley.edu                                                %      
%   Copyright (C) 2011    Hong Chen   (based on GAVRT library uncram block)   %
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
function uncram_init_xblock(varargin)
defaults = {...
    'num_slice',8, ...
    'slice_width', 8, ...
    'bin_pt', 7, ...
    'arith_type', 1};

num_slice = get_var('num_slice', 'defaults', defaults, varargin{:});
slice_width = get_var('slice_width', 'defaults', defaults, varargin{:});
bin_pt = get_var('bin_pt', 'defaults', defaults, varargin{:});
arith_type = get_var('arith_type', 'defaults', defaults, varargin{:});


atypes = {'Unsigned', 'Signed  (2''s comp)'};
if isempty(find(arith_type == [0,1], 1)),
    errordlg('Uncram: Arithmetic Type must be 0 or 1')
end

inport = xInport('inport');
outports = cell(1,num_slice);
for i = 1:num_slice
    outports{i} = xOutport(['outport',num2str(i)]);
end

slice_out = cell(1,num_slice);
slice_blk = cell(1,num_slice);
reinterpret = cell(1,num_slice);
for j = 1:num_slice,
    slice_out{j} = xSignal(['slice_out',num2str(j)]);
    slice_blk{j} = xBlock(struct('source','xbsIndex_r4/Slice','name', ['Slice', num2str(j)]) , ...
        struct('mode', 'Upper Bit Location + Width',...
                'nbits', slice_width,...
                'bit1',-(j-1)*slice_width), ...
                {inport}, ...
                {slice_out{j}});
    reinterpret{j} = xBlock(struct('name',['Reinterp', num2str(j)],'source','xbsIndex_r4/Reinterpret'), ...
        struct('force_arith_type', 'on',  ...
                'force_bin_pt', 'on', ...
                'bin_pt', bin_pt, ...
                'arith_type', atypes(arith_type+1)), ...
                {slice_out{j}}, ...
                {outports{j}});
end


end
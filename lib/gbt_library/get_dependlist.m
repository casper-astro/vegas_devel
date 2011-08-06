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
function dependlist = get_dependlist(blk_name)

switch blk_name
    
    case 'adder_tree'
        subblocks = {'simd_add_dsp48e'};

        
    case 'simd_add_dsp48e'
        subblocks = {};        

        
    case 'parallel_cic_filter'
        subblocks = {'parallel_polynomial_dec_stage'};

                
                
    case 'parallel_polynomial_dec_stage'
        subblocks = {'parallel_adder', ...
                     'parallel_differentiator', ...
                     'parallel_filter',...
                     'parallel_integrator', ...
                     'polynomial_shift_mult_transpose', ...
                     'adder_tree'};        

                
    case 'parallel_differentiator'
        subblocks = {};        
    
        
    case 'parallel_filter'
        subblocks = {};        
      
        
    case 'polynomial_shift_mult_transpose'
        subblocks = {'shift_mult_array'};        

   
    case 'shift_mult_array'
        subblocks = {'adder_tree'};        

        
    case 'parallel_adder'
        subblocks = {'adder_tree'};        

        
    case 'parallel_integrator'
        subblocks = {'adder_tree'};        

        
        
    otherwise
        dependlist = {strcat(blk_name,'_init_xblock')};
end

dependlist = {strcat(blk_name,'_init_xblock')};
for i = 1:length(subblocks)
    temp_list = get_dependlist(subblocks{i});
    dependlist = [dependlist, temp_list{:}];
    dependlist = unique(dependlist);
end

dependlist = {unique(dependlist)};


end
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
function parallel_polynomial_dec_stage_init_xblock(m,n,n_inputs,polyphase,add_latency,dec2_halfout, n_bits, bin_pt,recursive)

if ~isprime(n)
    disp('Only supports decimation rate that is a prime number');
    return;
end

if n==2 
    %parallel_polynomial_dec2_stage_init_xblock(m,n_inputs,polyphase,add_latency,0, n_bits, bin_pt,strcmp(dec2_halfout,'on'));
    if strcmp(dec2_halfout,'off')
        disp('to be implemented');
    else
        parallel_polynomial_dec2_stage_init_xblock(m,n_inputs,polyphase,add_latency,0, n_bits, bin_pt,dec2_halfout, recursive);
    end
elseif n==3
    parallel_polynomial_dec3_stage_init_xblock(m,n_inputs,polyphase,add_latency,0, n_bits, bin_pt, recursive)
else
    parallel_polynomial_decN_stage_init_xblock(m,n,n_inputs,polyphase,add_latency,0, n_bits, bin_pt, recursive)
end



end
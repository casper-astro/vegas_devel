%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://casper.berkeley.edu                                                %      
%   Copyright (C) 2011 Hong Chen                                              %
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
function check = parallel_fir_coefficient_check(filter_coeffs)
f0=filter_coeffs(1:2:end);
f1=filter_coeffs(2:2:end);


check = 0;
if length(f0) ~= length(f1)
    %errordlg('Even number of coefficients are needed');
    disp('Even number of coefficients are needed');
    check = 1;
    return;
end

f0pf1=f0+f1;


% validate coefficients
f0_round = round(f0 * 1e16) * 1e-16;
if isempty(f0)
    %errordlg('The number of coefficients must be integer multiples of the number of inputs');
    disp('The number of coefficients must be integer multiples of the number of inputs');
    check = 1;
    return;
end
num_fir_col = length(f0);
if f0_round(1:int32(length(f0)/2)) == f0_round(length(f0):-1:int32(length(f0)/2)+1)
    num_fir_col = num_fir_col / 2;
end
if num_fir_col <= 0
    %errordlg(['error: the number of column must be a positive integer. num_fir_col = ', num2str(num_fir_col)]);
    disp(['error: the number of column must be a positive integer. num_fir_col = ', num2str(num_fir_col)]);
    check = 1;
    return;
end

f1_round = round(f1 * 1e16) * 1e-16;
if isempty(f1)
    disp('The number of coefficients must be integer multiples of the number of inputs');
    %errordlg('The number of coefficients must be integer multiples of the number of inputs');
    check =1;
    return;
end
num_fir_col = length(f1);
if f1_round(1:int32(length(f1)/2)) == f1_round(length(f1):-1:int32(length(f1)/2)+1)
    num_fir_col = num_fir_col / 2;
end
if num_fir_col <= 0
    disp(['error: the number of column must be a positive integer. num_fir_col = ', num2str(num_fir_col)]);
    %errordlg(['error: the number of column must be a positive integer. num_fir_col = ', num2str(num_fir_col)]);
    check=1;
    return;
end

f0pf1_round = round(f0pf1 * 1e16) * 1e-16;
if isempty(f0pf1)
    disp('The number of coefficients must be integer multiples of the 4');
    %errordlg('The number of coefficients must be integer multiples of the 4');
    check = 1;
    return;
end
num_fir_col = length(f0pf1);
disp(num_fir_col);
if f0pf1_round(1:int32(length(f0pf1)/2)) == f0pf1_round(length(f0pf1):-1:int32(length(f0pf1)/2)+1)
    num_fir_col = num_fir_col / 2;
end
if num_fir_col <= 0
    disp(['error: the number of column must be a positive integer. num_fir_col = ', num2str(num_fir_col)]);
    %errordlg(['error: the number of column must be a positive integer. num_fir_col = ', num2str(num_fir_col)]);
    check = 1;
    return;
end
end
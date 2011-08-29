% parallel_fir_init(blk, varargin)
%
% blk = The block to initialize.
% varargin = {'varname', 'value', ...} pairs
%
% Valid varnames for this block are:
% n_inputs = The number of parallel input samples.
% coeff = The FIR coefficients, top-to-bottom.
% n_bits = Bit width out.
% quantization = Quantization behavior [Truncate, Round (unbiased: +/- Inf),
%    or Round (unbiased: Even Values)]
% add_latency = The latency of adders.
% mult_latency = The latency of multipliers.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://seti.ssl.berkeley.edu/casper/                                      %
%   Copyright (C) 2010 Mark Wagner, Hong Chen                                 %
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

function parallel_fir_init(blk,varargin)

% Declare any default values for arguments you might like.
% Added defaults and fixed the quatization default for 10.1 tools AWL
% defaults = {'n_inputs',4,'coeff',[1 2 3 4],'n_bits', 8, 'quantization', 'Round  (unbiased: +/- Inf)', 'add_latency', 2, 'mult_latency', 3};
% check_mask_type(blk, 'dec_fir');

% if same_state(blk, 'defaults', defaults, varargin{:}), return, end
% munge_block(blk, varargin{:});

filter_coeffs = get_var('filter_coeffs','defaults', defaults, varargin{:});

% round coefficients to make sure rounding error doesn't prevent us from
% detecting symmetric coefficients
% Set attribute format string (block annotation)

f0=filter_coeffs(1:2:end);
f1=filter_coeffs(2:2:end);

f0pf1=f0+f1;

set_param([blk,'/f0'],'coeff',f0);
set_param([blk,'/f1'],'coeff',f1);
set_param([blk,'/f0pf1'],'coeff',f0pf1);

annotation=sprintf('%d taps\n%d_%d r/i', length(coeff), n_bits, n_bits-1);
set_param(blk,'AttributesFormatString',annotation);
save_state(blk, 'defaults', defaults, varargin{:});


end
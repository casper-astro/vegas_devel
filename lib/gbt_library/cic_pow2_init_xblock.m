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
% n_stages:  order of CIC filter
% dec_rate:  Decimation rate
% n_inputs:  number of parallel inputs streams
% add_latency: Add latency
% n_bits:    bitwidth of input/output data
% bin_pt:    binary point
% input_clk_period: the actual sampling period of the input signal
function cic_pow2_init_xblock(blk,varargin)


% could also just implemented as a single stage instead of decompose into
% prime factors
defaults = {'n_stages', 3, ...
            'dec_rate', 16, ...
            'n_inputs', 4,...
            'add_latency', 1, ...
            'delay_len', 1, ...
            'n_bits', 18, ...
            'bin_pt', 16, ...
            'input_clk_period', 1, ...
            'recursives', [0 0 0 0 0 0 0 0 0 0 0 0 0], ...
            };


n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
n_stages = get_var('n_stages', 'defaults', defaults, varargin{:});
n_bits = get_var('n_bits', 'defaults', defaults, varargin{:});
bin_pt = get_var('bin_pt', 'defaults', defaults, varargin{:});
delay_len = get_var('delay_len', 'defaults', defaults', varargin{:});
add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
dec_rate = get_var('dec_rate', 'defaults', defaults, varargin{:});
input_clk_period = get_var('input_clk_period', 'defaults', defaults, varargin{:});
recursives = get_var('recursives', 'defaults', defaults, varargin{:});
disp(recursives)

if mod(log2(n_inputs),1) || mod(log2(dec_rate), 1)
    disp('Only supports n_inputs == 2^n && dec_rate ==2^n');
    return;
end


f=factor(dec_rate); % all 2's
len = length(f);
% if strcmp(half2first,'off')
%     f=f(end:-1:1);
% end

n_outputs = n_inputs/dec_rate;
if n_outputs < 1
    n_outputs = 1;
end

inports =cell(1,n_inputs);
outports = cell(1,n_outputs);

for i =1:n_inputs
    inports{i}=xInport(['in',num2str(i)]);
end
for i =1:n_outputs
    outports{i} = xOutport(['out',num2str(i)]);
end

sync_in = xInport('sync_in');
sync_out = xOutport('sync_out');

n_ins=cell(1,len+1);
input_period = cell(1,len+1); % the supposed input clock rate for each stage (not actual)
n_ins{1} = n_inputs;
input_period{1} = input_clk_period;
ninputs = n_inputs;
for i = 1:len
    if mod(ninputs/2,1) == 0
        n_ins{i+1}=ninputs/2;
        ninputs=ninputs/2;
        input_period{i+1} = input_period{i};
    else
        n_ins{i+1} = ninputs;
        input_period{i+1} = 2*input_period{i};
    end
end

sigs = cell(1,len+1);
sigs{1}=inports;
sigs{len+1}=outports;
for i =2:len
    sigs{i}=cell(1,n_ins{i});
    for j=1:n_ins{i}
        sigs{i}{j}=xSignal(['sig',num2str(i),'_',num2str(j)]);
    end
end



stage_blks=cell(1,len);
stage_config = cell(1,len);
stage_sync_outs = cell(1,len);
for i=1:len
   stage_sync_outs{i} = xSignal(['stage_sync_out',num2str(i)]);
   if recursives(i)
       stage_config{i}.source = str2func('cic_stage_dec2_recursive_init_xblock');
       stage_config{i}.name = ['Stage',num2str(i),'_dec',num2str(f(i))];
       stage_blks{i}=xBlock(stage_config{i}(1), ...
                           {[blk,'/',stage_config{i}.name],...
                           'n_stages',n_stages, ...
                            'dec_rate', f(i), ...
                            'n_inputs', n_ins{i},...
                            'delay_len', delay_len, ...
                            'input_clk_period', input_clk_period, ...  %clk_rates{i}, ...
                            'input_period', input_period{i}, ...
                            'add_latency',add_latency,...
                            'input_bits', n_bits, ...
                            'bin_pt', bin_pt}, ...
                             [sigs{i},{sync_in}], ...
                             [sigs{i+1},stage_sync_outs(i)]);                        
        n_bits = ceil(n_stages*log2(f(i)*1) + n_bits);
   else
       stage_config{i}.source = str2func('cic_stage_dec2_init_xblock');
       stage_config{i}.name = ['Stage',num2str(i),'_dec',num2str(f(i))];
       stage_blks{i}=xBlock(stage_config{i}(1), ...
                           {[blk,'/',stage_config{i}.name],...
                           'n_stages',n_stages, ...
                            'dec_rate', f(i), ...
                            'n_inputs', n_ins{i},...
                            'input_clk_period', input_clk_period, ...  %clk_rates{i}, ...
                            'input_period', input_period{i}, ...
                            'add_latency',add_latency,...
                            'n_bits', n_bits, ...
                            'bin_pt', bin_pt}, ...
                             [sigs{i},{sync_in}], ...
                             [sigs{i+1},stage_sync_outs(i)]);                         
        n_bits = ceil(n_stages*log2(f(i)*1) + n_bits);
   end
                      
   disp(['stage ',num2str(i),' completed!']);
end

sync_out.bind(stage_sync_outs{len});


if ~isempty(blk) && ~strcmp(blk(1),'/')
    fmtstr =sprintf('dec_rate = %d\nOrder = %d\n',dec_rate,n_stages);
    set_param(blk,'AttributesFormatString',fmtstr);

    clean_blocks(blk);
end


disp('all done done');
end
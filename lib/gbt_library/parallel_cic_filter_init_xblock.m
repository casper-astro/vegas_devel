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
% polyphase: whether to use polyphase structure within stages
% add_latency: Add latency
% dec2_halfout: When decimation rate is 2, the number of parallel output
%               stream can be just half of n_inputs
% half2first:  How to order the prime factors, from small to big or the other way around? 
% n_bits:    bitwidth of input/output data
% bin_pt:    binary point
% recursives: How to implement each stage
%             Whether to implement using recursive structure
%             Notice that when using this structure, changing the add_latency
%             needs extra attention
% reduced_crtc_path:   Whether to use the reduced critical path structure;
%                       this designed is abandoned since we don't want the
%                       add_latency to be zero, so it's forced to be 'off'
function parallel_cic_filter_init_xblock(blk,varargin)


% could also just implemented as a single stage instead of decompose into
% prime factors
defaults = {'n_stages', 3, ...
            'dec_rate', 3, ...
            'n_inputs', 4,...
            'polyphase', 'off',...
            'add_latency', 1, ...
            'dec2_halfout', 'on', ...
            'n_bits', 18, ...
            'bin_pt', 16, ...
            'reduced_crtc_path', 'off', ...
            'recursives', [1 1 1 1 1], ...
            'half2first', 'off', ...
            };


n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
n_stages = get_var('n_stages', 'defaults', defaults, varargin{:});
n_bits = get_var('n_bits', 'defaults', defaults, varargin{:});
bin_pt = get_var('bin_pt', 'defaults', defaults, varargin{:});
add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
dec_rate = get_var('dec_rate', 'defaults', defaults, varargin{:});
polyphase = get_var('polyphase', 'defaults', defaults, varargin{:});
recursives = get_var('recursives', 'defaults', defaults, varargin{:});
dec2_halfout = get_var('dec2_halfout', 'defaults', defaults, varargin{:});
half2first = get_var('half2first', 'defaults', defaults, varargin{:});
reduced_crtc_path = get_var('reduced_crtc_path', 'defaults', defaults, varargin{:});   
reduced_crtc_path = 'off';



f=factor(dec_rate);
len = length(f);
if strcmp(half2first,'off')
    f=f(end:-1:1);
end


recursives = [reshape(recursives, 1, []), zeros(1, len)]; % try to make the recursive list long enough

if strcmp(dec2_halfout, 'on')
    n_of_2s=length(find(f==2));
    n_outputs = n_inputs/(2^n_of_2s);
    if n_outputs < 1
        n_outputs = 1;
    end
else
    n_outputs = n_inputs;
end

inports =cell(1,n_inputs);
outports = cell(1,n_outputs);

for i =1:n_inputs
    inports{i}=xInport(['in',num2str(i)]);
end
for i =1:n_outputs
    outports{i} = xOutport(['out',num2str(i)]);
end

n_ins=cell(1,len+1);
n_ins{1} = n_inputs;
ninputs = n_inputs;
for i = 1:len
    if f(i)==2 && strcmp(dec2_halfout, 'on');
        if mod(ninputs/2,1) == 0
            n_ins{i+1}=ninputs/2;
            ninputs=ninputs/2;
        else
            n_ins{i+1} = ninputs;
        end
    else
        n_ins{i+1}=ninputs;
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


%terminators = cell(1,len);  % only need these for now, no longer need them after cleaning up the sync ports
%terminator_ins=cell(1,len);


stage_blks=cell(1,len);
stage_config = cell(1,len);
for i=1:len
   %terminator_ins{i}=xSignal(['ter',num2str(i)]);
   stage_config{i}.source = str2func('parallel_polynomial_dec_stage_init_xblock');
   stage_config{i}.name = ['Stage',num2str(i),'_dec',num2str(f(i))];
   stage_blks{i}=xBlock(stage_config{i}(1), ...
                       {[blk,'/',stage_config{i}.name],...
                       'n_stages',n_stages, ...
                        'dec_rate', f(i), ...
                        'n_inputs', n_ins{i},...
                        'polyphase', polyphase,...
                        'add_latency',add_latency,...
                        'dec2_halfout', dec2_halfout, ...
                        'half2first', half2first,...
                        'n_bits', n_bits, ...
                        'bin_pt', bin_pt,...
                        'reduced_crtc_path', reduced_crtc_path, ...
                        'recursive', recursives(i)}, ...
                          sigs{i}, ...
                          sigs{i+1});
                      
   n_bits = ceil(n_stages*log2(f(i)*1) + n_bits);
   
   disp(['stage ',num2str(i),' completed!']);
end

% for i =1:len
%     terminators{i} =xBlock(struct('source', 'Terminator', 'name', ['Terminator',num2str(i)]), ...
%                                {}, ...
%                           {terminator_ins{i}}, ...
%                           {});
% end

if ~isempty(blk) && ~strcmp(blk(1),'/')
    fmtstr =sprintf('dec_rate = %d\nOrder = %d\n',dec_rate,n_stages);
    set_param(blk,'AttributesFormatString',fmtstr);

    clean_blocks(blk);
end


disp('all done done');
end
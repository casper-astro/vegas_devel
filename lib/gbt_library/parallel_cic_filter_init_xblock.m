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
function parallel_cic_filter_init_xblock(blk,m,n,n_inputs,polyphase,add_latency,dec2_halfout, half2first, n_bits, bin_pt, recursive)

f=factor(n);
len = length(f);
if strcmp(half2first,'off')
    f=f(end:-1:1);
end

n_of_2s=length(find(f==2));
n_outputs = n_inputs/(2^n_of_2s);
if n_outputs < 1
    n_outputs = 1;
end

inports =cell(1,n_inputs);
outports = cell(1,n_outputs);

n_inputs
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
    if f(i)==2
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


terminators = cell(1,len);  % only need these for now, no longer need them after cleaning up the sync ports
terminator_ins=cell(1,len);


stage_blks=cell(1,len);
for i=1:len
   terminator_ins{i}=xSignal(['ter',num2str(i)]);
   stage_blks{i}=xBlock(struct('source', 'parallel_polynomial_dec_stage_init_xblock', 'name', ['Stage',num2str(i),'_dec',num2str(f(i))]), ...
                               {m,f(i),n_ins{i},polyphase,add_latency,dec2_halfout, n_bits, bin_pt,recursive}, ...
                          [{sigs{1}{1}},sigs{i}], ...
                          [{terminator_ins{i}},sigs{i+1}]);
end

for i =1:len
    terminators{i} =xBlock(struct('source', 'Terminator', 'name', ['Terminator',num2str(i)]), ...
                               {}, ...
                          {terminator_ins{i}}, ...
                          {});
end


fmtstr =sprintf('dec_rate = %d\nOrder = %d\n',n,m);
set_param(blk,'AttributesFormatString',fmtstr);

clean_blocks(blk);
end
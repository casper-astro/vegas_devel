%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://casper.berkeley.edu                                                %      
%   Copyright (C) 2011    Hong Chen  (based on CASPER library dsp_scope block)%
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
function parallel_scope_init_xblock( varargin)

defaults = {...
    'n_inputs',8, ...
    'sample_period',1};

n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
sample_period = get_var('sample_period', 'defaults', defaults, varargin{:});


inports = cell(1,n_inputs);
for i =1:n_inputs
    inports{i} = xInport(['in',num2str(i)]);
end

outport = xOutport('out');


switcher_in = cell(1,n_inputs);
for i =1:n_inputs
    switcher_in{i} = xSignal(['switcher_in',num2str(i)]);
end
sel_out = xSignal('sel_out');
sel = xBlock(struct('source','simulink/Sources/Counter Free-Running', 'name','sel'), ...                   
                    struct('NumBits', ceil( log2(n_inputs)), ...
                            'tsamp', sample_period/n_inputs), ...                        
                          {}, ...
                          {sel_out});
                      


gateways = cell(1,n_inputs);                    
for i=1:n_inputs
    gateways{i} = xBlock(struct('source','xbsTypes_r4/Gateway Out','name',['gateway_out',num2str(i)]), ...
                        struct('hdl_port','off'), ...
                        {inports{i}}, ...
                        {switcher_in{i}});
end
switcher = xBlock(struct('source', 'simulink/Signal Routing/Multiport Switch', 'name','switch'), ...
                    struct('Inputs', n_inputs, ...
                            'zeroidx', 'on'), ...
                            [{sel_out},switcher_in], ...
                            {outport});
                        
                        

end



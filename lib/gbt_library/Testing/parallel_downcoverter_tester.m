%% Jan 20, 2012
%% works with parallel_downconverter_test.mdl
%% 4 and 5 inputs
%%
%   Tested Cases:
%       (1) 4-input, dec_rate = 2, both types of Down Sample

dec_rate = 2;   % change this accordingly.
n_inputs = 4;   % change this accordingly
clk_rate_change = 0; % change this accordingly
outs = zeros(length(out1.signals.values),n_inputs);
for i = 1:n_inputs
    eval(['outs(:,i) = out' num2str(i) '.signals.values']);
    size(outs)
end
if ~clk_rate_change    % if the clock rate isn't changed in the downconverter
    out_unfold = zeros(length(out1.signals.values)*n_inputs/dec_rate,1);  
    for i = 1:length(out1.signals.values)/dec_rate
        for j = 1:n_inputs
            out_unfold((i-1)*n_inputs+j,1) = outs((i-1)*dec_rate+1,j);
        end
    end
else
    out_unfold = zeros(length(out1.signals.values)*n_inputs,1);
    for i =1:length(out1.signals.values)
        for j = 1:n_inputs
            out_unfold((i-1)*n_inputs+j,1) = outs(i,j);
        end
    end
end
dlmwrite('downconverter.txt',outs,'delimiter','\t','precision', '%16.8f');
dlmwrite('downconverter_unfold.txt',out_unfold);
%% Jan 19, 2012
%% Jan 20, 2012
%% This script works with cic_filter_test_16.mdl to test the behavior of the
%% multi-stage parallel cic filter
%% 8 parallel input in this case
%%
%  When using the xilinx Down Sample block in the CIC stage, the output of
%  parallel CIC block is exactly the same as the xilinx CIC filter except
%  for the delay (see pcic_xcic_comparison.txt);
%  When using the manual Down Sample block in the CIC stage, we need to
%  note now the clock rate isn't really changed in the CIC stage, but
%  rather the same values are hold for a few cycles. The result shows that
%  it still produces exactly the same result for both parallel CIC and
%  xilinx CIC, but with different delay value. { <----this is only true for a few cases}
%
%  Tested cases (bit-by-bit):
%       (1) dec_rate = 16(halved outputs), non-recursive
%  Tested cases (plot):
%       (1) dec_rate = 16(halved outputs), non-recursive


dec_rate = 2;   % change this accordingly.
n_inputs = 1;   % change this accordingly, used xilinx CIC?
clk_rate_change = 0; % change this accordingly, note the special case when the number of output ports are halved
pcic_outs = cell(1,n_inputs);
pcic = zeros(length(pcic_out1.signals.values),n_inputs);
for i = 1:n_inputs
    eval(['pcic_outs{i} = pcic_out' num2str(i) '.signals.values']);
    size(pcic)
    size(pcic_outs{i})
    pcic(:,i)= pcic_outs{i};
end
xcic=xcic_out.signals.values;
dlmwrite('cic.txt',pcic,'delimiter','\t','precision', '%16.8f');
dlmwrite('xcic.txt',xcic);
pcic_length = length(pcic_out1.signals.values)
if ~clk_rate_change   % if the clock rate isn't changed in the CIC
    pcic_unfold = zeros(pcic_length*n_inputs/dec_rate,1);  
    for i = 1:pcic_length/dec_rate
        for j = 1:n_inputs
            pcic_unfold((i-1)*n_inputs+j,1) = pcic((i-1)*dec_rate+1,j);
        end
    end
else
    pcic_unfold = zeros(pcic_length*n_inputs,1);
    for i =1:pcic_length
        for j = 1:n_inputs
            pcic_unfold((i-1)*n_inputs+j,1) = pcic(i,j);
        end
    end
end
size(pcic_unfold)
size(xcic)
a = length(pcic_unfold)
b = length(xcic)
if a < b
    pcic_unfold = [pcic_unfold;zeros(b-a,1)];
else
    xcic = [xcic;zeros(a-b,1)];
end
size(pcic_unfold)
size(xcic)
dlmwrite('pcic_unfold.txt',pcic_unfold);
dlmwrite('pcic_xcic_comparison.txt',[pcic_unfold,xcic],'delimiter','\t','precision', '%16.8f');

figure(1)
semilogy(abs(fft(pcic_unfold)),'b');
hold on;
semilogy(abs(fft(xcic)),'r');
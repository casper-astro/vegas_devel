%% Jan 19, 2012
%% Jan 20, 2012
%% This script works with cic_stage_test.mdl to test the behavior of the
%% single stage parallel cic filter (compare to the xilinx CIC)
%% 4 or 8 parallel inputs in this case
%%
%  When using the xilinx Down Sample block in the CIC stage, the output of
%  parallel CIC block is exactly the same as the xilinx CIC filter except
%  for the delay (see pcic_xcic_comparison.txt);
%
%  When using the manual Down Sample block in the CIC stage, we need to
%  note now the clock rate isn't really changed in the CIC stage, but
%  rather the same values are hold for a few cycles. The result shows that
%  it still produces exactly the same result for both parallel CIC and
%  xilinx CIC, but with different delay value.{ <----this is only true for
%  a few cases}
%
%  Test cases (bit-by-bit):
%       (1) order = 3, dec_rate = 3, non-recursive[Good for both], recursive[Fail! Wrong value for non-xilinx, Good for xilinx]
%           (1.5) order = 1, dec_rate = 3, non-recursive, recursive
%       (2) order = 3, dec_rate = 7, non-recursive[Good for both], recursive[OK for xilinx, Fail! Wrong value for non-xilinx]
%       (3) order = 3, dec_rate = 2(full outputs), non-recursive[Scrambled! for both
%                                                               but should be ok,just delay/mismatch, regularly scrambled],
%                                                  recursive[Scrambled! for both
%                                                               but should be ok,just delay/mismatch, regularly scrambled],
%       (4) order = 1, dec_rate = 2(full outputs) non-recursive[Scrambled! for both
%                                                               but should be ok,just delay/mismatch, regularly scrambled],
%                                                  recursive[Scrambled! for both but should be
%                                                               ok,just delay/mismatch, regularly scrambled],
%       (5) order = 1, dec_rate = 2(halved outputs), non-recursive
%
%  Test cases (plot):
%       (1) order = 3, dec_rate = 3, non-recursive[OK for both], recursive[Fail! for non-xilinx, OK for xilinx]
%           (1.5) order = 1, dec_rate = 3, non-recursive, recursive
%       (2) order = 3, dec_rate = 7, non-recursive[OK for both], [OK for xilinx, Fail! for non-xilinx]
%       (3) order = 3, dec_rate = 2(full outputs), non-recursive[OK for both], 
%                                                  recursive[OK for both]
%       (4) order = 1, dec_rate = 2(full outputs), non-recursive[OK for both], recursive[OK for both]

dec_rate = 2;   % change this accordingly.
n_inputs = 4;   % change this accordingly
clk_rate_change = 0; % change this accordingly, used xilinx Down Sample?
pcic = zeros(length(pcic_out1.signals.values),n_inputs);
for i = 1:n_inputs
    eval(['pcic(:,i) = pcic_out' num2str(i) '.signals.values']);
    size(pcic)
end
xcic=xcic_out.signals.values;
dlmwrite('cic.txt',pcic,'delimiter','\t','precision', '%16.8f');
dlmwrite('xcic.txt',xcic);
pcic_length = length(pcic_out1.signals.values)
if ~clk_rate_change    % if the clock rate isn't changed in the CIC
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
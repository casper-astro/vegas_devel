dec_rate = 4;

xCIC_input0 = xCIC_input.signals.values;
xCIC_result = xCIC.signals.values;
npCIC_input0 = npCIC_input.signals.values;
npCIC_result = npCIC.signals.values;
myCIC_in0 = myCIC_input.signals.values;
myCIC_in1 = myCIC_input1.signals.values;
myCIC_in2 = myCIC_input2.signals.values;
myCIC_in3 = myCIC_input3.signals.values;
myCIC_in = zeros(1,length(myCIC_in0)*4);
for i =1:length(myCIC_in0)
    myCIC_in(4*(i-1)+1) = myCIC_in0(i);
    myCIC_in(4*(i-1)+2) = myCIC_in1(i);
    myCIC_in(4*(i-1)+3) = myCIC_in2(i);
    myCIC_in(4*(i-1)+4) = myCIC_in3(i);
end
myCIC_in = transpose(myCIC_in);
myCIC_result = myCIC.signals.values;

xCIC_input_spec = 2*abs(fft(xCIC_input0(1:dec_rate:end)));
xCIC_result_spec = 2*abs(fft(xCIC_result));
xCIC_response  = xCIC_input_spec./xCIC_result_spec;
span = 25; % Size of the averaging window
window = ones(span,1)/span; 
xCIC_response_smoothed = convn(xCIC_response,window,'same');
semilogy(xCIC_response_smoothed);
%semilogy(npCIC_response,'r');

% hold on;
% 
% npCIC_input_spec = 2*abs(fft(npCIC_input0(1:dec_rate:end)));
% npCIC_result_spec = 2*abs(fft(npCIC_result));
% npCIC_response  = npCIC_input_spec./npCIC_result_spec;
% npCIC_response_smoothed = convn(npCIC_response,window,'same');
% semilogy(npCIC_response_smoothed,'r');
%semilogy(npCIC_response,'r');


hold on;
myCIC_input_spec = 2*abs(fft(myCIC_in(1:dec_rate:end)));
myCIC_result_spec = 2*abs(fft(myCIC_result));
myCIC_response  = myCIC_input_spec./myCIC_result_spec;
myCIC_response_smoothed = convn(myCIC_response,window,'same');
semilogy(myCIC_response_smoothed,'g');
%semilogy(npCIC_response,'r');


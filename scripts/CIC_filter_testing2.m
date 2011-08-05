dec_rate = 100;

input = cic_in.signals.values;
xcic_result = xcic_result.signals.values;
mycic_result = mycic_result.signals.values;

input_spec = 2*abs(fft(input(1:dec_rate:end)));
xCIC_result_spec = 2*abs(fft(xcic_result));
xCIC_response  = xCIC_result_spec./input_spec;
span = 25; % Size of the averaging window
window = ones(span,1)/span; 
xCIC_response_smoothed = convn(xCIC_response,window,'same');
%semilogy(xCIC_response_smoothed);
   %%semilogy(npCIC_response,'r');



%hold on;
myCIC_result_spec = 2*abs(fft(mycic_result));
myCIC_response  = myCIC_result_spec./input_spec;
myCIC_response_smoothed = convn(myCIC_response,window,'same');
%semilogy(npCIC_response_smoothed,'r');
    %%semilogy(npCIC_response,'r');





figure(1)
semilogy(myCIC_response_smoothed,'k');
hold on;
semilogy(xCIC_response_smoothed);

figure(2)
semilogy(xCIC_response_smoothed);
hold on;
semilogy(myCIC_response_smoothed,'g');







% %some crazy test
% hold on;
% crzCIC_input_spec = 2*abs(fft(myCIC_in(1:dec_rate:end)));
% crzCIC_input_spec = crzCIC_input_spec(1:end-1);
% crzCIC_result_spec = 2*abs(fft(xCIC_result));
% crzCIC_response  = crzCIC_result_spec./crzCIC_input_spec;
% crzCIC_response_smoothed = convn(crzCIC_response,window,'same');
% semilogy(crzCIC_response_smoothed,'r');
% %semilogy(npCIC_response,'r');
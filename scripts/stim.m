clocks = 2^5;
t = 0:clocks;

% interleaved adc takes 8 samples per 1 fpga clock
t = t'/8;

% freq in terms of fpga clocks
din = .8*sin(2*pi*t);

%din = chirp(t,1/16.0,t(end),4.0);
adc_sim_in=[t,din];

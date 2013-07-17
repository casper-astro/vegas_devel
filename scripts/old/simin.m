clocks = 2^20;

t=0:clocks;

% interleaved ADC takes 8 samples per 1 FPGA clock
t=t'/8;

% freq in terms of FPGA clocks
din = .7*sin(2*pi*t/5);

adc_sim_in = [t,din];


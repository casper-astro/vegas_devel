% coeffs = cic_fir_coeff_gen(dec_rate, order, fir_len, diff_delay, sampling_rate, passband);
function h = cic_fir_coeff_gen(dec_rate, order, n_taps, diff_delay, Fs, passband)
%%%%% Adapted from http://www.altera.com/literature/an/an455.pdf


%%%%%% CIC filter parameters %%%%%%
R = dec_rate; %% Decimation factor
M = diff_delay; %% Differential delay
N = order; %% Number of stages

B = 18; %% Coeffi. Bit-width
% Fs = 1.6e9; %% (High) Sampling freq in Hz before decimation
% passband = 0.3;
Fc = passband*(Fs/2)/R; %% Pass band edge in Hz


%%%%%%% fir2.m parameters %%%%%%
L = n_taps-1; %% Filter order; must be even
Fo = R*Fc/Fs; %% Fo = 0.75*(1/2) %% Normalized Cutoff freq; 0<Fo<=0.5/M;
% Fo = 0.5/M; %% use Fo=0.5 if you don't care responses are


%% outside the pass band
%%%%%%% CIC Compensator Design using fir2.m %%%%%%
p = 2e3; %% Granularity
s = 0.25/p; %% Step size
fp = [0:s:Fo]; %% Pass band frequency samples
fs = (Fo+s):s:0.5; %% Stop band frequency samples
f = [fp fs]*2; %% Normalized frequency samples; 0<=f<=1
Mp = ones(1,length(fp)); %% Pass band response; Mp(1)=1
Mp(2:end) = abs( M*R*sin(pi*fp(2:end)/R)./sin(pi*M*fp(2:end))).^N;
Mf = [Mp zeros(1,length(fs))];
f(end) = 1;
h = fir2(L,f,Mf); %% Filter length L+1

%disp(h);


h = h/max(h); %% Floating point coefficients
hz = round(h*power(2,B-1)-1); %% Fixed point coefficients

end

clear all;


lo_f = 237e6;

%%%%%% CIC filter parameters %%%%%%
R1 = 32; %% Decimation factor 1
R2 = 4; %% Decimation factor 2
M = 1; %% Differential delay
N1 = 4; %% Number of stages
N2 = 3;

N = N1;


B = 18; %% Coeffi. Bit-width
Fs = 3.0e9; %% (High) Sampling freq in Hz before decimation

L = 7;
passband = 0.4;
Fc = passband*(Fs/2)/2;


p = 2e3; %% Granularity
s = 0.25/p; %% Step size
f = (0:s:0.5)*2;

R = R1*R2;


f_range = 2; % how many times alias
f_ind = [f 1:2*s:f_range]; % The step of frequencies is 2*s
n_f_range = (0.5*Fs/R1)*f_range; % f_range in Htz


[initH, ripple] = remez(L, [0 2*passband 1 1], ...
    [0.5 0.5 0 0], {length(f)});
h((L+1):L:(2*L+1)) = [0.5, 0];
h(1:2:end)  = initH(1:end);
%freqz(h)

% frequency range in Htz
hf = n_f_range/length(f_ind):n_f_range/length(f_ind):n_f_range;  
hf = hf';
hf = hf - n_f_range/2;
% frequency response of the Halfband filter
hh = freqz(h, 1, hf, Fs/R1);  % the actual decimation rate after first CIC is Fs/R1
%plot(f, Mf, w/pi, abs(hh))


% frequecy response of the first CIC filter
% consisting of a normal dec16 CIC and a dec4 CIC without really decimating
Cf = hf'/(Fs/R1);  % normalized limited frequency range
Cf = Cf/R1; % limited normalized frequency range
% R1 = R1*4; % the CIC response shape should look like dec64
Cfp = abs( M*R1*sin(pi*Cf)./sin(pi*(M*R1)*Cf)).^-N;
% R1 = R1/4;


% frequecy response of the second CIC filter
% consisting of a dec2 CIC 
N = N2;
Cf1 = hf'/(Fs/R); % normalized frequency range, with regard to the final output sampling rate
Cf1 = Cf1/R2;
Cfp1 = abs( M*(R2)*sin(pi*Cf1)./sin(pi*(M*R2)*Cf1)).^-N;

combined_response = (Cfp.*hh').*Cfp1;
% combined_response = Cfp.*hh';
% combined_response = Cfp.*Cfp1;


dlmwrite(['dec_', num2str(R1*R2), '_order_', num2str(N1), num2str(N2), '_halfband_',num2str(L), '_0',num2str(10*passband),'.txt'], h, ' ');



figure(1)
plot(hf', Cfp,'--', ...
    hf, abs(hh),'-.',...
    hf', Cfp1,'--', ...
    hf, abs(combined_response),'LineWidth', 2);
xlabel(['Normalized Frequency (', num2str(hf(1)/1e6),' ~ ', num2str(hf(end)/1e6), 'MHz)  [', num2str(f_range), 'xNyquist for CIC1]'], ...
    'FontSize', 14);
ylabel('Frequency Response', ...
    'FontSize', 14);
title(['Frequency Response of a (decimation rate =', num2str(R1),',order = ', num2str(N1), ' CIC filter) + (',...
    num2str(length(find(h))), '\_', num2str(length(h)), 'tap Halfband FIR filter) + (dec',num2str(R2),' order = ', num2str(N2), ' CIC)'], ...
    'FontSize', 16);
legend(['Frequency Response of the dec\_rate = ', num2str(R1), ' differential delay =', num2str(R1), ' order = ', num2str(N1), ' CIC'], ...
    ['Frequency Response of the ', num2str(length(h)),'-tap(',num2str(length(find(h))),'-nonzero) Halfband FIR filter'], ...
    ['Frequency Response of the dec\_rate = differential delay = 4',' order = ', num2str(N2), ' CIC'], ...
    'Frequency Response of the combined Filter', ...
    'Location', 'SouthWest', ...
    'FontSize', 12);
grid on;

figure(2)
plot(hf', 20*log10(Cfp/max(Cfp)), '--', ...
    hf, 20*log10(abs(hh)/max(abs(hh))),'-.',...
    hf', 20*log10(Cfp1/max(Cfp1)),'--',...
    hf, 20*log10(abs(combined_response)/max(abs(combined_response))),'LineWidth', 2);
xlabel(['Normalized Frequency (', num2str(hf(1)/1e6),' ~ ', num2str(hf(end)/1e6), 'MHz)  [', num2str(f_range), 'xNyquist for CIC1]'], ...
    'FontSize', 14);
ylabel('Frequency Response (dB)', ...
    'FontSize', 14);
title(['Frequency Response of a (decimation rate =', num2str(R1),',order = ', num2str(N1), ' CIC filter) + (',...
    num2str(length(find(h))), '\_', num2str(length(h)), 'tap Halfband FIR filter) + (dec',num2str(R2),' order = ', num2str(N2), ' CIC)'], ...
    'FontSize', 16);
legend(['Frequency Response of the dec\_rate = ', num2str(R1), ' differential delay =', num2str(R1), ' order = ', num2str(N1), ' CIC'], ...
    ['Frequency Response of the ', num2str(length(h)),'-tap(',num2str(length(find(h))),'-nonzero) Halfband FIR filter'], ...
    ['Frequency Response of the dec\_rate = differential delay = 4', ' order = ', num2str(N2), ' CIC'], ...
    'Frequency Response of the combined Filter', ...
    'Location', 'SouthWest', ...
    'FontSize', 12);
grid on;


figure(3)
freqs = dlmread('freqs237.txt','\n');
freqs = freqs*1e6;
response = dlmread('response237.txt', '\n');
plot(hf + lo_f, 20*log10(abs(combined_response)/max(abs(combined_response))), 'c--', ...
	freqs, 20*log10(response/max(response)), 'LineWidth', 2);
xlabel(['Normalized Frequency (', num2str((hf(1)+lo_f)/1e6) ,'-', num2str((hf(end)+lo_f)/1e6), 'MHz)  [', ...
    num2str(f_range*R2/2), 'x Output Bandwidth]; LO freq: ', num2str(lo_f/1e6), 'MHz'], ...
    'FontSize', 14);
ylabel('Frequency Response (dB)', ...
    'FontSize', 14);
title(['Frequency Response of a (decimation rate =', num2str(R1),',order = ', num2str(N1), ' CIC filter) + (',...
    num2str(length(find(h))), '\_', num2str(length(h)), 'tap Halfband FIR filter) + (dec',num2str(R2),' order = ', num2str(N2), ' CIC)'], ...
    'FontSize', 16);
legend('Frequency Response of the combined Filter (prediction)', ...
    'Frequency Response (Actual data)', ...
    'Location', 'SouthWest', ...
    'FontSize', 12);
hold on;
plot([lo_f - Fs/(2*R1*R2), lo_f - Fs/(2*R1*R2)], [-200, 50], 'r');
hold on;
plot([lo_f + Fs/(2*R1*R2), lo_f + Fs/(2*R1*R2)], [-200, 50], 'r');
grid on;
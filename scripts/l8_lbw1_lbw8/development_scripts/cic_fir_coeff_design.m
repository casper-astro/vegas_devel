%%%%% Adapted from http://www.altera.com/literature/an/an455.pdf


%%%%%% CIC filter parameters %%%%%%
R = 64; %% Decimation factor
M = 1; %% Differential delay
N = 3; %% Number of stages

B = 18; %% Coeffi. Bit-width
Fs = 1.6e9; %% (High) Sampling freq in Hz before decimation
Fc = 0.5*(Fs/2)/R; %% Pass band edge in Hz


%%%%%%% fir2.m parameters %%%%%%
L = 16; %% Filter order; must be even
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



f_range = 2; % how many times alias
f_ind = [f 1:2*s:f_range]; % The step of frequencies is 2*s
n_f_range = (0.5*Fs/R)*f_range; % f_range in Htz

%[hh, w] = freqz(h, 1, length(f));
hf = n_f_range/length(f_ind):n_f_range/length(f_ind):n_f_range;
hf = hf';
hh= freqz(h, 1, hf, Fs/R);
%plot(f, Mf, w/pi, abs(hh))

%% Calculate the combined frequency response of CIC and FIR
%Cf = [fp fs];
%Cfp = abs( M*R*sin(pi*Cf/R)./sin(pi*M*Cf)).^-N;
Cf = hf'/(Fs/R);
Cfp = abs( M*R*sin(pi*Cf/R)./sin(pi*M*Cf)).^-N;
%Ffp = Mf.*(Cfp);
Ffp = abs(hh)'.*Cfp;



figure(1)
%plot(f, Mf, w/pi, abs(hh), 'g-.', f, Ffp, 'r', f, Cfp, 'k:', f, abs(Mf.*Cfp), 'm--');
plot(f_ind/f_range, [Mf, zeros(1, length(hf)-length(Mf))], hf/n_f_range, abs(hh), 'g-.', hf/n_f_range, Ffp, 'r', hf/n_f_range, Cfp, 'k:',f_ind/f_range, abs([Mf, zeros(1, length(hf)-length(Mf))].*Cfp), 'm--');
xlabel(['Normalized Frequency (0-', num2str(0.5*f_range*Fs/(R*1e6)), 'MHz)']);
ylabel('Frequency Response');
title(['Frequency Response of a ',num2str(N), '-order, decimation rate = differential delay = 64 CIC filter']);
legend('Ideal FIR response', 'Expected FIR response', 'Expected CIC-FIR response', 'Expected CIC response', 'Ideal FIR-CIC response');
figure(2)
%plot(f, 10*log10(Mf), w/pi, 10*log10(abs(hh)),'g-.', f, 10*log10(Ffp), 'r', f, 10*log10(Cfp), 'k:', f, 10*log10(abs(Mf.*Cfp)), 'm--');
plot(f_ind/f_range, 10*log10([Mf, zeros(1, length(hf)-length(Mf))]), hf/n_f_range, 10*log10(abs(hh)), 'g-.', hf/n_f_range, 10*log10(Ffp), 'r', hf/n_f_range, 10*log10(Cfp), 'k:', f_ind/f_range, 10*log10(abs([Mf, zeros(1, length(hf)-length(Mf))].*Cfp)), 'm--');
xlabel(['Normalized Frequency (0-', num2str(0.5*f_range*Fs/(R*1e6)), 'MHz)']);
ylabel('Frequency Response (dB)');
title(['Frequency Response of a ',num2str(N), '-order, decimation rate = differential delay = 64 CIC filter']);
legend('Ideal FIR response', 'Expected FIR response', 'Expected CIC-FIR response', 'Expected CIC response', 'Ideal FIR-CIC response');


shz = size(hz);
strhz = '[';
strh = '[';
for i = 1:shz(1,2)
    strhz = [strhz, num2str(hz(i)), ', '];
    strh = [strh, num2str(h(i)), ', '];
end
strhz = [strhz, ']'];
strh = [strh, ']'];
%disp(strhz);
disp(strh);

%%% [3, -1012, 1101, 2206, -5712, -738, 15184, -11375, -29048, 57685,
%%% 131071, 57685, -29048, -11375, 15184, -738, -5712, 2206, 1101, -1012, 3]
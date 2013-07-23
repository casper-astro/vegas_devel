%function coeff_vector = pfb_coeff_gen_calc(PFBSize, TotalTaps, WindowType,
%n_inputs, nput, fwidth, a, debug)

PFBSize = 10;
TotalTaps = 4;
WindowType = 'hamming';
n_inputs = 4;
nput = 0;
fwidth = 1;
a = -1;



% coeffs = 1:2^(PFBSize+n_inputs)*TotalTaps;
% for i =1:2^n_inputs
%     coeffs(i:2^n_inputs:end) = pfb_coeff_gen_calc(PFBSize, TotalTaps, WindowType, n_inputs, i, fwidth, a, 0);
% end
coeffs = pfb_coeff_gen_calc(PFBSize, TotalTaps, WindowType, n_inputs, nput, fwidth, a, 0);



N = 2^PFBSize;      % DFT length
k = 177;              % bin where DFT filter is centered
wk = 0; %2*pi*k/N;        % normalized radian center-frequency
wStep = 2*pi/N;
w = [0:wStep:2*pi - wStep]; % DFT frequency grid
interp = 10;
N2 = interp*N; % Denser grid showing "arbitrary" frequencies
w2Step = 2*pi/N2;
w2 = [0:w2Step:2*pi - w2Step]; % Extra dense frequency grid

clf;
subplot(4,1,1);
size(coeffs)
%[h, w] = freqz(coeffs, 1, 2^(PFBSize)*TotalTaps, 'whole');
h = freqz(coeffs, 1, w2-wk);
pfbr_n = 10*log10(abs(h))/max(10*log10(abs(h)));
plot(pfbr_n);
grid on;


subplot(4,1,2);
hold off;
X = (1 - exp(1i*(w2-wk)*N)) ./ (1 - exp(1i*(w2-wk)));
X(1+k*interp) = N; % Fix divide-by-zero point (overwrite NaN)
magX_n = abs(X)/max(abs(X));
magXd_n = magX_n(1:interp:N2); % DFT frequencies only
plot(w2,magX_n,'-'); hold on; grid on;
plot(w,magXd_n,'*');         % Show DFT sample points

%size(abs(h))
%size(magX)
subplot(4,1,3);
hold off;
plot(pfbr_n.*magX_n, '-o');
grid on;

% subplot(4,1,4)
% prev_response = pfbr_n.*magX_n;
% plot(prev_response, '-o');
% hold on;
% k = k+1;
% wk = 2*pi*k/N;
% pfbr_h = freqz(coeffs, 1, w2-wk);
% pfbr_n = 10*log10(abs(pfbr_h))/max(10*log10(abs(pfbr_h)));
% fftr = (1-exp(1i*(w2-wk)*N)) ./ (1 - exp(1i*(w2-wk)));
% fftr(1+k*interp) = N; % Fix divide-by-zero point (overwrite NaN)
% fftr_n = 10*log10(abs(fftr))/max(10*log10(abs(fftr)));
% plot(pfbr_n.*fftr_n, '-or');
% hold on;
% response_sum = prev_response + pfbr_n.*fftr_n;
% plot(response_sum, '-og');
% grid on;

freq_response_sum = zeros(1, N2);
subplot(4,1,4);
hold off;
for k = 178:202
    wk = 2*pi*k/N;  % normalized radian center-frequency
    pfbr_h = freqz(coeffs, 1, w2-wk);
    pfbr_n = 10*log10(abs(pfbr_h))/max(10*log10(abs(pfbr_h)));
    fftr = (1 - exp(1i*(w2-wk)*N)) ./ (1 - exp(1i*(w2-wk)));
    if k*interp < N2
        fftr(1+k*interp) = N; % Fix divide-by-zero point (overwrite NaN)
    end
    fftr_n = abs(fftr)/max(abs(fftr));
    %size(pfbr_n)
    %size(fftr_n)
    response = pfbr_n.*fftr_n;
    %size(freq_response_sum)
    %size(response)
    %plot(response, '-*r');
    hold on;
    freq_response_sum = freq_response_sum + response;
end
plot(freq_response_sum, '-o');
grid on;

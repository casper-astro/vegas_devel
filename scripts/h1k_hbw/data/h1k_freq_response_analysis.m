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
k = 178;              % bin where DFT filter is centered
wk = 2*pi*k/N;        % normalized radian center-frequency
wStep = 2*pi/N;
w = [0:wStep:2*pi - wStep]; % DFT frequency grid
interp = TotalTaps;
N2 = interp*N; % Denser grid showing "arbitrary" frequencies
w2Step = 2*pi/N2;
w2 = [0:w2Step:2*pi - w2Step]; % Extra dense frequency grid

clf;
subplot(3,1,1);
size(coeffs)
%[h, w] = freqz(coeffs, 1, 2^(PFBSize)*TotalTaps, 'whole');
h = freqz(coeffs, 1, w2-wk);
semilogy(abs(h));
grid on;



subplot(3,1,2);
hold off;
X = (1 - exp(1i*(w2-wk)*N)) ./ (1 - exp(1i*(w2-wk)));
X(1+k*interp) = N; % Fix divide-by-zero point (overwrite NaN)
magX = abs(X);
magXd = magX(1:interp:N2); % DFT frequencies only
plot(w2,magX,'-'); hold on; grid on;
plot(w,magXd,'*');         % Show DFT sample points

size(abs(h))
size(magX)
subplot(3,1,3);
hold off;
plot(10*log10(abs(h)).*magX, '-o');
grid on;
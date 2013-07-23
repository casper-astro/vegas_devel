filename1 = 'signal1.txt';
filename2 = 'signal2.txt';
dec_rate = 128;
sig1 = dlmread(filename1,'\n');
sig2 = dlmread(filename2,'\n');
Fs1 = 1.6e9;
Fs2 = Fs1/dec_rate;

L1 = length(sig1);
nfft1 = 2^nextpow2(L1);
f1 = Fs1/2*linspace(0,1,nfft1/2+1);
y1 = fft(sig1,nfft1)/L1;

L2 = length(sig2);
nfft2 = 2^nextpow2(L2);
f2 = Fs2/2*linspace(0,1,nfft2/2+1);
y2 = fft(sig2,nfft2)/L2;

semilogy(f1, 2*abs(y1(1:nfft1/2+1)));
hold on;
semilogy(f2, 2*abs(y2(1:nfft2/2+1)),'g');


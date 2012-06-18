function coeffs = halfband_coeff_gen(n_taps, passband)

L = n_taps;


p = 2e3; %% Granularity
s = 0.25/p; %% Step size
f = (0:s:0.5)*2;



[initH, ripple] = remez(L, [0 2*passband 1 1], ...
    [0.5 0.5 0 0], {length(f)});
h((L+1):L:(2*L+1)) = [0.5, 0];
h(1:2:end)  = initH(1:end);

coeffs = h;
%freqz(h)
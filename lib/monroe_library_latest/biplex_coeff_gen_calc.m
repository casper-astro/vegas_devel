function ActualCoeffs = biplex_coeff_gen_calc(FFTSize, FFTStage)
Coeffs = 0:2^(FFTStage-1)-1;

% Compute the complex, bit-reversed values of the twiddle factors
%br_indices = Coeffs; %bit_rev(Coeffs, FFTSize-1);
br_indices = bit_rev(Coeffs, FFTSize-1);
br_indices = -2*pi*1j*br_indices/2^FFTSize;
ActualCoeffs = exp(br_indices);
%ActualCoeffsStr = 'exp(-2*pi*1j*(bit_rev(Coeffs, FFTSize-1))/2^FFTSize)';
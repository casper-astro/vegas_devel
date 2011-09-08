function ActualCoeffs = direct_coeff_gen_calc(FFTSize, FFTStage, LargerFFTSize, StartStage, tap_in_stage)

% FFTSize = 4;
% FFTStage = 4;
% LargerFFTSize = 14;
% StartStage = 11;
% tap_in_stage = 2;



stage=FFTStage;
i= tap_in_stage; %zero to (2^FFTSize) -1
redundancy = 2^(LargerFFTSize - FFTSize);

r=0:redundancy-1;
n = bit_reverse(r, LargerFFTSize - FFTSize);
Coeffs = floor((i+n*2^(FFTSize-1))/2^(LargerFFTSize-(StartStage+stage-1)));
br_indices = bit_rev(Coeffs, LargerFFTSize-1);
br_indices = -2*pi*1j*br_indices/2^LargerFFTSize;
ActualCoeffs = exp(br_indices);
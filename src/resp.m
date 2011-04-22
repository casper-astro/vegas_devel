%script to readout ToWorkspace Struct

adc_vals_in=adc_input.signals.values(1:end);

adc_vals_out=adc_dump.signals.values(1:end,1:end);

%dimensions
%number of elements
numel(adc_vals_out)


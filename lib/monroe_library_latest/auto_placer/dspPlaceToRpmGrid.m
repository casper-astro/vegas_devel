function [res_x, res_y] = dspPlaceToRpmGrid(x, y)
%these values were pulled by hand from the Xilinx FPGA editor.
dsp_rpm_x_arr = [37, 49, 73, 85, 109, 121, 173, 185, 209, 221];

res_x = dsp_rpm_x_arr(x);
res_y = 5*(y-1);

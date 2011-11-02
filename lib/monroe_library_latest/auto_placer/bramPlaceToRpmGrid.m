function [res_x, res_y] = bramPlaceToRpmGrid(x, y)

%these values were pulled by hand from the Xilinx FPGA editor.
bram_rpm_x_arr = [25, 61, 97, 133, 161, 197, 233];

bram36_index = floor( (y-1)/2 );

res_x = bram_rpm_x_arr(x);
res_y = 10*(bram36_index);

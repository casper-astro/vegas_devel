### DESIGNS ###

**h16k_ver112c_1500.slx**
- Made small inconsequential changes from ver110_1080.slx to fix the timing errors. Timing reports are available in logs/*_system.twx

**h16k_ver110_1080.slx** 
- Updated blocks using update_casper_blocks(gcs) function to newer libraries (sma-wideband/mlib_devel: 404989f) to fix the ADC MMCM problem. Manually replaced software registers and other blocks that didn't succesfully update automatically. 

**h16k_ver110**
- Fixed bug in SSG block: 4 control bits were being taken from top (MSB) of ssg_ms_sel rather 
              than bottom (LSB) as intended

**h16k_ver107**
- Fix spec_tick interval to proper 2048 cycles. Built at NRAO

**h16k_ver106**
- Re-integrated fftwbr_core_ver101, added version control blocks
- built with ska-sa12a environment at NRAO
- Has error in spec_tick -> 4096 cycles instead of desired 2048. fixed in ver107

**h16k_ver105**
- Fixing order of the status bits

**h16k_ver104**
- This design has been updated with the new FFT core that has the 1/8 band fix (fftbr_core_ver101).

**h16k_ver101**
- Changed the order of the status bits to coincide with the
	      HPC code.  The ordering is now 3. BLANK 2. SR1 1. SR0 0. CAL

**fftwbr_core_ver101**
- This fft core has been redrawn with the 1/8 band fix latency in relational block in the mirror_spectrum script has been set to 1

### COMPILING ###

h16k_verXXX.mdl uses the "Black Boxing" trick to improve the development efficiency. The technique is described in the Casper Memo No.28(http://casper.berkeley.edu/wiki/images/a/a4/Black_box_memo.pdf).

~~Since some of the steps (Extract entity, place pcores in the design, etc) have been done already, so to compile h16k_ver100.mdl, there's no need to run through every steps described in the memo.~~ (This is true only if there is no change in the library that could lead to changes of the port data types, etc.)

To compile h16k_verXXX.mdl (using the provided pre-generated netlists):
1. Enter the directory vegas_devel/mdls/h16k
2. Unpack fftwbr_core.tar.gz
3. Unpack pfbfirr_core.tar.gz
4. Open newest h16k model file
5. Compile it as usual

Or going through the standard black-boxing routine:
1. Enter the directory vegas_devel/mdls/h16k
2. Open fftwbr_core_ver101.mdl, use update_casper_blocks(gcs) and ctrl+D to update the design.
3. Double click on the System Generator block, click the "Generate" button. Wait for the generation to complete.
4. In Matlab command window, run "extract_entity('fftwbr_core/fftwbr_core_ver101.ngc')"
5. Generate a new fftwbr_core_ver101_config.m as described in the black-box memo section 3.3 (page 6~7)
6. Open pfbfirr_core.mdl, update model as in step 1)
7. Double click on the System Generator block, click the "Generate" button. Wait for the generation to complete.
8. Extract entity, generate new *_config.m similarly as in step 3), 4) 
9. Open h16k_ver100.mdl
10. Compile it as with other regular mdl files.




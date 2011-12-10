This model runs at 375MHz, except for...
reshaper (removed)
speader (removed)
10Ge (runs at 333MHz, with good placement)

No functionality testing was performed on this model:  Matlab never generated good results in the first place.

Changes:
Reorganized the VACC inputs so that they were representative of the physical sources of the signals

Adjusted stage 2 so that the DSP had an opmode register, increased the initial value of the relevant counter by 1 to account for this.

Replaced stokes x timex x* sections with casper_library power blocks

Constraints and timing report included.
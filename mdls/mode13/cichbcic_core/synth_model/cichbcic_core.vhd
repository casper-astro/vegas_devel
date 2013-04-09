--------------------------------------------------------------------------------
--    This file is owned and controlled by Xilinx and must be used solely     --
--    for design, simulation, implementation and creation of design files     --
--    limited to Xilinx devices or technologies. Use with non-Xilinx          --
--    devices or technologies is expressly prohibited and immediately         --
--    terminates your license.                                                --
--                                                                            --
--    XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" SOLELY    --
--    FOR USE IN DEVELOPING PROGRAMS AND SOLUTIONS FOR XILINX DEVICES.  BY    --
--    PROVIDING THIS DESIGN, CODE, OR INFORMATION AS ONE POSSIBLE             --
--    IMPLEMENTATION OF THIS FEATURE, APPLICATION OR STANDARD, XILINX IS      --
--    MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION IS FREE FROM ANY      --
--    CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE FOR OBTAINING ANY       --
--    RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY       --
--    DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE   --
--    IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR          --
--    REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF         --
--    INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A   --
--    PARTICULAR PURPOSE.                                                     --
--                                                                            --
--    Xilinx products are not intended for use in life support appliances,    --
--    devices, or systems.  Use in such applications are expressly            --
--    prohibited.                                                             --
--                                                                            --
--    (c) Copyright 1995-2013 Xilinx, Inc.                                    --
--    All rights reserved.                                                    --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- You must compile the wrapper file mult_11_2_893416810381d560.vhd when simulating
-- the core, mult_11_2_893416810381d560. When compiling the wrapper file, be sure to
-- reference the XilinxCoreLib VHDL simulation library. For detailed
-- instructions, please refer to the "CORE Generator Help".

-- The synthesis directives "translate_off/translate_on" specified
-- below are supported by Xilinx, Mentor Graphics and Synplicity
-- synthesis tools. Ensure they are correct for your synthesis tool(s).

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
-- synthesis translate_off
LIBRARY XilinxCoreLib;
-- synthesis translate_on
ENTITY mult_11_2_893416810381d560 IS
  PORT (
    clk : IN STD_LOGIC;
    a : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
    ce : IN STD_LOGIC;
    sclr : IN STD_LOGIC;
    p : OUT STD_LOGIC_VECTOR(47 DOWNTO 0)
  );
END mult_11_2_893416810381d560;

ARCHITECTURE mult_11_2_893416810381d560_a OF mult_11_2_893416810381d560 IS
-- synthesis translate_off
COMPONENT wrapped_mult_11_2_893416810381d560
  PORT (
    clk : IN STD_LOGIC;
    a : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
    ce : IN STD_LOGIC;
    sclr : IN STD_LOGIC;
    p : OUT STD_LOGIC_VECTOR(47 DOWNTO 0)
  );
END COMPONENT;

-- Configuration specification
  FOR ALL : wrapped_mult_11_2_893416810381d560 USE ENTITY XilinxCoreLib.mult_gen_v11_2(behavioral)
    GENERIC MAP (
      c_a_type => 0,
      c_a_width => 16,
      c_b_type => 0,
      c_b_value => "10000001",
      c_b_width => 32,
      c_ccm_imp => 0,
      c_ce_overrides_sclr => 1,
      c_has_ce => 1,
      c_has_sclr => 1,
      c_has_zero_detect => 0,
      c_latency => 3,
      c_model_type => 0,
      c_mult_type => 1,
      c_optimize_goal => 1,
      c_out_high => 47,
      c_out_low => 0,
      c_round_output => 0,
      c_round_pt => 0,
      c_verbosity => 0,
      c_xdevicefamily => "virtex6"
    );
-- synthesis translate_on
BEGIN
-- synthesis translate_off
U0 : wrapped_mult_11_2_893416810381d560
  PORT MAP (
    clk => clk,
    a => a,
    b => b,
    ce => ce,
    sclr => sclr,
    p => p
  );
-- synthesis translate_on

END mult_11_2_893416810381d560_a;
--------------------------------------------------------------------------------
--    This file is owned and controlled by Xilinx and must be used solely     --
--    for design, simulation, implementation and creation of design files     --
--    limited to Xilinx devices or technologies. Use with non-Xilinx          --
--    devices or technologies is expressly prohibited and immediately         --
--    terminates your license.                                                --
--                                                                            --
--    XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" SOLELY    --
--    FOR USE IN DEVELOPING PROGRAMS AND SOLUTIONS FOR XILINX DEVICES.  BY    --
--    PROVIDING THIS DESIGN, CODE, OR INFORMATION AS ONE POSSIBLE             --
--    IMPLEMENTATION OF THIS FEATURE, APPLICATION OR STANDARD, XILINX IS      --
--    MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION IS FREE FROM ANY      --
--    CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE FOR OBTAINING ANY       --
--    RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY       --
--    DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE   --
--    IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR          --
--    REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF         --
--    INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A   --
--    PARTICULAR PURPOSE.                                                     --
--                                                                            --
--    Xilinx products are not intended for use in life support appliances,    --
--    devices, or systems.  Use in such applications are expressly            --
--    prohibited.                                                             --
--                                                                            --
--    (c) Copyright 1995-2013 Xilinx, Inc.                                    --
--    All rights reserved.                                                    --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- You must compile the wrapper file addsb_11_0_e35a3bf39f366fd8.vhd when simulating
-- the core, addsb_11_0_e35a3bf39f366fd8. When compiling the wrapper file, be sure to
-- reference the XilinxCoreLib VHDL simulation library. For detailed
-- instructions, please refer to the "CORE Generator Help".

-- The synthesis directives "translate_off/translate_on" specified
-- below are supported by Xilinx, Mentor Graphics and Synplicity
-- synthesis tools. Ensure they are correct for your synthesis tool(s).

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
-- synthesis translate_off
LIBRARY XilinxCoreLib;
-- synthesis translate_on
ENTITY addsb_11_0_e35a3bf39f366fd8 IS
  PORT (
    a : IN STD_LOGIC_VECTOR(49 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(49 DOWNTO 0);
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    s : OUT STD_LOGIC_VECTOR(49 DOWNTO 0)
  );
END addsb_11_0_e35a3bf39f366fd8;

ARCHITECTURE addsb_11_0_e35a3bf39f366fd8_a OF addsb_11_0_e35a3bf39f366fd8 IS
-- synthesis translate_off
COMPONENT wrapped_addsb_11_0_e35a3bf39f366fd8
  PORT (
    a : IN STD_LOGIC_VECTOR(49 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(49 DOWNTO 0);
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    s : OUT STD_LOGIC_VECTOR(49 DOWNTO 0)
  );
END COMPONENT;

-- Configuration specification
  FOR ALL : wrapped_addsb_11_0_e35a3bf39f366fd8 USE ENTITY XilinxCoreLib.c_addsub_v11_0(behavioral)
    GENERIC MAP (
      c_a_type => 0,
      c_a_width => 50,
      c_add_mode => 0,
      c_ainit_val => "0",
      c_b_constant => 0,
      c_b_type => 0,
      c_b_value => "00000000000000000000000000000000000000000000000000",
      c_b_width => 50,
      c_borrow_low => 1,
      c_bypass_low => 0,
      c_ce_overrides_bypass => 1,
      c_ce_overrides_sclr => 0,
      c_has_bypass => 0,
      c_has_c_in => 0,
      c_has_c_out => 0,
      c_has_ce => 1,
      c_has_sclr => 0,
      c_has_sinit => 0,
      c_has_sset => 0,
      c_implementation => 0,
      c_latency => 1,
      c_out_width => 50,
      c_sclr_overrides_sset => 1,
      c_sinit_val => "0",
      c_verbosity => 0,
      c_xdevicefamily => "virtex6"
    );
-- synthesis translate_on
BEGIN
-- synthesis translate_off
U0 : wrapped_addsb_11_0_e35a3bf39f366fd8
  PORT MAP (
    a => a,
    b => b,
    clk => clk,
    ce => ce,
    s => s
  );
-- synthesis translate_on

END addsb_11_0_e35a3bf39f366fd8_a;
--------------------------------------------------------------------------------
--    This file is owned and controlled by Xilinx and must be used solely     --
--    for design, simulation, implementation and creation of design files     --
--    limited to Xilinx devices or technologies. Use with non-Xilinx          --
--    devices or technologies is expressly prohibited and immediately         --
--    terminates your license.                                                --
--                                                                            --
--    XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" SOLELY    --
--    FOR USE IN DEVELOPING PROGRAMS AND SOLUTIONS FOR XILINX DEVICES.  BY    --
--    PROVIDING THIS DESIGN, CODE, OR INFORMATION AS ONE POSSIBLE             --
--    IMPLEMENTATION OF THIS FEATURE, APPLICATION OR STANDARD, XILINX IS      --
--    MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION IS FREE FROM ANY      --
--    CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE FOR OBTAINING ANY       --
--    RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY       --
--    DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE   --
--    IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR          --
--    REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF         --
--    INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A   --
--    PARTICULAR PURPOSE.                                                     --
--                                                                            --
--    Xilinx products are not intended for use in life support appliances,    --
--    devices, or systems.  Use in such applications are expressly            --
--    prohibited.                                                             --
--                                                                            --
--    (c) Copyright 1995-2013 Xilinx, Inc.                                    --
--    All rights reserved.                                                    --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- You must compile the wrapper file cntr_11_0_6400a835b899f648.vhd when simulating
-- the core, cntr_11_0_6400a835b899f648. When compiling the wrapper file, be sure to
-- reference the XilinxCoreLib VHDL simulation library. For detailed
-- instructions, please refer to the "CORE Generator Help".

-- The synthesis directives "translate_off/translate_on" specified
-- below are supported by Xilinx, Mentor Graphics and Synplicity
-- synthesis tools. Ensure they are correct for your synthesis tool(s).

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
-- synthesis translate_off
LIBRARY XilinxCoreLib;
-- synthesis translate_on
ENTITY cntr_11_0_6400a835b899f648 IS
  PORT (
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    sinit : IN STD_LOGIC;
    q : OUT STD_LOGIC_VECTOR(2 DOWNTO 0)
  );
END cntr_11_0_6400a835b899f648;

ARCHITECTURE cntr_11_0_6400a835b899f648_a OF cntr_11_0_6400a835b899f648 IS
-- synthesis translate_off
COMPONENT wrapped_cntr_11_0_6400a835b899f648
  PORT (
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    sinit : IN STD_LOGIC;
    q : OUT STD_LOGIC_VECTOR(2 DOWNTO 0)
  );
END COMPONENT;

-- Configuration specification
  FOR ALL : wrapped_cntr_11_0_6400a835b899f648 USE ENTITY XilinxCoreLib.c_counter_binary_v11_0(behavioral)
    GENERIC MAP (
      c_ainit_val => "0",
      c_ce_overrides_sync => 0,
      c_count_by => "1",
      c_count_mode => 1,
      c_count_to => "1",
      c_fb_latency => 0,
      c_has_ce => 1,
      c_has_load => 0,
      c_has_sclr => 0,
      c_has_sinit => 1,
      c_has_sset => 0,
      c_has_thresh0 => 0,
      c_implementation => 0,
      c_latency => 1,
      c_load_low => 0,
      c_restrict_count => 0,
      c_sclr_overrides_sset => 1,
      c_sinit_val => "111",
      c_thresh0_value => "1",
      c_verbosity => 0,
      c_width => 3,
      c_xdevicefamily => "virtex6"
    );
-- synthesis translate_on
BEGIN
-- synthesis translate_off
U0 : wrapped_cntr_11_0_6400a835b899f648
  PORT MAP (
    clk => clk,
    ce => ce,
    sinit => sinit,
    q => q
  );
-- synthesis translate_on

END cntr_11_0_6400a835b899f648_a;
--------------------------------------------------------------------------------
--    This file is owned and controlled by Xilinx and must be used solely     --
--    for design, simulation, implementation and creation of design files     --
--    limited to Xilinx devices or technologies. Use with non-Xilinx          --
--    devices or technologies is expressly prohibited and immediately         --
--    terminates your license.                                                --
--                                                                            --
--    XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" SOLELY    --
--    FOR USE IN DEVELOPING PROGRAMS AND SOLUTIONS FOR XILINX DEVICES.  BY    --
--    PROVIDING THIS DESIGN, CODE, OR INFORMATION AS ONE POSSIBLE             --
--    IMPLEMENTATION OF THIS FEATURE, APPLICATION OR STANDARD, XILINX IS      --
--    MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION IS FREE FROM ANY      --
--    CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE FOR OBTAINING ANY       --
--    RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY       --
--    DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE   --
--    IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR          --
--    REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF         --
--    INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A   --
--    PARTICULAR PURPOSE.                                                     --
--                                                                            --
--    Xilinx products are not intended for use in life support appliances,    --
--    devices, or systems.  Use in such applications are expressly            --
--    prohibited.                                                             --
--                                                                            --
--    (c) Copyright 1995-2013 Xilinx, Inc.                                    --
--    All rights reserved.                                                    --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- You must compile the wrapper file addsb_11_0_5bfb73f1589643d3.vhd when simulating
-- the core, addsb_11_0_5bfb73f1589643d3. When compiling the wrapper file, be sure to
-- reference the XilinxCoreLib VHDL simulation library. For detailed
-- instructions, please refer to the "CORE Generator Help".

-- The synthesis directives "translate_off/translate_on" specified
-- below are supported by Xilinx, Mentor Graphics and Synplicity
-- synthesis tools. Ensure they are correct for your synthesis tool(s).

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
-- synthesis translate_off
LIBRARY XilinxCoreLib;
-- synthesis translate_on
ENTITY addsb_11_0_5bfb73f1589643d3 IS
  PORT (
    a : IN STD_LOGIC_VECTOR(51 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(51 DOWNTO 0);
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    s : OUT STD_LOGIC_VECTOR(51 DOWNTO 0)
  );
END addsb_11_0_5bfb73f1589643d3;

ARCHITECTURE addsb_11_0_5bfb73f1589643d3_a OF addsb_11_0_5bfb73f1589643d3 IS
-- synthesis translate_off
COMPONENT wrapped_addsb_11_0_5bfb73f1589643d3
  PORT (
    a : IN STD_LOGIC_VECTOR(51 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(51 DOWNTO 0);
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    s : OUT STD_LOGIC_VECTOR(51 DOWNTO 0)
  );
END COMPONENT;

-- Configuration specification
  FOR ALL : wrapped_addsb_11_0_5bfb73f1589643d3 USE ENTITY XilinxCoreLib.c_addsub_v11_0(behavioral)
    GENERIC MAP (
      c_a_type => 0,
      c_a_width => 52,
      c_add_mode => 0,
      c_ainit_val => "0",
      c_b_constant => 0,
      c_b_type => 0,
      c_b_value => "0000000000000000000000000000000000000000000000000000",
      c_b_width => 52,
      c_borrow_low => 1,
      c_bypass_low => 0,
      c_ce_overrides_bypass => 1,
      c_ce_overrides_sclr => 0,
      c_has_bypass => 0,
      c_has_c_in => 0,
      c_has_c_out => 0,
      c_has_ce => 1,
      c_has_sclr => 0,
      c_has_sinit => 0,
      c_has_sset => 0,
      c_implementation => 0,
      c_latency => 1,
      c_out_width => 52,
      c_sclr_overrides_sset => 1,
      c_sinit_val => "0",
      c_verbosity => 0,
      c_xdevicefamily => "virtex6"
    );
-- synthesis translate_on
BEGIN
-- synthesis translate_off
U0 : wrapped_addsb_11_0_5bfb73f1589643d3
  PORT MAP (
    a => a,
    b => b,
    clk => clk,
    ce => ce,
    s => s
  );
-- synthesis translate_on

END addsb_11_0_5bfb73f1589643d3_a;
--------------------------------------------------------------------------------
--    This file is owned and controlled by Xilinx and must be used solely     --
--    for design, simulation, implementation and creation of design files     --
--    limited to Xilinx devices or technologies. Use with non-Xilinx          --
--    devices or technologies is expressly prohibited and immediately         --
--    terminates your license.                                                --
--                                                                            --
--    XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" SOLELY    --
--    FOR USE IN DEVELOPING PROGRAMS AND SOLUTIONS FOR XILINX DEVICES.  BY    --
--    PROVIDING THIS DESIGN, CODE, OR INFORMATION AS ONE POSSIBLE             --
--    IMPLEMENTATION OF THIS FEATURE, APPLICATION OR STANDARD, XILINX IS      --
--    MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION IS FREE FROM ANY      --
--    CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE FOR OBTAINING ANY       --
--    RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY       --
--    DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE   --
--    IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR          --
--    REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF         --
--    INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A   --
--    PARTICULAR PURPOSE.                                                     --
--                                                                            --
--    Xilinx products are not intended for use in life support appliances,    --
--    devices, or systems.  Use in such applications are expressly            --
--    prohibited.                                                             --
--                                                                            --
--    (c) Copyright 1995-2013 Xilinx, Inc.                                    --
--    All rights reserved.                                                    --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- You must compile the wrapper file addsb_11_0_5de09ee679db1560.vhd when simulating
-- the core, addsb_11_0_5de09ee679db1560. When compiling the wrapper file, be sure to
-- reference the XilinxCoreLib VHDL simulation library. For detailed
-- instructions, please refer to the "CORE Generator Help".

-- The synthesis directives "translate_off/translate_on" specified
-- below are supported by Xilinx, Mentor Graphics and Synplicity
-- synthesis tools. Ensure they are correct for your synthesis tool(s).

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
-- synthesis translate_off
LIBRARY XilinxCoreLib;
-- synthesis translate_on
ENTITY addsb_11_0_5de09ee679db1560 IS
  PORT (
    a : IN STD_LOGIC_VECTOR(50 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(50 DOWNTO 0);
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    s : OUT STD_LOGIC_VECTOR(50 DOWNTO 0)
  );
END addsb_11_0_5de09ee679db1560;

ARCHITECTURE addsb_11_0_5de09ee679db1560_a OF addsb_11_0_5de09ee679db1560 IS
-- synthesis translate_off
COMPONENT wrapped_addsb_11_0_5de09ee679db1560
  PORT (
    a : IN STD_LOGIC_VECTOR(50 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(50 DOWNTO 0);
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    s : OUT STD_LOGIC_VECTOR(50 DOWNTO 0)
  );
END COMPONENT;

-- Configuration specification
  FOR ALL : wrapped_addsb_11_0_5de09ee679db1560 USE ENTITY XilinxCoreLib.c_addsub_v11_0(behavioral)
    GENERIC MAP (
      c_a_type => 0,
      c_a_width => 51,
      c_add_mode => 0,
      c_ainit_val => "0",
      c_b_constant => 0,
      c_b_type => 0,
      c_b_value => "000000000000000000000000000000000000000000000000000",
      c_b_width => 51,
      c_borrow_low => 1,
      c_bypass_low => 0,
      c_ce_overrides_bypass => 1,
      c_ce_overrides_sclr => 0,
      c_has_bypass => 0,
      c_has_c_in => 0,
      c_has_c_out => 0,
      c_has_ce => 1,
      c_has_sclr => 0,
      c_has_sinit => 0,
      c_has_sset => 0,
      c_implementation => 0,
      c_latency => 1,
      c_out_width => 51,
      c_sclr_overrides_sset => 1,
      c_sinit_val => "0",
      c_verbosity => 0,
      c_xdevicefamily => "virtex6"
    );
-- synthesis translate_on
BEGIN
-- synthesis translate_off
U0 : wrapped_addsb_11_0_5de09ee679db1560
  PORT MAP (
    a => a,
    b => b,
    clk => clk,
    ce => ce,
    s => s
  );
-- synthesis translate_on

END addsb_11_0_5de09ee679db1560_a;
--------------------------------------------------------------------------------
--    This file is owned and controlled by Xilinx and must be used solely     --
--    for design, simulation, implementation and creation of design files     --
--    limited to Xilinx devices or technologies. Use with non-Xilinx          --
--    devices or technologies is expressly prohibited and immediately         --
--    terminates your license.                                                --
--                                                                            --
--    XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" SOLELY    --
--    FOR USE IN DEVELOPING PROGRAMS AND SOLUTIONS FOR XILINX DEVICES.  BY    --
--    PROVIDING THIS DESIGN, CODE, OR INFORMATION AS ONE POSSIBLE             --
--    IMPLEMENTATION OF THIS FEATURE, APPLICATION OR STANDARD, XILINX IS      --
--    MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION IS FREE FROM ANY      --
--    CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE FOR OBTAINING ANY       --
--    RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY       --
--    DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE   --
--    IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR          --
--    REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF         --
--    INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A   --
--    PARTICULAR PURPOSE.                                                     --
--                                                                            --
--    Xilinx products are not intended for use in life support appliances,    --
--    devices, or systems.  Use in such applications are expressly            --
--    prohibited.                                                             --
--                                                                            --
--    (c) Copyright 1995-2013 Xilinx, Inc.                                    --
--    All rights reserved.                                                    --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- You must compile the wrapper file addsb_11_0_a6179ec1a236388e.vhd when simulating
-- the core, addsb_11_0_a6179ec1a236388e. When compiling the wrapper file, be sure to
-- reference the XilinxCoreLib VHDL simulation library. For detailed
-- instructions, please refer to the "CORE Generator Help".

-- The synthesis directives "translate_off/translate_on" specified
-- below are supported by Xilinx, Mentor Graphics and Synplicity
-- synthesis tools. Ensure they are correct for your synthesis tool(s).

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
-- synthesis translate_off
LIBRARY XilinxCoreLib;
-- synthesis translate_on
ENTITY addsb_11_0_a6179ec1a236388e IS
  PORT (
    a : IN STD_LOGIC_VECTOR(48 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(48 DOWNTO 0);
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    s : OUT STD_LOGIC_VECTOR(48 DOWNTO 0)
  );
END addsb_11_0_a6179ec1a236388e;

ARCHITECTURE addsb_11_0_a6179ec1a236388e_a OF addsb_11_0_a6179ec1a236388e IS
-- synthesis translate_off
COMPONENT wrapped_addsb_11_0_a6179ec1a236388e
  PORT (
    a : IN STD_LOGIC_VECTOR(48 DOWNTO 0);
    b : IN STD_LOGIC_VECTOR(48 DOWNTO 0);
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    s : OUT STD_LOGIC_VECTOR(48 DOWNTO 0)
  );
END COMPONENT;

-- Configuration specification
  FOR ALL : wrapped_addsb_11_0_a6179ec1a236388e USE ENTITY XilinxCoreLib.c_addsub_v11_0(behavioral)
    GENERIC MAP (
      c_a_type => 0,
      c_a_width => 49,
      c_add_mode => 0,
      c_ainit_val => "0",
      c_b_constant => 0,
      c_b_type => 0,
      c_b_value => "0000000000000000000000000000000000000000000000000",
      c_b_width => 49,
      c_borrow_low => 1,
      c_bypass_low => 0,
      c_ce_overrides_bypass => 1,
      c_ce_overrides_sclr => 0,
      c_has_bypass => 0,
      c_has_c_in => 0,
      c_has_c_out => 0,
      c_has_ce => 1,
      c_has_sclr => 0,
      c_has_sinit => 0,
      c_has_sset => 0,
      c_implementation => 0,
      c_latency => 1,
      c_out_width => 49,
      c_sclr_overrides_sset => 1,
      c_sinit_val => "0",
      c_verbosity => 0,
      c_xdevicefamily => "virtex6"
    );
-- synthesis translate_on
BEGIN
-- synthesis translate_off
U0 : wrapped_addsb_11_0_a6179ec1a236388e
  PORT MAP (
    a => a,
    b => b,
    clk => clk,
    ce => ce,
    s => s
  );
-- synthesis translate_on

END addsb_11_0_a6179ec1a236388e_a;
--------------------------------------------------------------------------------
--    This file is owned and controlled by Xilinx and must be used solely     --
--    for design, simulation, implementation and creation of design files     --
--    limited to Xilinx devices or technologies. Use with non-Xilinx          --
--    devices or technologies is expressly prohibited and immediately         --
--    terminates your license.                                                --
--                                                                            --
--    XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" SOLELY    --
--    FOR USE IN DEVELOPING PROGRAMS AND SOLUTIONS FOR XILINX DEVICES.  BY    --
--    PROVIDING THIS DESIGN, CODE, OR INFORMATION AS ONE POSSIBLE             --
--    IMPLEMENTATION OF THIS FEATURE, APPLICATION OR STANDARD, XILINX IS      --
--    MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION IS FREE FROM ANY      --
--    CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE FOR OBTAINING ANY       --
--    RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY       --
--    DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE   --
--    IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR          --
--    REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF         --
--    INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A   --
--    PARTICULAR PURPOSE.                                                     --
--                                                                            --
--    Xilinx products are not intended for use in life support appliances,    --
--    devices, or systems.  Use in such applications are expressly            --
--    prohibited.                                                             --
--                                                                            --
--    (c) Copyright 1995-2013 Xilinx, Inc.                                    --
--    All rights reserved.                                                    --
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- You must compile the wrapper file cntr_11_0_a0d692c18ceb283e.vhd when simulating
-- the core, cntr_11_0_a0d692c18ceb283e. When compiling the wrapper file, be sure to
-- reference the XilinxCoreLib VHDL simulation library. For detailed
-- instructions, please refer to the "CORE Generator Help".

-- The synthesis directives "translate_off/translate_on" specified
-- below are supported by Xilinx, Mentor Graphics and Synplicity
-- synthesis tools. Ensure they are correct for your synthesis tool(s).

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
-- synthesis translate_off
LIBRARY XilinxCoreLib;
-- synthesis translate_on
ENTITY cntr_11_0_a0d692c18ceb283e IS
  PORT (
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    sinit : IN STD_LOGIC;
    q : OUT STD_LOGIC_VECTOR(1 DOWNTO 0)
  );
END cntr_11_0_a0d692c18ceb283e;

ARCHITECTURE cntr_11_0_a0d692c18ceb283e_a OF cntr_11_0_a0d692c18ceb283e IS
-- synthesis translate_off
COMPONENT wrapped_cntr_11_0_a0d692c18ceb283e
  PORT (
    clk : IN STD_LOGIC;
    ce : IN STD_LOGIC;
    sinit : IN STD_LOGIC;
    q : OUT STD_LOGIC_VECTOR(1 DOWNTO 0)
  );
END COMPONENT;

-- Configuration specification
  FOR ALL : wrapped_cntr_11_0_a0d692c18ceb283e USE ENTITY XilinxCoreLib.c_counter_binary_v11_0(behavioral)
    GENERIC MAP (
      c_ainit_val => "0",
      c_ce_overrides_sync => 0,
      c_count_by => "1",
      c_count_mode => 1,
      c_count_to => "1",
      c_fb_latency => 0,
      c_has_ce => 1,
      c_has_load => 0,
      c_has_sclr => 0,
      c_has_sinit => 1,
      c_has_sset => 0,
      c_has_thresh0 => 0,
      c_implementation => 0,
      c_latency => 1,
      c_load_low => 0,
      c_restrict_count => 0,
      c_sclr_overrides_sset => 1,
      c_sinit_val => "11",
      c_thresh0_value => "1",
      c_verbosity => 0,
      c_width => 2,
      c_xdevicefamily => "virtex6"
    );
-- synthesis translate_on
BEGIN
-- synthesis translate_off
U0 : wrapped_cntr_11_0_a0d692c18ceb283e
  PORT MAP (
    clk => clk,
    ce => ce,
    sinit => sinit,
    q => q
  );
-- synthesis translate_on

END cntr_11_0_a0d692c18ceb283e_a;

-------------------------------------------------------------------
-- System Generator version 14.2 VHDL source file.
--
-- Copyright(C) 2012 by Xilinx, Inc.  All rights reserved.  This
-- text/file contains proprietary, confidential information of Xilinx,
-- Inc., is distributed under license from Xilinx, Inc., and may be used,
-- copied and/or disclosed only pursuant to the terms of a valid license
-- agreement with Xilinx, Inc.  Xilinx hereby grants you a license to use
-- this text/file solely for design, simulation, implementation and
-- creation of design files limited to Xilinx devices or technologies.
-- Use with non-Xilinx devices or technologies is expressly prohibited
-- and immediately terminates your license unless covered by a separate
-- agreement.
--
-- Xilinx is providing this design, code, or information "as is" solely
-- for use in developing programs and solutions for Xilinx devices.  By
-- providing this design, code, or information as one possible
-- implementation of this feature, application or standard, Xilinx is
-- making no representation that this implementation is free from any
-- claims of infringement.  You are responsible for obtaining any rights
-- you may require for your implementation.  Xilinx expressly disclaims
-- any warranty whatsoever with respect to the adequacy of the
-- implementation, including but not limited to warranties of
-- merchantability or fitness for a particular purpose.
--
-- Xilinx products are not intended for use in life support appliances,
-- devices, or systems.  Use in such applications is expressly prohibited.
--
-- Any modifications that are made to the source code are done at the user's
-- sole risk and will be unsupported.
--
-- This copyright and support notice must be retained as part of this
-- text at all times.  (c) Copyright 1995-2012 Xilinx, Inc.  All rights
-- reserved.
-------------------------------------------------------------------
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
package conv_pkg is
    constant simulating : boolean := false
      -- synopsys translate_off
        or true
      -- synopsys translate_on
    ;
    constant xlUnsigned : integer := 1;
    constant xlSigned : integer := 2;
    constant xlFloat : integer := 3;
    constant xlWrap : integer := 1;
    constant xlSaturate : integer := 2;
    constant xlTruncate : integer := 1;
    constant xlRound : integer := 2;
    constant xlRoundBanker : integer := 3;
    constant xlAddMode : integer := 1;
    constant xlSubMode : integer := 2;
    attribute black_box : boolean;
    attribute syn_black_box : boolean;
    attribute fpga_dont_touch: string;
    attribute box_type :  string;
    attribute keep : string;
    attribute syn_keep : boolean;
    function std_logic_vector_to_unsigned(inp : std_logic_vector) return unsigned;
    function unsigned_to_std_logic_vector(inp : unsigned) return std_logic_vector;
    function std_logic_vector_to_signed(inp : std_logic_vector) return signed;
    function signed_to_std_logic_vector(inp : signed) return std_logic_vector;
    function unsigned_to_signed(inp : unsigned) return signed;
    function signed_to_unsigned(inp : signed) return unsigned;
    function pos(inp : std_logic_vector; arith : INTEGER) return boolean;
    function all_same(inp: std_logic_vector) return boolean;
    function all_zeros(inp: std_logic_vector) return boolean;
    function is_point_five(inp: std_logic_vector) return boolean;
    function all_ones(inp: std_logic_vector) return boolean;
    function convert_type (inp : std_logic_vector; old_width, old_bin_pt,
                           old_arith, new_width, new_bin_pt, new_arith,
                           quantization, overflow : INTEGER)
        return std_logic_vector;
    function cast (inp : std_logic_vector; old_bin_pt,
                   new_width, new_bin_pt, new_arith : INTEGER)
        return std_logic_vector;
    function shift_division_result(quotient, fraction: std_logic_vector;
                                   fraction_width, shift_value, shift_dir: INTEGER)
        return std_logic_vector;
    function shift_op (inp: std_logic_vector;
                       result_width, shift_value, shift_dir: INTEGER)
        return std_logic_vector;
    function vec_slice (inp : std_logic_vector; upper, lower : INTEGER)
        return std_logic_vector;
    function s2u_slice (inp : signed; upper, lower : INTEGER)
        return unsigned;
    function u2u_slice (inp : unsigned; upper, lower : INTEGER)
        return unsigned;
    function s2s_cast (inp : signed; old_bin_pt,
                   new_width, new_bin_pt : INTEGER)
        return signed;
    function u2s_cast (inp : unsigned; old_bin_pt,
                   new_width, new_bin_pt : INTEGER)
        return signed;
    function s2u_cast (inp : signed; old_bin_pt,
                   new_width, new_bin_pt : INTEGER)
        return unsigned;
    function u2u_cast (inp : unsigned; old_bin_pt,
                   new_width, new_bin_pt : INTEGER)
        return unsigned;
    function u2v_cast (inp : unsigned; old_bin_pt,
                   new_width, new_bin_pt : INTEGER)
        return std_logic_vector;
    function s2v_cast (inp : signed; old_bin_pt,
                   new_width, new_bin_pt : INTEGER)
        return std_logic_vector;
    function trunc (inp : std_logic_vector; old_width, old_bin_pt, old_arith,
                    new_width, new_bin_pt, new_arith : INTEGER)
        return std_logic_vector;
    function round_towards_inf (inp : std_logic_vector; old_width, old_bin_pt,
                                old_arith, new_width, new_bin_pt,
                                new_arith : INTEGER) return std_logic_vector;
    function round_towards_even (inp : std_logic_vector; old_width, old_bin_pt,
                                old_arith, new_width, new_bin_pt,
                                new_arith : INTEGER) return std_logic_vector;
    function max_signed(width : INTEGER) return std_logic_vector;
    function min_signed(width : INTEGER) return std_logic_vector;
    function saturation_arith(inp:  std_logic_vector;  old_width, old_bin_pt,
                              old_arith, new_width, new_bin_pt, new_arith
                              : INTEGER) return std_logic_vector;
    function wrap_arith(inp:  std_logic_vector;  old_width, old_bin_pt,
                        old_arith, new_width, new_bin_pt, new_arith : INTEGER)
                        return std_logic_vector;
    function fractional_bits(a_bin_pt, b_bin_pt: INTEGER) return INTEGER;
    function integer_bits(a_width, a_bin_pt, b_width, b_bin_pt: INTEGER)
        return INTEGER;
    function sign_ext(inp : std_logic_vector; new_width : INTEGER)
        return std_logic_vector;
    function zero_ext(inp : std_logic_vector; new_width : INTEGER)
        return std_logic_vector;
    function zero_ext(inp : std_logic; new_width : INTEGER)
        return std_logic_vector;
    function extend_MSB(inp : std_logic_vector; new_width, arith : INTEGER)
        return std_logic_vector;
    function align_input(inp : std_logic_vector; old_width, delta, new_arith,
                          new_width: INTEGER)
        return std_logic_vector;
    function pad_LSB(inp : std_logic_vector; new_width: integer)
        return std_logic_vector;
    function pad_LSB(inp : std_logic_vector; new_width, arith : integer)
        return std_logic_vector;
    function max(L, R: INTEGER) return INTEGER;
    function min(L, R: INTEGER) return INTEGER;
    function "="(left,right: STRING) return boolean;
    function boolean_to_signed (inp : boolean; width: integer)
        return signed;
    function boolean_to_unsigned (inp : boolean; width: integer)
        return unsigned;
    function boolean_to_vector (inp : boolean)
        return std_logic_vector;
    function std_logic_to_vector (inp : std_logic)
        return std_logic_vector;
    function integer_to_std_logic_vector (inp : integer;  width, arith : integer)
        return std_logic_vector;
    function std_logic_vector_to_integer (inp : std_logic_vector;  arith : integer)
        return integer;
    function std_logic_to_integer(constant inp : std_logic := '0')
        return integer;
    function bin_string_element_to_std_logic_vector (inp : string;  width, index : integer)
        return std_logic_vector;
    function bin_string_to_std_logic_vector (inp : string)
        return std_logic_vector;
    function hex_string_to_std_logic_vector (inp : string; width : integer)
        return std_logic_vector;
    function makeZeroBinStr (width : integer) return STRING;
    function and_reduce(inp: std_logic_vector) return std_logic;
    -- synopsys translate_off
    function is_binary_string_invalid (inp : string)
        return boolean;
    function is_binary_string_undefined (inp : string)
        return boolean;
    function is_XorU(inp : std_logic_vector)
        return boolean;
    function to_real(inp : std_logic_vector; bin_pt : integer; arith : integer)
        return real;
    function std_logic_to_real(inp : std_logic; bin_pt : integer; arith : integer)
        return real;
    function real_to_std_logic_vector (inp : real;  width, bin_pt, arith : integer)
        return std_logic_vector;
    function real_string_to_std_logic_vector (inp : string;  width, bin_pt, arith : integer)
        return std_logic_vector;
    constant display_precision : integer := 20;
    function real_to_string (inp : real) return string;
    function valid_bin_string(inp : string) return boolean;
    function std_logic_vector_to_bin_string(inp : std_logic_vector) return string;
    function std_logic_to_bin_string(inp : std_logic) return string;
    function std_logic_vector_to_bin_string_w_point(inp : std_logic_vector; bin_pt : integer)
        return string;
    function real_to_bin_string(inp : real;  width, bin_pt, arith : integer)
        return string;
    type stdlogic_to_char_t is array(std_logic) of character;
    constant to_char : stdlogic_to_char_t := (
        'U' => 'U',
        'X' => 'X',
        '0' => '0',
        '1' => '1',
        'Z' => 'Z',
        'W' => 'W',
        'L' => 'L',
        'H' => 'H',
        '-' => '-');
    -- synopsys translate_on
end conv_pkg;
package body conv_pkg is
    function std_logic_vector_to_unsigned(inp : std_logic_vector)
        return unsigned
    is
    begin
        return unsigned (inp);
    end;
    function unsigned_to_std_logic_vector(inp : unsigned)
        return std_logic_vector
    is
    begin
        return std_logic_vector(inp);
    end;
    function std_logic_vector_to_signed(inp : std_logic_vector)
        return signed
    is
    begin
        return  signed (inp);
    end;
    function signed_to_std_logic_vector(inp : signed)
        return std_logic_vector
    is
    begin
        return std_logic_vector(inp);
    end;
    function unsigned_to_signed (inp : unsigned)
        return signed
    is
    begin
        return signed(std_logic_vector(inp));
    end;
    function signed_to_unsigned (inp : signed)
        return unsigned
    is
    begin
        return unsigned(std_logic_vector(inp));
    end;
    function pos(inp : std_logic_vector; arith : INTEGER)
        return boolean
    is
        constant width : integer := inp'length;
        variable vec : std_logic_vector(width-1 downto 0);
    begin
        vec := inp;
        if arith = xlUnsigned then
            return true;
        else
            if vec(width-1) = '0' then
                return true;
            else
                return false;
            end if;
        end if;
        return true;
    end;
    function max_signed(width : INTEGER)
        return std_logic_vector
    is
        variable ones : std_logic_vector(width-2 downto 0);
        variable result : std_logic_vector(width-1 downto 0);
    begin
        ones := (others => '1');
        result(width-1) := '0';
        result(width-2 downto 0) := ones;
        return result;
    end;
    function min_signed(width : INTEGER)
        return std_logic_vector
    is
        variable zeros : std_logic_vector(width-2 downto 0);
        variable result : std_logic_vector(width-1 downto 0);
    begin
        zeros := (others => '0');
        result(width-1) := '1';
        result(width-2 downto 0) := zeros;
        return result;
    end;
    function and_reduce(inp: std_logic_vector) return std_logic
    is
        variable result: std_logic;
        constant width : integer := inp'length;
        variable vec : std_logic_vector(width-1 downto 0);
    begin
        vec := inp;
        result := vec(0);
        if width > 1 then
            for i in 1 to width-1 loop
                result := result and vec(i);
            end loop;
        end if;
        return result;
    end;
    function all_same(inp: std_logic_vector) return boolean
    is
        variable result: boolean;
        constant width : integer := inp'length;
        variable vec : std_logic_vector(width-1 downto 0);
    begin
        vec := inp;
        result := true;
        if width > 0 then
            for i in 1 to width-1 loop
                if vec(i) /= vec(0) then
                    result := false;
                end if;
            end loop;
        end if;
        return result;
    end;
    function all_zeros(inp: std_logic_vector)
        return boolean
    is
        constant width : integer := inp'length;
        variable vec : std_logic_vector(width-1 downto 0);
        variable zero : std_logic_vector(width-1 downto 0);
        variable result : boolean;
    begin
        zero := (others => '0');
        vec := inp;
        -- synopsys translate_off
        if (is_XorU(vec)) then
            return false;
        end if;
         -- synopsys translate_on
        if (std_logic_vector_to_unsigned(vec) = std_logic_vector_to_unsigned(zero)) then
            result := true;
        else
            result := false;
        end if;
        return result;
    end;
    function is_point_five(inp: std_logic_vector)
        return boolean
    is
        constant width : integer := inp'length;
        variable vec : std_logic_vector(width-1 downto 0);
        variable result : boolean;
    begin
        vec := inp;
        -- synopsys translate_off
        if (is_XorU(vec)) then
            return false;
        end if;
         -- synopsys translate_on
        if (width > 1) then
           if ((vec(width-1) = '1') and (all_zeros(vec(width-2 downto 0)) = true)) then
               result := true;
           else
               result := false;
           end if;
        else
           if (vec(width-1) = '1') then
               result := true;
           else
               result := false;
           end if;
        end if;
        return result;
    end;
    function all_ones(inp: std_logic_vector)
        return boolean
    is
        constant width : integer := inp'length;
        variable vec : std_logic_vector(width-1 downto 0);
        variable one : std_logic_vector(width-1 downto 0);
        variable result : boolean;
    begin
        one := (others => '1');
        vec := inp;
        -- synopsys translate_off
        if (is_XorU(vec)) then
            return false;
        end if;
         -- synopsys translate_on
        if (std_logic_vector_to_unsigned(vec) = std_logic_vector_to_unsigned(one)) then
            result := true;
        else
            result := false;
        end if;
        return result;
    end;
    function full_precision_num_width(quantization, overflow, old_width,
                                      old_bin_pt, old_arith,
                                      new_width, new_bin_pt, new_arith : INTEGER)
        return integer
    is
        variable result : integer;
    begin
        result := old_width + 2;
        return result;
    end;
    function quantized_num_width(quantization, overflow, old_width, old_bin_pt,
                                 old_arith, new_width, new_bin_pt, new_arith
                                 : INTEGER)
        return integer
    is
        variable right_of_dp, left_of_dp, result : integer;
    begin
        right_of_dp := max(new_bin_pt, old_bin_pt);
        left_of_dp := max((new_width - new_bin_pt), (old_width - old_bin_pt));
        result := (old_width + 2) + (new_bin_pt - old_bin_pt);
        return result;
    end;
    function convert_type (inp : std_logic_vector; old_width, old_bin_pt,
                           old_arith, new_width, new_bin_pt, new_arith,
                           quantization, overflow : INTEGER)
        return std_logic_vector
    is
        constant fp_width : integer :=
            full_precision_num_width(quantization, overflow, old_width,
                                     old_bin_pt, old_arith, new_width,
                                     new_bin_pt, new_arith);
        constant fp_bin_pt : integer := old_bin_pt;
        constant fp_arith : integer := old_arith;
        variable full_precision_result : std_logic_vector(fp_width-1 downto 0);
        constant q_width : integer :=
            quantized_num_width(quantization, overflow, old_width, old_bin_pt,
                                old_arith, new_width, new_bin_pt, new_arith);
        constant q_bin_pt : integer := new_bin_pt;
        constant q_arith : integer := old_arith;
        variable quantized_result : std_logic_vector(q_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
    begin
        result := (others => '0');
        full_precision_result := cast(inp, old_bin_pt, fp_width, fp_bin_pt,
                                      fp_arith);
        if (quantization = xlRound) then
            quantized_result := round_towards_inf(full_precision_result,
                                                  fp_width, fp_bin_pt,
                                                  fp_arith, q_width, q_bin_pt,
                                                  q_arith);
        elsif (quantization = xlRoundBanker) then
            quantized_result := round_towards_even(full_precision_result,
                                                  fp_width, fp_bin_pt,
                                                  fp_arith, q_width, q_bin_pt,
                                                  q_arith);
        else
            quantized_result := trunc(full_precision_result, fp_width, fp_bin_pt,
                                      fp_arith, q_width, q_bin_pt, q_arith);
        end if;
        if (overflow = xlSaturate) then
            result := saturation_arith(quantized_result, q_width, q_bin_pt,
                                       q_arith, new_width, new_bin_pt, new_arith);
        else
             result := wrap_arith(quantized_result, q_width, q_bin_pt, q_arith,
                                  new_width, new_bin_pt, new_arith);
        end if;
        return result;
    end;
    function cast (inp : std_logic_vector; old_bin_pt, new_width,
                   new_bin_pt, new_arith : INTEGER)
        return std_logic_vector
    is
        constant old_width : integer := inp'length;
        constant left_of_dp : integer := (new_width - new_bin_pt)
                                         - (old_width - old_bin_pt);
        constant right_of_dp : integer := (new_bin_pt - old_bin_pt);
        variable vec : std_logic_vector(old_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
        variable j   : integer;
    begin
        vec := inp;
        for i in new_width-1 downto 0 loop
            j := i - right_of_dp;
            if ( j > old_width-1) then
                if (new_arith = xlUnsigned) then
                    result(i) := '0';
                else
                    result(i) := vec(old_width-1);
                end if;
            elsif ( j >= 0) then
                result(i) := vec(j);
            else
                result(i) := '0';
            end if;
        end loop;
        return result;
    end;
    function shift_division_result(quotient, fraction: std_logic_vector;
                                   fraction_width, shift_value, shift_dir: INTEGER)
        return std_logic_vector
    is
        constant q_width : integer := quotient'length;
        constant f_width : integer := fraction'length;
        constant vec_MSB : integer := q_width+f_width-1;
        constant result_MSB : integer := q_width+fraction_width-1;
        constant result_LSB : integer := vec_MSB-result_MSB;
        variable vec : std_logic_vector(vec_MSB downto 0);
        variable result : std_logic_vector(result_MSB downto 0);
    begin
        vec := ( quotient & fraction );
        if shift_dir = 1 then
            for i in vec_MSB downto 0 loop
                if (i < shift_value) then
                     vec(i) := '0';
                else
                    vec(i) := vec(i-shift_value);
                end if;
            end loop;
        else
            for i in 0 to vec_MSB loop
                if (i > vec_MSB-shift_value) then
                    vec(i) := vec(vec_MSB);
                else
                    vec(i) := vec(i+shift_value);
                end if;
            end loop;
        end if;
        result := vec(vec_MSB downto result_LSB);
        return result;
    end;
    function shift_op (inp: std_logic_vector;
                       result_width, shift_value, shift_dir: INTEGER)
        return std_logic_vector
    is
        constant inp_width : integer := inp'length;
        constant vec_MSB : integer := inp_width-1;
        constant result_MSB : integer := result_width-1;
        constant result_LSB : integer := vec_MSB-result_MSB;
        variable vec : std_logic_vector(vec_MSB downto 0);
        variable result : std_logic_vector(result_MSB downto 0);
    begin
        vec := inp;
        if shift_dir = 1 then
            for i in vec_MSB downto 0 loop
                if (i < shift_value) then
                     vec(i) := '0';
                else
                    vec(i) := vec(i-shift_value);
                end if;
            end loop;
        else
            for i in 0 to vec_MSB loop
                if (i > vec_MSB-shift_value) then
                    vec(i) := vec(vec_MSB);
                else
                    vec(i) := vec(i+shift_value);
                end if;
            end loop;
        end if;
        result := vec(vec_MSB downto result_LSB);
        return result;
    end;
    function vec_slice (inp : std_logic_vector; upper, lower : INTEGER)
      return std_logic_vector
    is
    begin
        return inp(upper downto lower);
    end;
    function s2u_slice (inp : signed; upper, lower : INTEGER)
      return unsigned
    is
    begin
        return unsigned(vec_slice(std_logic_vector(inp), upper, lower));
    end;
    function u2u_slice (inp : unsigned; upper, lower : INTEGER)
      return unsigned
    is
    begin
        return unsigned(vec_slice(std_logic_vector(inp), upper, lower));
    end;
    function s2s_cast (inp : signed; old_bin_pt, new_width, new_bin_pt : INTEGER)
        return signed
    is
    begin
        return signed(cast(std_logic_vector(inp), old_bin_pt, new_width, new_bin_pt, xlSigned));
    end;
    function s2u_cast (inp : signed; old_bin_pt, new_width,
                   new_bin_pt : INTEGER)
        return unsigned
    is
    begin
        return unsigned(cast(std_logic_vector(inp), old_bin_pt, new_width, new_bin_pt, xlSigned));
    end;
    function u2s_cast (inp : unsigned; old_bin_pt, new_width,
                   new_bin_pt : INTEGER)
        return signed
    is
    begin
        return signed(cast(std_logic_vector(inp), old_bin_pt, new_width, new_bin_pt, xlUnsigned));
    end;
    function u2u_cast (inp : unsigned; old_bin_pt, new_width,
                   new_bin_pt : INTEGER)
        return unsigned
    is
    begin
        return unsigned(cast(std_logic_vector(inp), old_bin_pt, new_width, new_bin_pt, xlUnsigned));
    end;
    function u2v_cast (inp : unsigned; old_bin_pt, new_width,
                   new_bin_pt : INTEGER)
        return std_logic_vector
    is
    begin
        return cast(std_logic_vector(inp), old_bin_pt, new_width, new_bin_pt, xlUnsigned);
    end;
    function s2v_cast (inp : signed; old_bin_pt, new_width,
                   new_bin_pt : INTEGER)
        return std_logic_vector
    is
    begin
        return cast(std_logic_vector(inp), old_bin_pt, new_width, new_bin_pt, xlSigned);
    end;
    function boolean_to_signed (inp : boolean; width : integer)
        return signed
    is
        variable result : signed(width - 1 downto 0);
    begin
        result := (others => '0');
        if inp then
          result(0) := '1';
        else
          result(0) := '0';
        end if;
        return result;
    end;
    function boolean_to_unsigned (inp : boolean; width : integer)
        return unsigned
    is
        variable result : unsigned(width - 1 downto 0);
    begin
        result := (others => '0');
        if inp then
          result(0) := '1';
        else
          result(0) := '0';
        end if;
        return result;
    end;
    function boolean_to_vector (inp : boolean)
        return std_logic_vector
    is
        variable result : std_logic_vector(1 - 1 downto 0);
    begin
        result := (others => '0');
        if inp then
          result(0) := '1';
        else
          result(0) := '0';
        end if;
        return result;
    end;
    function std_logic_to_vector (inp : std_logic)
        return std_logic_vector
    is
        variable result : std_logic_vector(1 - 1 downto 0);
    begin
        result(0) := inp;
        return result;
    end;
    function trunc (inp : std_logic_vector; old_width, old_bin_pt, old_arith,
                                new_width, new_bin_pt, new_arith : INTEGER)
        return std_logic_vector
    is
        constant right_of_dp : integer := (old_bin_pt - new_bin_pt);
        variable vec : std_logic_vector(old_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
    begin
        vec := inp;
        if right_of_dp >= 0 then
            if new_arith = xlUnsigned then
                result := zero_ext(vec(old_width-1 downto right_of_dp), new_width);
            else
                result := sign_ext(vec(old_width-1 downto right_of_dp), new_width);
            end if;
        else
            if new_arith = xlUnsigned then
                result := zero_ext(pad_LSB(vec, old_width +
                                           abs(right_of_dp)), new_width);
            else
                result := sign_ext(pad_LSB(vec, old_width +
                                           abs(right_of_dp)), new_width);
            end if;
        end if;
        return result;
    end;
    function round_towards_inf (inp : std_logic_vector; old_width, old_bin_pt,
                                old_arith, new_width, new_bin_pt, new_arith
                                : INTEGER)
        return std_logic_vector
    is
        constant right_of_dp : integer := (old_bin_pt - new_bin_pt);
        constant expected_new_width : integer :=  old_width - right_of_dp  + 1;
        variable vec : std_logic_vector(old_width-1 downto 0);
        variable one_or_zero : std_logic_vector(new_width-1 downto 0);
        variable truncated_val : std_logic_vector(new_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
    begin
        vec := inp;
        if right_of_dp >= 0 then
            if new_arith = xlUnsigned then
                truncated_val := zero_ext(vec(old_width-1 downto right_of_dp),
                                          new_width);
            else
                truncated_val := sign_ext(vec(old_width-1 downto right_of_dp),
                                          new_width);
            end if;
        else
            if new_arith = xlUnsigned then
                truncated_val := zero_ext(pad_LSB(vec, old_width +
                                                  abs(right_of_dp)), new_width);
            else
                truncated_val := sign_ext(pad_LSB(vec, old_width +
                                                  abs(right_of_dp)), new_width);
            end if;
        end if;
        one_or_zero := (others => '0');
        if (new_arith = xlSigned) then
            if (vec(old_width-1) = '0') then
                one_or_zero(0) := '1';
            end if;
            if (right_of_dp >= 2) and (right_of_dp <= old_width) then
                if (all_zeros(vec(right_of_dp-2 downto 0)) = false) then
                    one_or_zero(0) := '1';
                end if;
            end if;
            if (right_of_dp >= 1) and (right_of_dp <= old_width) then
                if vec(right_of_dp-1) = '0' then
                    one_or_zero(0) := '0';
                end if;
            else
                one_or_zero(0) := '0';
            end if;
        else
            if (right_of_dp >= 1) and (right_of_dp <= old_width) then
                one_or_zero(0) :=  vec(right_of_dp-1);
            end if;
        end if;
        if new_arith = xlSigned then
            result := signed_to_std_logic_vector(std_logic_vector_to_signed(truncated_val) +
                                                 std_logic_vector_to_signed(one_or_zero));
        else
            result := unsigned_to_std_logic_vector(std_logic_vector_to_unsigned(truncated_val) +
                                                  std_logic_vector_to_unsigned(one_or_zero));
        end if;
        return result;
    end;
    function round_towards_even (inp : std_logic_vector; old_width, old_bin_pt,
                                old_arith, new_width, new_bin_pt, new_arith
                                : INTEGER)
        return std_logic_vector
    is
        constant right_of_dp : integer := (old_bin_pt - new_bin_pt);
        constant expected_new_width : integer :=  old_width - right_of_dp  + 1;
        variable vec : std_logic_vector(old_width-1 downto 0);
        variable one_or_zero : std_logic_vector(new_width-1 downto 0);
        variable truncated_val : std_logic_vector(new_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
    begin
        vec := inp;
        if right_of_dp >= 0 then
            if new_arith = xlUnsigned then
                truncated_val := zero_ext(vec(old_width-1 downto right_of_dp),
                                          new_width);
            else
                truncated_val := sign_ext(vec(old_width-1 downto right_of_dp),
                                          new_width);
            end if;
        else
            if new_arith = xlUnsigned then
                truncated_val := zero_ext(pad_LSB(vec, old_width +
                                                  abs(right_of_dp)), new_width);
            else
                truncated_val := sign_ext(pad_LSB(vec, old_width +
                                                  abs(right_of_dp)), new_width);
            end if;
        end if;
        one_or_zero := (others => '0');
        if (right_of_dp >= 1) and (right_of_dp <= old_width) then
            if (is_point_five(vec(right_of_dp-1 downto 0)) = false) then
                one_or_zero(0) :=  vec(right_of_dp-1);
            else
                one_or_zero(0) :=  vec(right_of_dp);
            end if;
        end if;
        if new_arith = xlSigned then
            result := signed_to_std_logic_vector(std_logic_vector_to_signed(truncated_val) +
                                                 std_logic_vector_to_signed(one_or_zero));
        else
            result := unsigned_to_std_logic_vector(std_logic_vector_to_unsigned(truncated_val) +
                                                  std_logic_vector_to_unsigned(one_or_zero));
        end if;
        return result;
    end;
    function saturation_arith(inp:  std_logic_vector;  old_width, old_bin_pt,
                              old_arith, new_width, new_bin_pt, new_arith
                              : INTEGER)
        return std_logic_vector
    is
        constant left_of_dp : integer := (old_width - old_bin_pt) -
                                         (new_width - new_bin_pt);
        variable vec : std_logic_vector(old_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
        variable overflow : boolean;
    begin
        vec := inp;
        overflow := true;
        result := (others => '0');
        if (new_width >= old_width) then
            overflow := false;
        end if;
        if ((old_arith = xlSigned and new_arith = xlSigned) and (old_width > new_width)) then
            if all_same(vec(old_width-1 downto new_width-1)) then
                overflow := false;
            end if;
        end if;
        if (old_arith = xlSigned and new_arith = xlUnsigned) then
            if (old_width > new_width) then
                if all_zeros(vec(old_width-1 downto new_width)) then
                    overflow := false;
                end if;
            else
                if (old_width = new_width) then
                    if (vec(new_width-1) = '0') then
                        overflow := false;
                    end if;
                end if;
            end if;
        end if;
        if (old_arith = xlUnsigned and new_arith = xlUnsigned) then
            if (old_width > new_width) then
                if all_zeros(vec(old_width-1 downto new_width)) then
                    overflow := false;
                end if;
            else
                if (old_width = new_width) then
                    overflow := false;
                end if;
            end if;
        end if;
        if ((old_arith = xlUnsigned and new_arith = xlSigned) and (old_width > new_width)) then
            if all_same(vec(old_width-1 downto new_width-1)) then
                overflow := false;
            end if;
        end if;
        if overflow then
            if new_arith = xlSigned then
                if vec(old_width-1) = '0' then
                    result := max_signed(new_width);
                else
                    result := min_signed(new_width);
                end if;
            else
                if ((old_arith = xlSigned) and vec(old_width-1) = '1') then
                    result := (others => '0');
                else
                    result := (others => '1');
                end if;
            end if;
        else
            if (old_arith = xlSigned) and (new_arith = xlUnsigned) then
                if (vec(old_width-1) = '1') then
                    vec := (others => '0');
                end if;
            end if;
            if new_width <= old_width then
                result := vec(new_width-1 downto 0);
            else
                if new_arith = xlUnsigned then
                    result := zero_ext(vec, new_width);
                else
                    result := sign_ext(vec, new_width);
                end if;
            end if;
        end if;
        return result;
    end;
   function wrap_arith(inp:  std_logic_vector;  old_width, old_bin_pt,
                       old_arith, new_width, new_bin_pt, new_arith : INTEGER)
        return std_logic_vector
    is
        variable result : std_logic_vector(new_width-1 downto 0);
        variable result_arith : integer;
    begin
        if (old_arith = xlSigned) and (new_arith = xlUnsigned) then
            result_arith := xlSigned;
        end if;
        result := cast(inp, old_bin_pt, new_width, new_bin_pt, result_arith);
        return result;
    end;
    function fractional_bits(a_bin_pt, b_bin_pt: INTEGER) return INTEGER is
    begin
        return max(a_bin_pt, b_bin_pt);
    end;
    function integer_bits(a_width, a_bin_pt, b_width, b_bin_pt: INTEGER)
        return INTEGER is
    begin
        return  max(a_width - a_bin_pt, b_width - b_bin_pt);
    end;
    function pad_LSB(inp : std_logic_vector; new_width: integer)
        return STD_LOGIC_VECTOR
    is
        constant orig_width : integer := inp'length;
        variable vec : std_logic_vector(orig_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
        variable pos : integer;
        constant pad_pos : integer := new_width - orig_width - 1;
    begin
        vec := inp;
        pos := new_width-1;
        if (new_width >= orig_width) then
            for i in orig_width-1 downto 0 loop
                result(pos) := vec(i);
                pos := pos - 1;
            end loop;
            if pad_pos >= 0 then
                for i in pad_pos downto 0 loop
                    result(i) := '0';
                end loop;
            end if;
        end if;
        return result;
    end;
    function sign_ext(inp : std_logic_vector; new_width : INTEGER)
        return std_logic_vector
    is
        constant old_width : integer := inp'length;
        variable vec : std_logic_vector(old_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
    begin
        vec := inp;
        if new_width >= old_width then
            result(old_width-1 downto 0) := vec;
            if new_width-1 >= old_width then
                for i in new_width-1 downto old_width loop
                    result(i) := vec(old_width-1);
                end loop;
            end if;
        else
            result(new_width-1 downto 0) := vec(new_width-1 downto 0);
        end if;
        return result;
    end;
    function zero_ext(inp : std_logic_vector; new_width : INTEGER)
        return std_logic_vector
    is
        constant old_width : integer := inp'length;
        variable vec : std_logic_vector(old_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
    begin
        vec := inp;
        if new_width >= old_width then
            result(old_width-1 downto 0) := vec;
            if new_width-1 >= old_width then
                for i in new_width-1 downto old_width loop
                    result(i) := '0';
                end loop;
            end if;
        else
            result(new_width-1 downto 0) := vec(new_width-1 downto 0);
        end if;
        return result;
    end;
    function zero_ext(inp : std_logic; new_width : INTEGER)
        return std_logic_vector
    is
        variable result : std_logic_vector(new_width-1 downto 0);
    begin
        result(0) := inp;
        for i in new_width-1 downto 1 loop
            result(i) := '0';
        end loop;
        return result;
    end;
    function extend_MSB(inp : std_logic_vector; new_width, arith : INTEGER)
        return std_logic_vector
    is
        constant orig_width : integer := inp'length;
        variable vec : std_logic_vector(orig_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
    begin
        vec := inp;
        if arith = xlUnsigned then
            result := zero_ext(vec, new_width);
        else
            result := sign_ext(vec, new_width);
        end if;
        return result;
    end;
    function pad_LSB(inp : std_logic_vector; new_width, arith: integer)
        return STD_LOGIC_VECTOR
    is
        constant orig_width : integer := inp'length;
        variable vec : std_logic_vector(orig_width-1 downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
        variable pos : integer;
    begin
        vec := inp;
        pos := new_width-1;
        if (arith = xlUnsigned) then
            result(pos) := '0';
            pos := pos - 1;
        else
            result(pos) := vec(orig_width-1);
            pos := pos - 1;
        end if;
        if (new_width >= orig_width) then
            for i in orig_width-1 downto 0 loop
                result(pos) := vec(i);
                pos := pos - 1;
            end loop;
            if pos >= 0 then
                for i in pos downto 0 loop
                    result(i) := '0';
                end loop;
            end if;
        end if;
        return result;
    end;
    function align_input(inp : std_logic_vector; old_width, delta, new_arith,
                         new_width: INTEGER)
        return std_logic_vector
    is
        variable vec : std_logic_vector(old_width-1 downto 0);
        variable padded_inp : std_logic_vector((old_width + delta)-1  downto 0);
        variable result : std_logic_vector(new_width-1 downto 0);
    begin
        vec := inp;
        if delta > 0 then
            padded_inp := pad_LSB(vec, old_width+delta);
            result := extend_MSB(padded_inp, new_width, new_arith);
        else
            result := extend_MSB(vec, new_width, new_arith);
        end if;
        return result;
    end;
    function max(L, R: INTEGER) return INTEGER is
    begin
        if L > R then
            return L;
        else
            return R;
        end if;
    end;
    function min(L, R: INTEGER) return INTEGER is
    begin
        if L < R then
            return L;
        else
            return R;
        end if;
    end;
    function "="(left,right: STRING) return boolean is
    begin
        if (left'length /= right'length) then
            return false;
        else
            test : for i in 1 to left'length loop
                if left(i) /= right(i) then
                    return false;
                end if;
            end loop test;
            return true;
        end if;
    end;
    -- synopsys translate_off
    function is_binary_string_invalid (inp : string)
        return boolean
    is
        variable vec : string(1 to inp'length);
        variable result : boolean;
    begin
        vec := inp;
        result := false;
        for i in 1 to vec'length loop
            if ( vec(i) = 'X' ) then
                result := true;
            end if;
        end loop;
        return result;
    end;
    function is_binary_string_undefined (inp : string)
        return boolean
    is
        variable vec : string(1 to inp'length);
        variable result : boolean;
    begin
        vec := inp;
        result := false;
        for i in 1 to vec'length loop
            if ( vec(i) = 'U' ) then
                result := true;
            end if;
        end loop;
        return result;
    end;
    function is_XorU(inp : std_logic_vector)
        return boolean
    is
        constant width : integer := inp'length;
        variable vec : std_logic_vector(width-1 downto 0);
        variable result : boolean;
    begin
        vec := inp;
        result := false;
        for i in 0 to width-1 loop
            if (vec(i) = 'U') or (vec(i) = 'X') then
                result := true;
            end if;
        end loop;
        return result;
    end;
    function to_real(inp : std_logic_vector; bin_pt : integer; arith : integer)
        return real
    is
        variable  vec : std_logic_vector(inp'length-1 downto 0);
        variable result, shift_val, undefined_real : real;
        variable neg_num : boolean;
    begin
        vec := inp;
        result := 0.0;
        neg_num := false;
        if vec(inp'length-1) = '1' then
            neg_num := true;
        end if;
        for i in 0 to inp'length-1 loop
            if  vec(i) = 'U' or vec(i) = 'X' then
                return undefined_real;
            end if;
            if arith = xlSigned then
                if neg_num then
                    if vec(i) = '0' then
                        result := result + 2.0**i;
                    end if;
                else
                    if vec(i) = '1' then
                        result := result + 2.0**i;
                    end if;
                end if;
            else
                if vec(i) = '1' then
                    result := result + 2.0**i;
                end if;
            end if;
        end loop;
        if arith = xlSigned then
            if neg_num then
                result := result + 1.0;
                result := result * (-1.0);
            end if;
        end if;
        shift_val := 2.0**(-1*bin_pt);
        result := result * shift_val;
        return result;
    end;
    function std_logic_to_real(inp : std_logic; bin_pt : integer; arith : integer)
        return real
    is
        variable result : real := 0.0;
    begin
        if inp = '1' then
            result := 1.0;
        end if;
        if arith = xlSigned then
            assert false
                report "It doesn't make sense to convert a 1 bit number to a signed real.";
        end if;
        return result;
    end;
    -- synopsys translate_on
    function integer_to_std_logic_vector (inp : integer;  width, arith : integer)
        return std_logic_vector
    is
        variable result : std_logic_vector(width-1 downto 0);
        variable unsigned_val : unsigned(width-1 downto 0);
        variable signed_val : signed(width-1 downto 0);
    begin
        if (arith = xlSigned) then
            signed_val := to_signed(inp, width);
            result := signed_to_std_logic_vector(signed_val);
        else
            unsigned_val := to_unsigned(inp, width);
            result := unsigned_to_std_logic_vector(unsigned_val);
        end if;
        return result;
    end;
    function std_logic_vector_to_integer (inp : std_logic_vector;  arith : integer)
        return integer
    is
        constant width : integer := inp'length;
        variable unsigned_val : unsigned(width-1 downto 0);
        variable signed_val : signed(width-1 downto 0);
        variable result : integer;
    begin
        if (arith = xlSigned) then
            signed_val := std_logic_vector_to_signed(inp);
            result := to_integer(signed_val);
        else
            unsigned_val := std_logic_vector_to_unsigned(inp);
            result := to_integer(unsigned_val);
        end if;
        return result;
    end;
    function std_logic_to_integer(constant inp : std_logic := '0')
        return integer
    is
    begin
        if inp = '1' then
            return 1;
        else
            return 0;
        end if;
    end;
    function makeZeroBinStr (width : integer) return STRING is
        variable result : string(1 to width+3);
    begin
        result(1) := '0';
        result(2) := 'b';
        for i in 3 to width+2 loop
            result(i) := '0';
        end loop;
        result(width+3) := '.';
        return result;
    end;
    -- synopsys translate_off
    function real_string_to_std_logic_vector (inp : string;  width, bin_pt, arith : integer)
        return std_logic_vector
    is
        variable result : std_logic_vector(width-1 downto 0);
    begin
        result := (others => '0');
        return result;
    end;
    function real_to_std_logic_vector (inp : real;  width, bin_pt, arith : integer)
        return std_logic_vector
    is
        variable real_val : real;
        variable int_val : integer;
        variable result : std_logic_vector(width-1 downto 0) := (others => '0');
        variable unsigned_val : unsigned(width-1 downto 0) := (others => '0');
        variable signed_val : signed(width-1 downto 0) := (others => '0');
    begin
        real_val := inp;
        int_val := integer(real_val * 2.0**(bin_pt));
        if (arith = xlSigned) then
            signed_val := to_signed(int_val, width);
            result := signed_to_std_logic_vector(signed_val);
        else
            unsigned_val := to_unsigned(int_val, width);
            result := unsigned_to_std_logic_vector(unsigned_val);
        end if;
        return result;
    end;
    -- synopsys translate_on
    function valid_bin_string (inp : string)
        return boolean
    is
        variable vec : string(1 to inp'length);
    begin
        vec := inp;
        if (vec(1) = '0' and vec(2) = 'b') then
            return true;
        else
            return false;
        end if;
    end;
    function hex_string_to_std_logic_vector(inp: string; width : integer)
        return std_logic_vector is
        constant strlen       : integer := inp'LENGTH;
        variable result       : std_logic_vector(width-1 downto 0);
        variable bitval       : std_logic_vector((strlen*4)-1 downto 0);
        variable posn         : integer;
        variable ch           : character;
        variable vec          : string(1 to strlen);
    begin
        vec := inp;
        result := (others => '0');
        posn := (strlen*4)-1;
        for i in 1 to strlen loop
            ch := vec(i);
            case ch is
                when '0' => bitval(posn downto posn-3) := "0000";
                when '1' => bitval(posn downto posn-3) := "0001";
                when '2' => bitval(posn downto posn-3) := "0010";
                when '3' => bitval(posn downto posn-3) := "0011";
                when '4' => bitval(posn downto posn-3) := "0100";
                when '5' => bitval(posn downto posn-3) := "0101";
                when '6' => bitval(posn downto posn-3) := "0110";
                when '7' => bitval(posn downto posn-3) := "0111";
                when '8' => bitval(posn downto posn-3) := "1000";
                when '9' => bitval(posn downto posn-3) := "1001";
                when 'A' | 'a' => bitval(posn downto posn-3) := "1010";
                when 'B' | 'b' => bitval(posn downto posn-3) := "1011";
                when 'C' | 'c' => bitval(posn downto posn-3) := "1100";
                when 'D' | 'd' => bitval(posn downto posn-3) := "1101";
                when 'E' | 'e' => bitval(posn downto posn-3) := "1110";
                when 'F' | 'f' => bitval(posn downto posn-3) := "1111";
                when others => bitval(posn downto posn-3) := "XXXX";
                               -- synopsys translate_off
                               ASSERT false
                                   REPORT "Invalid hex value" SEVERITY ERROR;
                               -- synopsys translate_on
            end case;
            posn := posn - 4;
        end loop;
        if (width <= strlen*4) then
            result :=  bitval(width-1 downto 0);
        else
            result((strlen*4)-1 downto 0) := bitval;
        end if;
        return result;
    end;
    function bin_string_to_std_logic_vector (inp : string)
        return std_logic_vector
    is
        variable pos : integer;
        variable vec : string(1 to inp'length);
        variable result : std_logic_vector(inp'length-1 downto 0);
    begin
        vec := inp;
        pos := inp'length-1;
        result := (others => '0');
        for i in 1 to vec'length loop
            -- synopsys translate_off
            if (pos < 0) and (vec(i) = '0' or vec(i) = '1' or vec(i) = 'X' or vec(i) = 'U')  then
                assert false
                    report "Input string is larger than output std_logic_vector. Truncating output.";
                return result;
            end if;
            -- synopsys translate_on
            if vec(i) = '0' then
                result(pos) := '0';
                pos := pos - 1;
            end if;
            if vec(i) = '1' then
                result(pos) := '1';
                pos := pos - 1;
            end if;
            -- synopsys translate_off
            if (vec(i) = 'X' or vec(i) = 'U') then
                result(pos) := 'U';
                pos := pos - 1;
            end if;
            -- synopsys translate_on
        end loop;
        return result;
    end;
    function bin_string_element_to_std_logic_vector (inp : string;  width, index : integer)
        return std_logic_vector
    is
        constant str_width : integer := width + 4;
        constant inp_len : integer := inp'length;
        constant num_elements : integer := (inp_len + 1)/str_width;
        constant reverse_index : integer := (num_elements-1) - index;
        variable left_pos : integer;
        variable right_pos : integer;
        variable vec : string(1 to inp'length);
        variable result : std_logic_vector(width-1 downto 0);
    begin
        vec := inp;
        result := (others => '0');
        if (reverse_index = 0) and (reverse_index < num_elements) and (inp_len-3 >= width) then
            left_pos := 1;
            right_pos := width + 3;
            result := bin_string_to_std_logic_vector(vec(left_pos to right_pos));
        end if;
        if (reverse_index > 0) and (reverse_index < num_elements) and (inp_len-3 >= width) then
            left_pos := (reverse_index * str_width) + 1;
            right_pos := left_pos + width + 2;
            result := bin_string_to_std_logic_vector(vec(left_pos to right_pos));
        end if;
        return result;
    end;
   -- synopsys translate_off
    function std_logic_vector_to_bin_string(inp : std_logic_vector)
        return string
    is
        variable vec : std_logic_vector(1 to inp'length);
        variable result : string(vec'range);
    begin
        vec := inp;
        for i in vec'range loop
            result(i) := to_char(vec(i));
        end loop;
        return result;
    end;
    function std_logic_to_bin_string(inp : std_logic)
        return string
    is
        variable result : string(1 to 3);
    begin
        result(1) := '0';
        result(2) := 'b';
        result(3) := to_char(inp);
        return result;
    end;
    function std_logic_vector_to_bin_string_w_point(inp : std_logic_vector; bin_pt : integer)
        return string
    is
        variable width : integer := inp'length;
        variable vec : std_logic_vector(width-1 downto 0);
        variable str_pos : integer;
        variable result : string(1 to width+3);
    begin
        vec := inp;
        str_pos := 1;
        result(str_pos) := '0';
        str_pos := 2;
        result(str_pos) := 'b';
        str_pos := 3;
        for i in width-1 downto 0  loop
            if (((width+3) - bin_pt) = str_pos) then
                result(str_pos) := '.';
                str_pos := str_pos + 1;
            end if;
            result(str_pos) := to_char(vec(i));
            str_pos := str_pos + 1;
        end loop;
        if (bin_pt = 0) then
            result(str_pos) := '.';
        end if;
        return result;
    end;
    function real_to_bin_string(inp : real;  width, bin_pt, arith : integer)
        return string
    is
        variable result : string(1 to width);
        variable vec : std_logic_vector(width-1 downto 0);
    begin
        vec := real_to_std_logic_vector(inp, width, bin_pt, arith);
        result := std_logic_vector_to_bin_string(vec);
        return result;
    end;
    function real_to_string (inp : real) return string
    is
        variable result : string(1 to display_precision) := (others => ' ');
    begin
        result(real'image(inp)'range) := real'image(inp);
        return result;
    end;
    -- synopsys translate_on
end conv_pkg;

-------------------------------------------------------------------
-- System Generator version 14.2 VHDL source file.
--
-- Copyright(C) 2012 by Xilinx, Inc.  All rights reserved.  This
-- text/file contains proprietary, confidential information of Xilinx,
-- Inc., is distributed under license from Xilinx, Inc., and may be used,
-- copied and/or disclosed only pursuant to the terms of a valid license
-- agreement with Xilinx, Inc.  Xilinx hereby grants you a license to use
-- this text/file solely for design, simulation, implementation and
-- creation of design files limited to Xilinx devices or technologies.
-- Use with non-Xilinx devices or technologies is expressly prohibited
-- and immediately terminates your license unless covered by a separate
-- agreement.
--
-- Xilinx is providing this design, code, or information "as is" solely
-- for use in developing programs and solutions for Xilinx devices.  By
-- providing this design, code, or information as one possible
-- implementation of this feature, application or standard, Xilinx is
-- making no representation that this implementation is free from any
-- claims of infringement.  You are responsible for obtaining any rights
-- you may require for your implementation.  Xilinx expressly disclaims
-- any warranty whatsoever with respect to the adequacy of the
-- implementation, including but not limited to warranties of
-- merchantability or fitness for a particular purpose.
--
-- Xilinx products are not intended for use in life support appliances,
-- devices, or systems.  Use in such applications is expressly prohibited.
--
-- Any modifications that are made to the source code are done at the user's
-- sole risk and will be unsupported.
--
-- This copyright and support notice must be retained as part of this
-- text at all times.  (c) Copyright 1995-2012 Xilinx, Inc.  All rights
-- reserved.
-------------------------------------------------------------------
-- synopsys translate_off
library unisim;
use unisim.vcomponents.all;
-- synopsys translate_on
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;
entity srl17e is
    generic (width : integer:=16;
             latency : integer :=8);
    port (clk   : in std_logic;
          ce    : in std_logic;
          d     : in std_logic_vector(width-1 downto 0);
          q     : out std_logic_vector(width-1 downto 0));
end srl17e;
architecture structural of srl17e is
    component SRL16E
        port (D   : in STD_ULOGIC;
              CE  : in STD_ULOGIC;
              CLK : in STD_ULOGIC;
              A0  : in STD_ULOGIC;
              A1  : in STD_ULOGIC;
              A2  : in STD_ULOGIC;
              A3  : in STD_ULOGIC;
              Q   : out STD_ULOGIC);
    end component;
    attribute syn_black_box of SRL16E : component is true;
    attribute fpga_dont_touch of SRL16E : component is "true";
    component FDE
        port(
            Q  :        out   STD_ULOGIC;
            D  :        in    STD_ULOGIC;
            C  :        in    STD_ULOGIC;
            CE :        in    STD_ULOGIC);
    end component;
    attribute syn_black_box of FDE : component is true;
    attribute fpga_dont_touch of FDE : component is "true";
    constant a : std_logic_vector(4 downto 0) :=
        integer_to_std_logic_vector(latency-2,5,xlSigned);
    signal d_delayed : std_logic_vector(width-1 downto 0);
    signal srl16_out : std_logic_vector(width-1 downto 0);
begin
    d_delayed <= d after 200 ps;
    reg_array : for i in 0 to width-1 generate
        srl16_used: if latency > 1 generate
            u1 : srl16e port map(clk => clk,
                                 d => d_delayed(i),
                                 q => srl16_out(i),
                                 ce => ce,
                                 a0 => a(0),
                                 a1 => a(1),
                                 a2 => a(2),
                                 a3 => a(3));
        end generate;
        srl16_not_used: if latency <= 1 generate
            srl16_out(i) <= d_delayed(i);
        end generate;
        fde_used: if latency /= 0  generate
            u2 : fde port map(c => clk,
                              d => srl16_out(i),
                              q => q(i),
                              ce => ce);
        end generate;
        fde_not_used: if latency = 0  generate
            q(i) <= srl16_out(i);
        end generate;
    end generate;
 end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;
entity synth_reg is
    generic (width           : integer := 8;
             latency         : integer := 1);
    port (i       : in std_logic_vector(width-1 downto 0);
          ce      : in std_logic;
          clr     : in std_logic;
          clk     : in std_logic;
          o       : out std_logic_vector(width-1 downto 0));
end synth_reg;
architecture structural of synth_reg is
    component srl17e
        generic (width : integer:=16;
                 latency : integer :=8);
        port (clk : in std_logic;
              ce  : in std_logic;
              d   : in std_logic_vector(width-1 downto 0);
              q   : out std_logic_vector(width-1 downto 0));
    end component;
    function calc_num_srl17es (latency : integer)
        return integer
    is
        variable remaining_latency : integer;
        variable result : integer;
    begin
        result := latency / 17;
        remaining_latency := latency - (result * 17);
        if (remaining_latency /= 0) then
            result := result + 1;
        end if;
        return result;
    end;
    constant complete_num_srl17es : integer := latency / 17;
    constant num_srl17es : integer := calc_num_srl17es(latency);
    constant remaining_latency : integer := latency - (complete_num_srl17es * 17);
    type register_array is array (num_srl17es downto 0) of
        std_logic_vector(width-1 downto 0);
    signal z : register_array;
begin
    z(0) <= i;
    complete_ones : if complete_num_srl17es > 0 generate
        srl17e_array: for i in 0 to complete_num_srl17es-1 generate
            delay_comp : srl17e
                generic map (width => width,
                             latency => 17)
                port map (clk => clk,
                          ce  => ce,
                          d       => z(i),
                          q       => z(i+1));
        end generate;
    end generate;
    partial_one : if remaining_latency > 0 generate
        last_srl17e : srl17e
            generic map (width => width,
                         latency => remaining_latency)
            port map (clk => clk,
                      ce  => ce,
                      d   => z(num_srl17es-1),
                      q   => z(num_srl17es));
    end generate;
    o <= z(num_srl17es);
end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;
entity synth_reg_reg is
    generic (width           : integer := 8;
             latency         : integer := 1);
    port (i       : in std_logic_vector(width-1 downto 0);
          ce      : in std_logic;
          clr     : in std_logic;
          clk     : in std_logic;
          o       : out std_logic_vector(width-1 downto 0));
end synth_reg_reg;
architecture behav of synth_reg_reg is
  type reg_array_type is array (latency-1 downto 0) of std_logic_vector(width -1 downto 0);
  signal reg_bank : reg_array_type := (others => (others => '0'));
  signal reg_bank_in : reg_array_type := (others => (others => '0'));
  attribute syn_allow_retiming : boolean;
  attribute syn_srlstyle : string;
  attribute syn_allow_retiming of reg_bank : signal is true;
  attribute syn_allow_retiming of reg_bank_in : signal is true;
  attribute syn_srlstyle of reg_bank : signal is "registers";
  attribute syn_srlstyle of reg_bank_in : signal is "registers";
begin
  latency_eq_0: if latency = 0 generate
    o <= i;
  end generate latency_eq_0;
  latency_gt_0: if latency >= 1 generate
    o <= reg_bank(latency-1);
    reg_bank_in(0) <= i;
    loop_gen: for idx in latency-2 downto 0 generate
      reg_bank_in(idx+1) <= reg_bank(idx);
    end generate loop_gen;
    sync_loop: for sync_idx in latency-1 downto 0 generate
      sync_proc: process (clk)
      begin
        if clk'event and clk = '1' then
          if clr = '1' then
            reg_bank_in <= (others => (others => '0'));
          elsif ce = '1'  then
            reg_bank(sync_idx) <= reg_bank_in(sync_idx);
          end if;
        end if;
      end process sync_proc;
    end generate sync_loop;
  end generate latency_gt_0;
end behav;

-------------------------------------------------------------------
-- System Generator version 14.2 VHDL source file.
--
-- Copyright(C) 2012 by Xilinx, Inc.  All rights reserved.  This
-- text/file contains proprietary, confidential information of Xilinx,
-- Inc., is distributed under license from Xilinx, Inc., and may be used,
-- copied and/or disclosed only pursuant to the terms of a valid license
-- agreement with Xilinx, Inc.  Xilinx hereby grants you a license to use
-- this text/file solely for design, simulation, implementation and
-- creation of design files limited to Xilinx devices or technologies.
-- Use with non-Xilinx devices or technologies is expressly prohibited
-- and immediately terminates your license unless covered by a separate
-- agreement.
--
-- Xilinx is providing this design, code, or information "as is" solely
-- for use in developing programs and solutions for Xilinx devices.  By
-- providing this design, code, or information as one possible
-- implementation of this feature, application or standard, Xilinx is
-- making no representation that this implementation is free from any
-- claims of infringement.  You are responsible for obtaining any rights
-- you may require for your implementation.  Xilinx expressly disclaims
-- any warranty whatsoever with respect to the adequacy of the
-- implementation, including but not limited to warranties of
-- merchantability or fitness for a particular purpose.
--
-- Xilinx products are not intended for use in life support appliances,
-- devices, or systems.  Use in such applications is expressly prohibited.
--
-- Any modifications that are made to the source code are done at the user's
-- sole risk and will be unsupported.
--
-- This copyright and support notice must be retained as part of this
-- text at all times.  (c) Copyright 1995-2012 Xilinx, Inc.  All rights
-- reserved.
-------------------------------------------------------------------
-- synopsys translate_off
library unisim;
use unisim.vcomponents.all;
-- synopsys translate_on
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;
entity single_reg_w_init is
  generic (
    width: integer := 8;
    init_index: integer := 0;
    init_value: bit_vector := b"0000"
  );
  port (
    i: in std_logic_vector(width - 1 downto 0);
    ce: in std_logic;
    clr: in std_logic;
    clk: in std_logic;
    o: out std_logic_vector(width - 1 downto 0)
  );
end single_reg_w_init;
architecture structural of single_reg_w_init is
  function build_init_const(width: integer;
                            init_index: integer;
                            init_value: bit_vector)
    return std_logic_vector
  is
    variable result: std_logic_vector(width - 1 downto 0);
  begin
    if init_index = 0 then
      result := (others => '0');
    elsif init_index = 1 then
      result := (others => '0');
      result(0) := '1';
    else
      result := to_stdlogicvector(init_value);
    end if;
    return result;
  end;
  component fdre
    port (
      q: out std_ulogic;
      d: in  std_ulogic;
      c: in  std_ulogic;
      ce: in  std_ulogic;
      r: in  std_ulogic
    );
  end component;
  attribute syn_black_box of fdre: component is true;
  attribute fpga_dont_touch of fdre: component is "true";
  component fdse
    port (
      q: out std_ulogic;
      d: in  std_ulogic;
      c: in  std_ulogic;
      ce: in  std_ulogic;
      s: in  std_ulogic
    );
  end component;
  attribute syn_black_box of fdse: component is true;
  attribute fpga_dont_touch of fdse: component is "true";
  constant init_const: std_logic_vector(width - 1 downto 0)
    := build_init_const(width, init_index, init_value);
begin
  fd_prim_array: for index in 0 to width - 1 generate
    bit_is_0: if (init_const(index) = '0') generate
      fdre_comp: fdre
        port map (
          c => clk,
          d => i(index),
          q => o(index),
          ce => ce,
          r => clr
        );
    end generate;
    bit_is_1: if (init_const(index) = '1') generate
      fdse_comp: fdse
        port map (
          c => clk,
          d => i(index),
          q => o(index),
          ce => ce,
          s => clr
        );
    end generate;
  end generate;
end architecture structural;
-- synopsys translate_off
library unisim;
use unisim.vcomponents.all;
-- synopsys translate_on
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;
entity synth_reg_w_init is
  generic (
    width: integer := 8;
    init_index: integer := 0;
    init_value: bit_vector := b"0000";
    latency: integer := 1
  );
  port (
    i: in std_logic_vector(width - 1 downto 0);
    ce: in std_logic;
    clr: in std_logic;
    clk: in std_logic;
    o: out std_logic_vector(width - 1 downto 0)
  );
end synth_reg_w_init;
architecture structural of synth_reg_w_init is
  component single_reg_w_init
    generic (
      width: integer := 8;
      init_index: integer := 0;
      init_value: bit_vector := b"0000"
    );
    port (
      i: in std_logic_vector(width - 1 downto 0);
      ce: in std_logic;
      clr: in std_logic;
      clk: in std_logic;
      o: out std_logic_vector(width - 1 downto 0)
    );
  end component;
  signal dly_i: std_logic_vector((latency + 1) * width - 1 downto 0);
  signal dly_clr: std_logic;
begin
  latency_eq_0: if (latency = 0) generate
    o <= i;
  end generate;
  latency_gt_0: if (latency >= 1) generate
    dly_i((latency + 1) * width - 1 downto latency * width) <= i
      after 200 ps;
    dly_clr <= clr after 200 ps;
    fd_array: for index in latency downto 1 generate
       reg_comp: single_reg_w_init
          generic map (
            width => width,
            init_index => init_index,
            init_value => init_value
          )
          port map (
            clk => clk,
            i => dly_i((index + 1) * width - 1 downto index * width),
            o => dly_i(index * width - 1 downto (index - 1) * width),
            ce => ce,
            clr => dly_clr
          );
    end generate;
    o <= dly_i(width - 1 downto 0);
  end generate;
end structural;

-------------------------------------------------------------------
-- System Generator version 14.2 VHDL source file.
--
-- Copyright(C) 2012 by Xilinx, Inc.  All rights reserved.  This
-- text/file contains proprietary, confidential information of Xilinx,
-- Inc., is distributed under license from Xilinx, Inc., and may be used,
-- copied and/or disclosed only pursuant to the terms of a valid license
-- agreement with Xilinx, Inc.  Xilinx hereby grants you a license to use
-- this text/file solely for design, simulation, implementation and
-- creation of design files limited to Xilinx devices or technologies.
-- Use with non-Xilinx devices or technologies is expressly prohibited
-- and immediately terminates your license unless covered by a separate
-- agreement.
--
-- Xilinx is providing this design, code, or information "as is" solely
-- for use in developing programs and solutions for Xilinx devices.  By
-- providing this design, code, or information as one possible
-- implementation of this feature, application or standard, Xilinx is
-- making no representation that this implementation is free from any
-- claims of infringement.  You are responsible for obtaining any rights
-- you may require for your implementation.  Xilinx expressly disclaims
-- any warranty whatsoever with respect to the adequacy of the
-- implementation, including but not limited to warranties of
-- merchantability or fitness for a particular purpose.
--
-- Xilinx products are not intended for use in life support appliances,
-- devices, or systems.  Use in such applications is expressly prohibited.
--
-- Any modifications that are made to the source code are done at the user's
-- sole risk and will be unsupported.
--
-- This copyright and support notice must be retained as part of this
-- text at all times.  (c) Copyright 1995-2012 Xilinx, Inc.  All rights
-- reserved.
-------------------------------------------------------------------
-- synopsys translate_off
library XilinxCoreLib;
-- synopsys translate_on
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use work.conv_pkg.all;
entity xladdsub_cichbcic_core is
  generic (
    core_name0: string := "";
    a_width: integer := 16;
    a_bin_pt: integer := 4;
    a_arith: integer := xlUnsigned;
    c_in_width: integer := 16;
    c_in_bin_pt: integer := 4;
    c_in_arith: integer := xlUnsigned;
    c_out_width: integer := 16;
    c_out_bin_pt: integer := 4;
    c_out_arith: integer := xlUnsigned;
    b_width: integer := 8;
    b_bin_pt: integer := 2;
    b_arith: integer := xlUnsigned;
    s_width: integer := 17;
    s_bin_pt: integer := 4;
    s_arith: integer := xlUnsigned;
    rst_width: integer := 1;
    rst_bin_pt: integer := 0;
    rst_arith: integer := xlUnsigned;
    en_width: integer := 1;
    en_bin_pt: integer := 0;
    en_arith: integer := xlUnsigned;
    full_s_width: integer := 17;
    full_s_arith: integer := xlUnsigned;
    mode: integer := xlAddMode;
    extra_registers: integer := 0;
    latency: integer := 0;
    quantization: integer := xlTruncate;
    overflow: integer := xlWrap;
    c_latency: integer := 0;
    c_output_width: integer := 17;
    c_has_c_in : integer := 0;
    c_has_c_out : integer := 0
  );
  port (
    a: in std_logic_vector(a_width - 1 downto 0);
    b: in std_logic_vector(b_width - 1 downto 0);
    c_in : in std_logic_vector (0 downto 0) := "0";
    ce: in std_logic;
    clr: in std_logic := '0';
    clk: in std_logic;
    rst: in std_logic_vector(rst_width - 1 downto 0) := "0";
    en: in std_logic_vector(en_width - 1 downto 0) := "1";
    c_out : out std_logic_vector (0 downto 0);
    s: out std_logic_vector(s_width - 1 downto 0)
  );
end xladdsub_cichbcic_core;
architecture behavior of xladdsub_cichbcic_core is
  component synth_reg
    generic (
      width: integer := 16;
      latency: integer := 5
    );
    port (
      i: in std_logic_vector(width - 1 downto 0);
      ce: in std_logic;
      clr: in std_logic;
      clk: in std_logic;
      o: out std_logic_vector(width - 1 downto 0)
    );
  end component;
  function format_input(inp: std_logic_vector; old_width, delta, new_arith,
                        new_width: integer)
    return std_logic_vector
  is
    variable vec: std_logic_vector(old_width-1 downto 0);
    variable padded_inp: std_logic_vector((old_width + delta)-1  downto 0);
    variable result: std_logic_vector(new_width-1 downto 0);
  begin
    vec := inp;
    if (delta > 0) then
      padded_inp := pad_LSB(vec, old_width+delta);
      result := extend_MSB(padded_inp, new_width, new_arith);
    else
      result := extend_MSB(vec, new_width, new_arith);
    end if;
    return result;
  end;
  constant full_s_bin_pt: integer := fractional_bits(a_bin_pt, b_bin_pt);
  constant full_a_width: integer := full_s_width;
  constant full_b_width: integer := full_s_width;
  signal full_a: std_logic_vector(full_a_width - 1 downto 0);
  signal full_b: std_logic_vector(full_b_width - 1 downto 0);
  signal core_s: std_logic_vector(full_s_width - 1 downto 0);
  signal conv_s: std_logic_vector(s_width - 1 downto 0);
  signal temp_cout : std_logic;
  signal internal_clr: std_logic;
  signal internal_ce: std_logic;
  signal extra_reg_ce: std_logic;
  signal override: std_logic;
  signal logic1: std_logic_vector(0 downto 0);
  component addsb_11_0_a6179ec1a236388e
    port (
          a: in std_logic_vector(49 - 1 downto 0);
    clk: in std_logic:= '0';
    ce: in std_logic:= '0';
    s: out std_logic_vector(c_output_width - 1 downto 0);
    b: in std_logic_vector(49 - 1 downto 0)
    );
  end component;
  component addsb_11_0_5bfb73f1589643d3
    port (
          a: in std_logic_vector(52 - 1 downto 0);
    clk: in std_logic:= '0';
    ce: in std_logic:= '0';
    s: out std_logic_vector(c_output_width - 1 downto 0);
    b: in std_logic_vector(52 - 1 downto 0)
    );
  end component;
  component addsb_11_0_e35a3bf39f366fd8
    port (
          a: in std_logic_vector(50 - 1 downto 0);
    clk: in std_logic:= '0';
    ce: in std_logic:= '0';
    s: out std_logic_vector(c_output_width - 1 downto 0);
    b: in std_logic_vector(50 - 1 downto 0)
    );
  end component;
  component addsb_11_0_5de09ee679db1560
    port (
          a: in std_logic_vector(51 - 1 downto 0);
    clk: in std_logic:= '0';
    ce: in std_logic:= '0';
    s: out std_logic_vector(c_output_width - 1 downto 0);
    b: in std_logic_vector(51 - 1 downto 0)
    );
  end component;
begin
  internal_clr <= (clr or (rst(0))) and ce;
  internal_ce <= ce and en(0);
  logic1(0) <= '1';
  addsub_process: process (a, b, core_s)
  begin
    full_a <= format_input (a, a_width, b_bin_pt - a_bin_pt, a_arith,
                            full_a_width);
    full_b <= format_input (b, b_width, a_bin_pt - b_bin_pt, b_arith,
                            full_b_width);
    conv_s <= convert_type (core_s, full_s_width, full_s_bin_pt, full_s_arith,
                            s_width, s_bin_pt, s_arith, quantization, overflow);
  end process addsub_process;

  comp0: if ((core_name0 = "addsb_11_0_a6179ec1a236388e")) generate
    core_instance0: addsb_11_0_a6179ec1a236388e
      port map (
         a => full_a,
         clk => clk,
         ce => internal_ce,
         s => core_s,
         b => full_b
      );
  end generate;
  comp1: if ((core_name0 = "addsb_11_0_5bfb73f1589643d3")) generate
    core_instance1: addsb_11_0_5bfb73f1589643d3
      port map (
         a => full_a,
         clk => clk,
         ce => internal_ce,
         s => core_s,
         b => full_b
      );
  end generate;
  comp2: if ((core_name0 = "addsb_11_0_e35a3bf39f366fd8")) generate
    core_instance2: addsb_11_0_e35a3bf39f366fd8
      port map (
         a => full_a,
         clk => clk,
         ce => internal_ce,
         s => core_s,
         b => full_b
      );
  end generate;
  comp3: if ((core_name0 = "addsb_11_0_5de09ee679db1560")) generate
    core_instance3: addsb_11_0_5de09ee679db1560
      port map (
         a => full_a,
         clk => clk,
         ce => internal_ce,
         s => core_s,
         b => full_b
      );
  end generate;
  latency_test: if (extra_registers > 0) generate
      override_test: if (c_latency > 1) generate
       override_pipe: synth_reg
          generic map (
            width => 1,
            latency => c_latency
          )
          port map (
            i => logic1,
            ce => internal_ce,
            clr => internal_clr,
            clk => clk,
            o(0) => override);
       extra_reg_ce <= ce and en(0) and override;
      end generate override_test;
      no_override: if ((c_latency = 0) or (c_latency = 1)) generate
       extra_reg_ce <= ce and en(0);
      end generate no_override;
      extra_reg: synth_reg
        generic map (
          width => s_width,
          latency => extra_registers
        )
        port map (
          i => conv_s,
          ce => extra_reg_ce,
          clr => internal_clr,
          clk => clk,
          o => s
        );
      cout_test: if (c_has_c_out = 1) generate
      c_out_extra_reg: synth_reg
        generic map (
          width => 1,
          latency => extra_registers
        )
        port map (
          i(0) => temp_cout,
          ce => extra_reg_ce,
          clr => internal_clr,
          clk => clk,
          o => c_out
        );
      end generate cout_test;
  end generate;
  latency_s: if ((latency = 0) or (extra_registers = 0)) generate
    s <= conv_s;
  end generate latency_s;
  latency0: if (((latency = 0) or (extra_registers = 0)) and
                 (c_has_c_out = 1)) generate
    c_out(0) <= temp_cout;
  end generate latency0;
  tie_dangling_cout: if (c_has_c_out = 0) generate
    c_out <= "0";
  end generate tie_dangling_cout;
end architecture behavior;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity delay_d5831e814a is
  port (
    d : in std_logic_vector((48 - 1) downto 0);
    q : out std_logic_vector((48 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end delay_d5831e814a;


architecture behavior of delay_d5831e814a is
  signal d_1_22: std_logic_vector((48 - 1) downto 0);
  type array_type_op_mem_20_24 is array (0 to (4 - 1)) of std_logic_vector((48 - 1) downto 0);
  signal op_mem_20_24: array_type_op_mem_20_24 := (
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000");
  signal op_mem_20_24_front_din: std_logic_vector((48 - 1) downto 0);
  signal op_mem_20_24_back: std_logic_vector((48 - 1) downto 0);
  signal op_mem_20_24_push_front_pop_back_en: std_logic;
begin
  d_1_22 <= d;
  op_mem_20_24_back <= op_mem_20_24(3);
  proc_op_mem_20_24: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_20_24_push_front_pop_back_en = '1')) then
        for i in 3 downto 1 loop 
          op_mem_20_24(i) <= op_mem_20_24(i-1);
        end loop;
        op_mem_20_24(0) <= op_mem_20_24_front_din;
      end if;
    end if;
  end process proc_op_mem_20_24;
  op_mem_20_24_front_din <= d_1_22;
  op_mem_20_24_push_front_pop_back_en <= '1';
  q <= op_mem_20_24_back;
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity constant_9f5572ba51 is
  port (
    op : out std_logic_vector((16 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end constant_9f5572ba51;


architecture behavior of constant_9f5572ba51 is
begin
  op <= "0000000000000000";
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity counter_4d9f59591b is
  port (
    rst : in std_logic_vector((1 - 1) downto 0);
    op : out std_logic_vector((1 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end counter_4d9f59591b;


architecture behavior of counter_4d9f59591b is
  signal rst_1_40: boolean;
  signal count_reg_20_23: unsigned((1 - 1) downto 0) := "1";
  signal count_reg_20_23_rst: std_logic;
  signal rel_34_8: boolean;
  signal rst_limit_join_34_5: boolean;
  signal bool_44_4: boolean;
  signal count_reg_join_44_1: signed((3 - 1) downto 0);
  signal count_reg_join_44_1_rst: std_logic;
  signal rst_limit_join_44_1: boolean;
begin
  rst_1_40 <= ((rst) = "1");
  proc_count_reg_20_23: process (clk)
  is
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (count_reg_20_23_rst = '1')) then
        count_reg_20_23 <= "1";
      elsif (ce = '1') then 
        count_reg_20_23 <= count_reg_20_23 - std_logic_vector_to_unsigned("1");
      end if;
    end if;
  end process proc_count_reg_20_23;
  rel_34_8 <= count_reg_20_23 = std_logic_vector_to_unsigned("0");
  proc_if_34_5: process (rel_34_8)
  is
  begin
    if rel_34_8 then
      rst_limit_join_34_5 <= true;
    else 
      rst_limit_join_34_5 <= false;
    end if;
  end process proc_if_34_5;
  bool_44_4 <= rst_1_40 or rst_limit_join_34_5;
  proc_if_44_1: process (bool_44_4, count_reg_20_23, rst_limit_join_34_5)
  is
  begin
    if bool_44_4 then
      count_reg_join_44_1_rst <= '1';
    else 
      count_reg_join_44_1_rst <= '0';
    end if;
    if bool_44_4 then
      rst_limit_join_44_1 <= false;
    else 
      rst_limit_join_44_1 <= rst_limit_join_34_5;
    end if;
  end process proc_if_44_1;
  count_reg_20_23_rst <= count_reg_join_44_1_rst;
  op <= unsigned_to_std_logic_vector(count_reg_20_23);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity logical_aacf6e1b0e is
  port (
    d0 : in std_logic_vector((1 - 1) downto 0);
    d1 : in std_logic_vector((1 - 1) downto 0);
    y : out std_logic_vector((1 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end logical_aacf6e1b0e;


architecture behavior of logical_aacf6e1b0e is
  signal d0_1_24: std_logic;
  signal d1_1_27: std_logic;
  signal fully_2_1_bit: std_logic;
begin
  d0_1_24 <= d0(0);
  d1_1_27 <= d1(0);
  fully_2_1_bit <= d0_1_24 or d1_1_27;
  y <= std_logic_to_vector(fully_2_1_bit);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity relational_af078c6141 is
  port (
    a : in std_logic_vector((1 - 1) downto 0);
    b : in std_logic_vector((16 - 1) downto 0);
    op : out std_logic_vector((1 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end relational_af078c6141;


architecture behavior of relational_af078c6141 is
  signal a_1_31: unsigned((1 - 1) downto 0);
  signal b_1_34: signed((16 - 1) downto 0);
  signal cast_12_12: signed((16 - 1) downto 0);
  signal result_12_3_rel: boolean;
begin
  a_1_31 <= std_logic_vector_to_unsigned(a);
  b_1_34 <= std_logic_vector_to_signed(b);
  cast_12_12 <= u2s_cast(a_1_31, 0, 16, 14);
  result_12_3_rel <= cast_12_12 = b_1_34;
  op <= boolean_to_vector(result_12_3_rel);
end behavior;


-------------------------------------------------------------------
-- System Generator version 14.2 VHDL source file.
--
-- Copyright(C) 2012 by Xilinx, Inc.  All rights reserved.  This
-- text/file contains proprietary, confidential information of Xilinx,
-- Inc., is distributed under license from Xilinx, Inc., and may be used,
-- copied and/or disclosed only pursuant to the terms of a valid license
-- agreement with Xilinx, Inc.  Xilinx hereby grants you a license to use
-- this text/file solely for design, simulation, implementation and
-- creation of design files limited to Xilinx devices or technologies.
-- Use with non-Xilinx devices or technologies is expressly prohibited
-- and immediately terminates your license unless covered by a separate
-- agreement.
--
-- Xilinx is providing this design, code, or information "as is" solely
-- for use in developing programs and solutions for Xilinx devices.  By
-- providing this design, code, or information as one possible
-- implementation of this feature, application or standard, Xilinx is
-- making no representation that this implementation is free from any
-- claims of infringement.  You are responsible for obtaining any rights
-- you may require for your implementation.  Xilinx expressly disclaims
-- any warranty whatsoever with respect to the adequacy of the
-- implementation, including but not limited to warranties of
-- merchantability or fitness for a particular purpose.
--
-- Xilinx products are not intended for use in life support appliances,
-- devices, or systems.  Use in such applications is expressly prohibited.
--
-- Any modifications that are made to the source code are done at the user's
-- sole risk and will be unsupported.
--
-- This copyright and support notice must be retained as part of this
-- text at all times.  (c) Copyright 1995-2012 Xilinx, Inc.  All rights
-- reserved.
-------------------------------------------------------------------
-- synopsys translate_off
library XilinxCoreLib;
-- synopsys translate_on
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use work.conv_pkg.all;
entity xlmult_cichbcic_core is
  generic (
    core_name0: string := "";
    a_width: integer := 4;
    a_bin_pt: integer := 2;
    a_arith: integer := xlSigned;
    b_width: integer := 4;
    b_bin_pt: integer := 1;
    b_arith: integer := xlSigned;
    p_width: integer := 8;
    p_bin_pt: integer := 2;
    p_arith: integer := xlSigned;
    rst_width: integer := 1;
    rst_bin_pt: integer := 0;
    rst_arith: integer := xlUnsigned;
    en_width: integer := 1;
    en_bin_pt: integer := 0;
    en_arith: integer := xlUnsigned;
    quantization: integer := xlTruncate;
    overflow: integer := xlWrap;
    extra_registers: integer := 0;
    c_a_width: integer := 7;
    c_b_width: integer := 7;
    c_type: integer := 0;
    c_a_type: integer := 0;
    c_b_type: integer := 0;
    c_pipelined: integer := 1;
    c_baat: integer := 4;
    multsign: integer := xlSigned;
    c_output_width: integer := 16
  );
  port (
    a: in std_logic_vector(a_width - 1 downto 0);
    b: in std_logic_vector(b_width - 1 downto 0);
    ce: in std_logic;
    clr: in std_logic;
    clk: in std_logic;
    core_ce: in std_logic := '0';
    core_clr: in std_logic := '0';
    core_clk: in std_logic := '0';
    rst: in std_logic_vector(rst_width - 1 downto 0);
    en: in std_logic_vector(en_width - 1 downto 0);
    p: out std_logic_vector(p_width - 1 downto 0)
  );
end xlmult_cichbcic_core;
architecture behavior of xlmult_cichbcic_core is
  component synth_reg
    generic (
      width: integer := 16;
      latency: integer := 5
    );
    port (
      i: in std_logic_vector(width - 1 downto 0);
      ce: in std_logic;
      clr: in std_logic;
      clk: in std_logic;
      o: out std_logic_vector(width - 1 downto 0)
    );
  end component;
  component mult_11_2_893416810381d560
    port (
      b: in std_logic_vector(c_b_width - 1 downto 0);
      p: out std_logic_vector(c_output_width - 1 downto 0);
      clk: in std_logic;
      ce: in std_logic;
      sclr: in std_logic;
      a: in std_logic_vector(c_a_width - 1 downto 0)
    );
  end component;

  attribute syn_black_box of mult_11_2_893416810381d560:
    component is true;
  attribute fpga_dont_touch of mult_11_2_893416810381d560:
    component is "true";
  attribute box_type of mult_11_2_893416810381d560:
    component  is "black_box";
  signal tmp_a: std_logic_vector(c_a_width - 1 downto 0);
  signal conv_a: std_logic_vector(c_a_width - 1 downto 0);
  signal tmp_b: std_logic_vector(c_b_width - 1 downto 0);
  signal conv_b: std_logic_vector(c_b_width - 1 downto 0);
  signal tmp_p: std_logic_vector(c_output_width - 1 downto 0);
  signal conv_p: std_logic_vector(p_width - 1 downto 0);
  -- synopsys translate_off
  signal real_a, real_b, real_p: real;
  -- synopsys translate_on
  signal rfd: std_logic;
  signal rdy: std_logic;
  signal nd: std_logic;
  signal internal_ce: std_logic;
  signal internal_clr: std_logic;
  signal internal_core_ce: std_logic;
begin
-- synopsys translate_off
-- synopsys translate_on
  internal_ce <= ce and en(0);
  internal_core_ce <= core_ce and en(0);
  internal_clr <= (clr or rst(0)) and ce;
  nd <= internal_ce;
  input_process:  process (a,b)
  begin
    tmp_a <= zero_ext(a, c_a_width);
    tmp_b <= zero_ext(b, c_b_width);
  end process;
  output_process: process (tmp_p)
  begin
    conv_p <= convert_type(tmp_p, c_output_width, a_bin_pt+b_bin_pt, multsign,
                           p_width, p_bin_pt, p_arith, quantization, overflow);
  end process;
  comp0: if ((core_name0 = "mult_11_2_893416810381d560")) generate
    core_instance0: mult_11_2_893416810381d560
      port map (
        a => tmp_a,
        clk => clk,
        ce => internal_ce,
        sclr => internal_clr,
        p => tmp_p,
        b => tmp_b
      );
  end generate;
  latency_gt_0: if (extra_registers > 0) generate
    reg: synth_reg
      generic map (
        width => p_width,
        latency => extra_registers
      )
      port map (
        i => conv_p,
        ce => internal_ce,
        clr => internal_clr,
        clk => clk,
        o => p
      );
  end generate;
  latency_eq_0: if (extra_registers = 0) generate
    p <= conv_p;
  end generate;
end architecture behavior;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity constant_37b2f0b7ea is
  port (
    op : out std_logic_vector((16 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end constant_37b2f0b7ea;


architecture behavior of constant_37b2f0b7ea is
begin
  op <= "1111111001001110";
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity constant_d4111c362e is
  port (
    op : out std_logic_vector((16 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end constant_d4111c362e;


architecture behavior of constant_d4111c362e is
begin
  op <= "1111101000000110";
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity constant_b0dd5d0cf3 is
  port (
    op : out std_logic_vector((16 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end constant_b0dd5d0cf3;


architecture behavior of constant_b0dd5d0cf3 is
begin
  op <= "0000001011010010";
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity constant_32e8526bad is
  port (
    op : out std_logic_vector((16 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end constant_32e8526bad;


architecture behavior of constant_32e8526bad is
begin
  op <= "0001010000010111";
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity constant_6c8e1bed76 is
  port (
    op : out std_logic_vector((16 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end constant_6c8e1bed76;


architecture behavior of constant_6c8e1bed76 is
begin
  op <= "0010000000000000";
end behavior;


-------------------------------------------------------------------
-- System Generator version 14.2 VHDL source file.
--
-- Copyright(C) 2012 by Xilinx, Inc.  All rights reserved.  This
-- text/file contains proprietary, confidential information of Xilinx,
-- Inc., is distributed under license from Xilinx, Inc., and may be used,
-- copied and/or disclosed only pursuant to the terms of a valid license
-- agreement with Xilinx, Inc.  Xilinx hereby grants you a license to use
-- this text/file solely for design, simulation, implementation and
-- creation of design files limited to Xilinx devices or technologies.
-- Use with non-Xilinx devices or technologies is expressly prohibited
-- and immediately terminates your license unless covered by a separate
-- agreement.
--
-- Xilinx is providing this design, code, or information "as is" solely
-- for use in developing programs and solutions for Xilinx devices.  By
-- providing this design, code, or information as one possible
-- implementation of this feature, application or standard, Xilinx is
-- making no representation that this implementation is free from any
-- claims of infringement.  You are responsible for obtaining any rights
-- you may require for your implementation.  Xilinx expressly disclaims
-- any warranty whatsoever with respect to the adequacy of the
-- implementation, including but not limited to warranties of
-- merchantability or fitness for a particular purpose.
--
-- Xilinx products are not intended for use in life support appliances,
-- devices, or systems.  Use in such applications is expressly prohibited.
--
-- Any modifications that are made to the source code are done at the user's
-- sole risk and will be unsupported.
--
-- This copyright and support notice must be retained as part of this
-- text at all times.  (c) Copyright 1995-2012 Xilinx, Inc.  All rights
-- reserved.
-------------------------------------------------------------------
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;
entity xldelay is
   generic(width        : integer := -1;
           latency      : integer := -1;
           reg_retiming : integer :=  0;
           reset        : integer :=  0);
   port(d       : in std_logic_vector (width-1 downto 0);
        ce      : in std_logic;
        clk     : in std_logic;
        en      : in std_logic;
        rst     : in std_logic;
        q       : out std_logic_vector (width-1 downto 0));
end xldelay;
architecture behavior of xldelay is
   component synth_reg
      generic (width       : integer;
               latency     : integer);
      port (i       : in std_logic_vector(width-1 downto 0);
            ce      : in std_logic;
            clr     : in std_logic;
            clk     : in std_logic;
            o       : out std_logic_vector(width-1 downto 0));
   end component;
   component synth_reg_reg
      generic (width       : integer;
               latency     : integer);
      port (i       : in std_logic_vector(width-1 downto 0);
            ce      : in std_logic;
            clr     : in std_logic;
            clk     : in std_logic;
            o       : out std_logic_vector(width-1 downto 0));
   end component;
   signal internal_ce  : std_logic;
begin
   internal_ce  <= ce and en;
   srl_delay: if ((reg_retiming = 0) and (reset = 0)) or (latency < 1) generate
     synth_reg_srl_inst : synth_reg
       generic map (
         width   => width,
         latency => latency)
       port map (
         i   => d,
         ce  => internal_ce,
         clr => '0',
         clk => clk,
         o   => q);
   end generate srl_delay;
   reg_delay: if ((reg_retiming = 1) or (reset = 1)) and (latency >= 1) generate
     synth_reg_reg_inst : synth_reg_reg
       generic map (
         width   => width,
         latency => latency)
       port map (
         i   => d,
         ce  => internal_ce,
         clr => rst,
         clk => clk,
         o   => q);
   end generate reg_delay;
end architecture behavior;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_6eb5f4740f is
  port (
    a : in std_logic_vector((24 - 1) downto 0);
    b : in std_logic_vector((24 - 1) downto 0);
    s : out std_logic_vector((25 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_6eb5f4740f;


architecture behavior of addsub_6eb5f4740f is
  signal a_17_32: signed((24 - 1) downto 0);
  signal b_17_35: signed((24 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((25 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "0000000000000000000000000");
  signal op_mem_91_20_front_din: signed((25 - 1) downto 0);
  signal op_mem_91_20_back: signed((25 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((25 - 1) downto 0);
  signal cast_69_22: signed((25 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((25 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 25, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 25, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_b4fd6cc060 is
  port (
    a : in std_logic_vector((25 - 1) downto 0);
    b : in std_logic_vector((25 - 1) downto 0);
    s : out std_logic_vector((26 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_b4fd6cc060;


architecture behavior of addsub_b4fd6cc060 is
  signal a_17_32: signed((25 - 1) downto 0);
  signal b_17_35: signed((25 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((26 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "00000000000000000000000000");
  signal op_mem_91_20_front_din: signed((26 - 1) downto 0);
  signal op_mem_91_20_back: signed((26 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((26 - 1) downto 0);
  signal cast_69_22: signed((26 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((26 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 26, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 26, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_f6988cbd01 is
  port (
    a : in std_logic_vector((26 - 1) downto 0);
    b : in std_logic_vector((26 - 1) downto 0);
    s : out std_logic_vector((27 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_f6988cbd01;


architecture behavior of addsub_f6988cbd01 is
  signal a_17_32: signed((26 - 1) downto 0);
  signal b_17_35: signed((26 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((27 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((27 - 1) downto 0);
  signal op_mem_91_20_back: signed((27 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((27 - 1) downto 0);
  signal cast_69_22: signed((27 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((27 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 27, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 27, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_7d1974bd4f is
  port (
    a : in std_logic_vector((27 - 1) downto 0);
    b : in std_logic_vector((27 - 1) downto 0);
    s : out std_logic_vector((28 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_7d1974bd4f;


architecture behavior of addsub_7d1974bd4f is
  signal a_17_32: signed((27 - 1) downto 0);
  signal b_17_35: signed((27 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((28 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "0000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((28 - 1) downto 0);
  signal op_mem_91_20_back: signed((28 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((28 - 1) downto 0);
  signal cast_69_22: signed((28 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((28 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 28, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 28, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_a6d6abc1fb is
  port (
    a : in std_logic_vector((28 - 1) downto 0);
    b : in std_logic_vector((28 - 1) downto 0);
    s : out std_logic_vector((29 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_a6d6abc1fb;


architecture behavior of addsub_a6d6abc1fb is
  signal a_17_32: signed((28 - 1) downto 0);
  signal b_17_35: signed((28 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((29 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "00000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((29 - 1) downto 0);
  signal op_mem_91_20_back: signed((29 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((29 - 1) downto 0);
  signal cast_69_22: signed((29 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((29 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 29, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 29, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_2de4d085a7 is
  port (
    a : in std_logic_vector((29 - 1) downto 0);
    b : in std_logic_vector((29 - 1) downto 0);
    s : out std_logic_vector((30 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_2de4d085a7;


architecture behavior of addsub_2de4d085a7 is
  signal a_17_32: signed((29 - 1) downto 0);
  signal b_17_35: signed((29 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((30 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((30 - 1) downto 0);
  signal op_mem_91_20_back: signed((30 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((30 - 1) downto 0);
  signal cast_69_22: signed((30 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((30 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 30, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 30, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_78fc83082e is
  port (
    a : in std_logic_vector((30 - 1) downto 0);
    b : in std_logic_vector((30 - 1) downto 0);
    s : out std_logic_vector((31 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_78fc83082e;


architecture behavior of addsub_78fc83082e is
  signal a_17_32: signed((30 - 1) downto 0);
  signal b_17_35: signed((30 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((31 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "0000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((31 - 1) downto 0);
  signal op_mem_91_20_back: signed((31 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((31 - 1) downto 0);
  signal cast_69_22: signed((31 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((31 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 31, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 31, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_2a7fe08e67 is
  port (
    a : in std_logic_vector((31 - 1) downto 0);
    b : in std_logic_vector((31 - 1) downto 0);
    s : out std_logic_vector((32 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_2a7fe08e67;


architecture behavior of addsub_2a7fe08e67 is
  signal a_17_32: signed((31 - 1) downto 0);
  signal b_17_35: signed((31 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((32 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "00000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((32 - 1) downto 0);
  signal op_mem_91_20_back: signed((32 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((32 - 1) downto 0);
  signal cast_69_22: signed((32 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((32 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 32, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 32, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_f1849463e9 is
  port (
    a : in std_logic_vector((32 - 1) downto 0);
    b : in std_logic_vector((32 - 1) downto 0);
    s : out std_logic_vector((33 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_f1849463e9;


architecture behavior of addsub_f1849463e9 is
  signal a_17_32: signed((32 - 1) downto 0);
  signal b_17_35: signed((32 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((33 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((33 - 1) downto 0);
  signal op_mem_91_20_back: signed((33 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((33 - 1) downto 0);
  signal cast_69_22: signed((33 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((33 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 33, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 33, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_e63f1751f5 is
  port (
    a : in std_logic_vector((33 - 1) downto 0);
    b : in std_logic_vector((33 - 1) downto 0);
    s : out std_logic_vector((34 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_e63f1751f5;


architecture behavior of addsub_e63f1751f5 is
  signal a_17_32: signed((33 - 1) downto 0);
  signal b_17_35: signed((33 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((34 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "0000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((34 - 1) downto 0);
  signal op_mem_91_20_back: signed((34 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((34 - 1) downto 0);
  signal cast_69_22: signed((34 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((34 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 34, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 34, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_5d08e5a27e is
  port (
    a : in std_logic_vector((34 - 1) downto 0);
    b : in std_logic_vector((34 - 1) downto 0);
    s : out std_logic_vector((35 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_5d08e5a27e;


architecture behavior of addsub_5d08e5a27e is
  signal a_17_32: signed((34 - 1) downto 0);
  signal b_17_35: signed((34 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((35 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "00000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((35 - 1) downto 0);
  signal op_mem_91_20_back: signed((35 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((35 - 1) downto 0);
  signal cast_69_22: signed((35 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((35 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 35, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 35, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_849323e8f6 is
  port (
    a : in std_logic_vector((35 - 1) downto 0);
    b : in std_logic_vector((35 - 1) downto 0);
    s : out std_logic_vector((36 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_849323e8f6;


architecture behavior of addsub_849323e8f6 is
  signal a_17_32: signed((35 - 1) downto 0);
  signal b_17_35: signed((35 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((36 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((36 - 1) downto 0);
  signal op_mem_91_20_back: signed((36 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((36 - 1) downto 0);
  signal cast_69_22: signed((36 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((36 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 36, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 36, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_f62aecf512 is
  port (
    a : in std_logic_vector((36 - 1) downto 0);
    b : in std_logic_vector((36 - 1) downto 0);
    s : out std_logic_vector((37 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_f62aecf512;


architecture behavior of addsub_f62aecf512 is
  signal a_17_32: signed((36 - 1) downto 0);
  signal b_17_35: signed((36 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((37 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "0000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((37 - 1) downto 0);
  signal op_mem_91_20_back: signed((37 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((37 - 1) downto 0);
  signal cast_69_22: signed((37 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((37 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 37, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 37, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_fa63be3dfe is
  port (
    a : in std_logic_vector((37 - 1) downto 0);
    b : in std_logic_vector((37 - 1) downto 0);
    s : out std_logic_vector((38 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_fa63be3dfe;


architecture behavior of addsub_fa63be3dfe is
  signal a_17_32: signed((37 - 1) downto 0);
  signal b_17_35: signed((37 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((38 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "00000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((38 - 1) downto 0);
  signal op_mem_91_20_back: signed((38 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((38 - 1) downto 0);
  signal cast_69_22: signed((38 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((38 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 38, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 38, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_8f8925e093 is
  port (
    a : in std_logic_vector((38 - 1) downto 0);
    b : in std_logic_vector((38 - 1) downto 0);
    s : out std_logic_vector((39 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_8f8925e093;


architecture behavior of addsub_8f8925e093 is
  signal a_17_32: signed((38 - 1) downto 0);
  signal b_17_35: signed((38 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((39 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "000000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((39 - 1) downto 0);
  signal op_mem_91_20_back: signed((39 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((39 - 1) downto 0);
  signal cast_69_22: signed((39 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((39 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 39, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 39, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_958f59c040 is
  port (
    a : in std_logic_vector((39 - 1) downto 0);
    b : in std_logic_vector((39 - 1) downto 0);
    s : out std_logic_vector((40 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_958f59c040;


architecture behavior of addsub_958f59c040 is
  signal a_17_32: signed((39 - 1) downto 0);
  signal b_17_35: signed((39 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((40 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "0000000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((40 - 1) downto 0);
  signal op_mem_91_20_back: signed((40 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((40 - 1) downto 0);
  signal cast_69_22: signed((40 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((40 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 40, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 40, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;


-------------------------------------------------------------------
-- System Generator version 14.2 VHDL source file.
--
-- Copyright(C) 2012 by Xilinx, Inc.  All rights reserved.  This
-- text/file contains proprietary, confidential information of Xilinx,
-- Inc., is distributed under license from Xilinx, Inc., and may be used,
-- copied and/or disclosed only pursuant to the terms of a valid license
-- agreement with Xilinx, Inc.  Xilinx hereby grants you a license to use
-- this text/file solely for design, simulation, implementation and
-- creation of design files limited to Xilinx devices or technologies.
-- Use with non-Xilinx devices or technologies is expressly prohibited
-- and immediately terminates your license unless covered by a separate
-- agreement.
--
-- Xilinx is providing this design, code, or information "as is" solely
-- for use in developing programs and solutions for Xilinx devices.  By
-- providing this design, code, or information as one possible
-- implementation of this feature, application or standard, Xilinx is
-- making no representation that this implementation is free from any
-- claims of infringement.  You are responsible for obtaining any rights
-- you may require for your implementation.  Xilinx expressly disclaims
-- any warranty whatsoever with respect to the adequacy of the
-- implementation, including but not limited to warranties of
-- merchantability or fitness for a particular purpose.
--
-- Xilinx products are not intended for use in life support appliances,
-- devices, or systems.  Use in such applications is expressly prohibited.
--
-- Any modifications that are made to the source code are done at the user's
-- sole risk and will be unsupported.
--
-- This copyright and support notice must be retained as part of this
-- text at all times.  (c) Copyright 1995-2012 Xilinx, Inc.  All rights
-- reserved.
-------------------------------------------------------------------
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;
entity xlregister is
   generic (d_width          : integer := 5;
            init_value       : bit_vector := b"00");
   port (d   : in std_logic_vector (d_width-1 downto 0);
         rst : in std_logic_vector(0 downto 0) := "0";
         en  : in std_logic_vector(0 downto 0) := "1";
         ce  : in std_logic;
         clk : in std_logic;
         q   : out std_logic_vector (d_width-1 downto 0));
end xlregister;
architecture behavior of xlregister is
   component synth_reg_w_init
      generic (width      : integer;
               init_index : integer;
               init_value : bit_vector;
               latency    : integer);
      port (i   : in std_logic_vector(width-1 downto 0);
            ce  : in std_logic;
            clr : in std_logic;
            clk : in std_logic;
            o   : out std_logic_vector(width-1 downto 0));
   end component;
   -- synopsys translate_off
   signal real_d, real_q           : real;
   -- synopsys translate_on
   signal internal_clr             : std_logic;
   signal internal_ce              : std_logic;
begin
   internal_clr <= rst(0) and ce;
   internal_ce  <= en(0) and ce;
   synth_reg_inst : synth_reg_w_init
      generic map (width      => d_width,
                   init_index => 2,
                   init_value => init_value,
                   latency    => 1)
      port map (i   => d,
                ce  => internal_ce,
                clr => internal_clr,
                clk => clk,
                o   => q);
end architecture behavior;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_4b2650d0e6 is
  port (
    a : in std_logic_vector((40 - 1) downto 0);
    b : in std_logic_vector((40 - 1) downto 0);
    s : out std_logic_vector((41 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_4b2650d0e6;


architecture behavior of addsub_4b2650d0e6 is
  signal a_17_32: signed((40 - 1) downto 0);
  signal b_17_35: signed((40 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((41 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "00000000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((41 - 1) downto 0);
  signal op_mem_91_20_back: signed((41 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((41 - 1) downto 0);
  signal cast_69_22: signed((41 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((41 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 41, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 41, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_614a30f654 is
  port (
    a : in std_logic_vector((41 - 1) downto 0);
    b : in std_logic_vector((41 - 1) downto 0);
    s : out std_logic_vector((42 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_614a30f654;


architecture behavior of addsub_614a30f654 is
  signal a_17_32: signed((41 - 1) downto 0);
  signal b_17_35: signed((41 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((42 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "000000000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((42 - 1) downto 0);
  signal op_mem_91_20_back: signed((42 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((42 - 1) downto 0);
  signal cast_69_22: signed((42 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((42 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 42, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 42, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_b829aecccd is
  port (
    a : in std_logic_vector((42 - 1) downto 0);
    b : in std_logic_vector((42 - 1) downto 0);
    s : out std_logic_vector((43 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_b829aecccd;


architecture behavior of addsub_b829aecccd is
  signal a_17_32: signed((42 - 1) downto 0);
  signal b_17_35: signed((42 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((43 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "0000000000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((43 - 1) downto 0);
  signal op_mem_91_20_back: signed((43 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((43 - 1) downto 0);
  signal cast_69_22: signed((43 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((43 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 43, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 43, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_d65218df4d is
  port (
    a : in std_logic_vector((43 - 1) downto 0);
    b : in std_logic_vector((43 - 1) downto 0);
    s : out std_logic_vector((44 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_d65218df4d;


architecture behavior of addsub_d65218df4d is
  signal a_17_32: signed((43 - 1) downto 0);
  signal b_17_35: signed((43 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (1 - 1)) of signed((44 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    0 => "00000000000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((44 - 1) downto 0);
  signal op_mem_91_20_back: signed((44 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (1 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    0 => "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((44 - 1) downto 0);
  signal cast_69_22: signed((44 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((44 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(0);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(0);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 22, 44, 22);
  cast_69_22 <= s2s_cast(b_17_35, 22, 44, 22);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;


-------------------------------------------------------------------
-- System Generator version 14.2 VHDL source file.
--
-- Copyright(C) 2012 by Xilinx, Inc.  All rights reserved.  This
-- text/file contains proprietary, confidential information of Xilinx,
-- Inc., is distributed under license from Xilinx, Inc., and may be used,
-- copied and/or disclosed only pursuant to the terms of a valid license
-- agreement with Xilinx, Inc.  Xilinx hereby grants you a license to use
-- this text/file solely for design, simulation, implementation and
-- creation of design files limited to Xilinx devices or technologies.
-- Use with non-Xilinx devices or technologies is expressly prohibited
-- and immediately terminates your license unless covered by a separate
-- agreement.
--
-- Xilinx is providing this design, code, or information "as is" solely
-- for use in developing programs and solutions for Xilinx devices.  By
-- providing this design, code, or information as one possible
-- implementation of this feature, application or standard, Xilinx is
-- making no representation that this implementation is free from any
-- claims of infringement.  You are responsible for obtaining any rights
-- you may require for your implementation.  Xilinx expressly disclaims
-- any warranty whatsoever with respect to the adequacy of the
-- implementation, including but not limited to warranties of
-- merchantability or fitness for a particular purpose.
--
-- Xilinx products are not intended for use in life support appliances,
-- devices, or systems.  Use in such applications is expressly prohibited.
--
-- Any modifications that are made to the source code are done at the user's
-- sole risk and will be unsupported.
--
-- This copyright and support notice must be retained as part of this
-- text at all times.  (c) Copyright 1995-2012 Xilinx, Inc.  All rights
-- reserved.
-------------------------------------------------------------------
-- synopsys translate_off
library XilinxCoreLib;
-- synopsys translate_on
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;
entity xlcounter_limit_cichbcic_core is
  generic (
    core_name0: string := "";
    op_width: integer := 5;
    op_arith: integer := xlSigned;
    cnt_63_48: integer:= 0;
    cnt_47_32: integer:= 0;
    cnt_31_16: integer:= 0;
    cnt_15_0: integer:= 0;
    count_limited: integer := 0
  );
  port (
    ce: in std_logic;
    clr: in std_logic;
    clk: in std_logic;
    op: out std_logic_vector(op_width - 1 downto 0);
    up: in std_logic_vector(0 downto 0) := (others => '0');
    en: in std_logic_vector(0 downto 0);
    rst: in std_logic_vector(0 downto 0)
  );
end xlcounter_limit_cichbcic_core ;
architecture behavior of xlcounter_limit_cichbcic_core is
  signal high_cnt_to: std_logic_vector(31 downto 0);
  signal low_cnt_to: std_logic_vector(31 downto 0);
  signal cnt_to: std_logic_vector(63 downto 0);
  signal core_sinit, op_thresh0, core_ce: std_logic;
  signal rst_overrides_en: std_logic;
  signal op_net: std_logic_vector(op_width - 1 downto 0);
  -- synopsys translate_off
  signal real_op : real;
   -- synopsys translate_on
  function equals(op, cnt_to : std_logic_vector; width, arith : integer)
    return std_logic
  is
    variable signed_op, signed_cnt_to : signed (width - 1 downto 0);
    variable unsigned_op, unsigned_cnt_to : unsigned (width - 1 downto 0);
    variable result : std_logic;
  begin
    -- synopsys translate_off
    if ((is_XorU(op)) or (is_XorU(cnt_to)) ) then
      result := '0';
      return result;
    end if;
    -- synopsys translate_on
    if (op = cnt_to) then
      result := '1';
    else
      result := '0';
    end if;
    return result;
  end;
  component cntr_11_0_6400a835b899f648
    port (
      clk: in std_logic;
      ce: in std_logic;
      SINIT: in std_logic;
      q: out std_logic_vector(op_width - 1 downto 0)
    );
  end component;

  attribute syn_black_box of cntr_11_0_6400a835b899f648:
    component is true;
  attribute fpga_dont_touch of cntr_11_0_6400a835b899f648:
    component is "true";
  attribute box_type of cntr_11_0_6400a835b899f648:
    component  is "black_box";
  component cntr_11_0_a0d692c18ceb283e
    port (
      clk: in std_logic;
      ce: in std_logic;
      SINIT: in std_logic;
      q: out std_logic_vector(op_width - 1 downto 0)
    );
  end component;

  attribute syn_black_box of cntr_11_0_a0d692c18ceb283e:
    component is true;
  attribute fpga_dont_touch of cntr_11_0_a0d692c18ceb283e:
    component is "true";
  attribute box_type of cntr_11_0_a0d692c18ceb283e:
    component  is "black_box";
-- synopsys translate_off
  constant zeroVec : std_logic_vector(op_width - 1 downto 0) := (others => '0');
  constant oneVec : std_logic_vector(op_width - 1 downto 0) := (others => '1');
  constant zeroStr : string(1 to op_width) :=
    std_logic_vector_to_bin_string(zeroVec);
  constant oneStr : string(1 to op_width) :=
    std_logic_vector_to_bin_string(oneVec);
-- synopsys translate_on
begin
  -- synopsys translate_off
  -- synopsys translate_on
  cnt_to(63 downto 48) <= integer_to_std_logic_vector(cnt_63_48, 16, op_arith);
  cnt_to(47 downto 32) <= integer_to_std_logic_vector(cnt_47_32, 16, op_arith);
  cnt_to(31 downto 16) <= integer_to_std_logic_vector(cnt_31_16, 16, op_arith);
  cnt_to(15 downto 0) <= integer_to_std_logic_vector(cnt_15_0, 16, op_arith);
  op <= op_net;
  core_ce <= ce and en(0);
  rst_overrides_en <= rst(0) or en(0);
  limit : if (count_limited = 1) generate
    eq_cnt_to : process (op_net, cnt_to)
    begin
      op_thresh0 <= equals(op_net, cnt_to(op_width - 1 downto 0),
                     op_width, op_arith);
    end process;
    core_sinit <= (op_thresh0 or clr or rst(0)) and ce and rst_overrides_en;
  end generate;
  no_limit : if (count_limited = 0) generate
    core_sinit <= (clr or rst(0)) and ce and rst_overrides_en;
  end generate;
  comp0: if ((core_name0 = "cntr_11_0_6400a835b899f648")) generate
    core_instance0: cntr_11_0_6400a835b899f648
      port map (
        clk => clk,
        ce => core_ce,
        SINIT => core_sinit,
        q => op_net
      );
  end generate;
  comp1: if ((core_name0 = "cntr_11_0_a0d692c18ceb283e")) generate
    core_instance1: cntr_11_0_a0d692c18ceb283e
      port map (
        clk => clk,
        ce => core_ce,
        SINIT => core_sinit,
        q => op_net
      );
  end generate;
end  behavior;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity relational_0c602bf0fd is
  port (
    a : in std_logic_vector((3 - 1) downto 0);
    b : in std_logic_vector((16 - 1) downto 0);
    op : out std_logic_vector((1 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end relational_0c602bf0fd;


architecture behavior of relational_0c602bf0fd is
  signal a_1_31: unsigned((3 - 1) downto 0);
  signal b_1_34: signed((16 - 1) downto 0);
  signal cast_12_12: signed((18 - 1) downto 0);
  signal cast_12_17: signed((18 - 1) downto 0);
  signal result_12_3_rel: boolean;
begin
  a_1_31 <= std_logic_vector_to_unsigned(a);
  b_1_34 <= std_logic_vector_to_signed(b);
  cast_12_12 <= u2s_cast(a_1_31, 0, 18, 14);
  cast_12_17 <= s2s_cast(b_1_34, 14, 18, 14);
  result_12_3_rel <= cast_12_12 = cast_12_17;
  op <= boolean_to_vector(result_12_3_rel);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_b17f382b30 is
  port (
    a : in std_logic_vector((35 - 1) downto 0);
    b : in std_logic_vector((35 - 1) downto 0);
    s : out std_logic_vector((36 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_b17f382b30;


architecture behavior of addsub_b17f382b30 is
  signal a_17_32: signed((35 - 1) downto 0);
  signal b_17_35: signed((35 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (8 - 1)) of signed((36 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    "000000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((36 - 1) downto 0);
  signal op_mem_91_20_back: signed((36 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (8 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    "0",
    "0",
    "0",
    "0",
    "0",
    "0",
    "0",
    "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((36 - 1) downto 0);
  signal cast_69_22: signed((36 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((36 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(7);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        for i in 7 downto 1 loop 
          op_mem_91_20(i) <= op_mem_91_20(i-1);
        end loop;
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(7);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        for i_x_000000 in 7 downto 1 loop 
          cout_mem_92_22(i_x_000000) <= cout_mem_92_22(i_x_000000-1);
        end loop;
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 12, 36, 12);
  cast_69_22 <= s2s_cast(b_17_35, 12, 36, 12);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_1f7f5d69b1 is
  port (
    a : in std_logic_vector((36 - 1) downto 0);
    b : in std_logic_vector((36 - 1) downto 0);
    s : out std_logic_vector((37 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_1f7f5d69b1;


architecture behavior of addsub_1f7f5d69b1 is
  signal a_17_32: signed((36 - 1) downto 0);
  signal b_17_35: signed((36 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (8 - 1)) of signed((37 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    "0000000000000000000000000000000000000",
    "0000000000000000000000000000000000000",
    "0000000000000000000000000000000000000",
    "0000000000000000000000000000000000000",
    "0000000000000000000000000000000000000",
    "0000000000000000000000000000000000000",
    "0000000000000000000000000000000000000",
    "0000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((37 - 1) downto 0);
  signal op_mem_91_20_back: signed((37 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (8 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    "0",
    "0",
    "0",
    "0",
    "0",
    "0",
    "0",
    "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((37 - 1) downto 0);
  signal cast_69_22: signed((37 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((37 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(7);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        for i in 7 downto 1 loop 
          op_mem_91_20(i) <= op_mem_91_20(i-1);
        end loop;
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(7);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        for i_x_000000 in 7 downto 1 loop 
          cout_mem_92_22(i_x_000000) <= cout_mem_92_22(i_x_000000-1);
        end loop;
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 12, 37, 12);
  cast_69_22 <= s2s_cast(b_17_35, 12, 37, 12);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_619c589516 is
  port (
    a : in std_logic_vector((37 - 1) downto 0);
    b : in std_logic_vector((37 - 1) downto 0);
    s : out std_logic_vector((38 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_619c589516;


architecture behavior of addsub_619c589516 is
  signal a_17_32: signed((37 - 1) downto 0);
  signal b_17_35: signed((37 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (8 - 1)) of signed((38 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    "00000000000000000000000000000000000000",
    "00000000000000000000000000000000000000",
    "00000000000000000000000000000000000000",
    "00000000000000000000000000000000000000",
    "00000000000000000000000000000000000000",
    "00000000000000000000000000000000000000",
    "00000000000000000000000000000000000000",
    "00000000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((38 - 1) downto 0);
  signal op_mem_91_20_back: signed((38 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (8 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    "0",
    "0",
    "0",
    "0",
    "0",
    "0",
    "0",
    "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((38 - 1) downto 0);
  signal cast_69_22: signed((38 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((38 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(7);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        for i in 7 downto 1 loop 
          op_mem_91_20(i) <= op_mem_91_20(i-1);
        end loop;
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(7);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        for i_x_000000 in 7 downto 1 loop 
          cout_mem_92_22(i_x_000000) <= cout_mem_92_22(i_x_000000-1);
        end loop;
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 12, 38, 12);
  cast_69_22 <= s2s_cast(b_17_35, 12, 38, 12);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity relational_fda93ba512 is
  port (
    a : in std_logic_vector((2 - 1) downto 0);
    b : in std_logic_vector((16 - 1) downto 0);
    op : out std_logic_vector((1 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end relational_fda93ba512;


architecture behavior of relational_fda93ba512 is
  signal a_1_31: unsigned((2 - 1) downto 0);
  signal b_1_34: signed((16 - 1) downto 0);
  signal cast_12_12: signed((17 - 1) downto 0);
  signal cast_12_17: signed((17 - 1) downto 0);
  signal result_12_3_rel: boolean;
begin
  a_1_31 <= std_logic_vector_to_unsigned(a);
  b_1_34 <= std_logic_vector_to_signed(b);
  cast_12_12 <= u2s_cast(a_1_31, 0, 17, 14);
  cast_12_17 <= s2s_cast(b_1_34, 14, 17, 14);
  result_12_3_rel <= cast_12_12 = cast_12_17;
  op <= boolean_to_vector(result_12_3_rel);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_38883d04b4 is
  port (
    a : in std_logic_vector((32 - 1) downto 0);
    b : in std_logic_vector((32 - 1) downto 0);
    s : out std_logic_vector((33 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_38883d04b4;


architecture behavior of addsub_38883d04b4 is
  signal a_17_32: signed((32 - 1) downto 0);
  signal b_17_35: signed((32 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (4 - 1)) of signed((33 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    "000000000000000000000000000000000",
    "000000000000000000000000000000000",
    "000000000000000000000000000000000",
    "000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((33 - 1) downto 0);
  signal op_mem_91_20_back: signed((33 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (4 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    "0",
    "0",
    "0",
    "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((33 - 1) downto 0);
  signal cast_69_22: signed((33 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((33 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(3);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        for i in 3 downto 1 loop 
          op_mem_91_20(i) <= op_mem_91_20(i-1);
        end loop;
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(3);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        for i_x_000000 in 3 downto 1 loop 
          cout_mem_92_22(i_x_000000) <= cout_mem_92_22(i_x_000000-1);
        end loop;
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 12, 33, 12);
  cast_69_22 <= s2s_cast(b_17_35, 12, 33, 12);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_20d573ab12 is
  port (
    a : in std_logic_vector((33 - 1) downto 0);
    b : in std_logic_vector((33 - 1) downto 0);
    s : out std_logic_vector((34 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_20d573ab12;


architecture behavior of addsub_20d573ab12 is
  signal a_17_32: signed((33 - 1) downto 0);
  signal b_17_35: signed((33 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (4 - 1)) of signed((34 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    "0000000000000000000000000000000000",
    "0000000000000000000000000000000000",
    "0000000000000000000000000000000000",
    "0000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((34 - 1) downto 0);
  signal op_mem_91_20_back: signed((34 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (4 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    "0",
    "0",
    "0",
    "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((34 - 1) downto 0);
  signal cast_69_22: signed((34 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((34 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(3);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        for i in 3 downto 1 loop 
          op_mem_91_20(i) <= op_mem_91_20(i-1);
        end loop;
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(3);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        for i_x_000000 in 3 downto 1 loop 
          cout_mem_92_22(i_x_000000) <= cout_mem_92_22(i_x_000000-1);
        end loop;
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 12, 34, 12);
  cast_69_22 <= s2s_cast(b_17_35, 12, 34, 12);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity addsub_a3fafab502 is
  port (
    a : in std_logic_vector((34 - 1) downto 0);
    b : in std_logic_vector((34 - 1) downto 0);
    s : out std_logic_vector((35 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end addsub_a3fafab502;


architecture behavior of addsub_a3fafab502 is
  signal a_17_32: signed((34 - 1) downto 0);
  signal b_17_35: signed((34 - 1) downto 0);
  type array_type_op_mem_91_20 is array (0 to (4 - 1)) of signed((35 - 1) downto 0);
  signal op_mem_91_20: array_type_op_mem_91_20 := (
    "00000000000000000000000000000000000",
    "00000000000000000000000000000000000",
    "00000000000000000000000000000000000",
    "00000000000000000000000000000000000");
  signal op_mem_91_20_front_din: signed((35 - 1) downto 0);
  signal op_mem_91_20_back: signed((35 - 1) downto 0);
  signal op_mem_91_20_push_front_pop_back_en: std_logic;
  type array_type_cout_mem_92_22 is array (0 to (4 - 1)) of unsigned((1 - 1) downto 0);
  signal cout_mem_92_22: array_type_cout_mem_92_22 := (
    "0",
    "0",
    "0",
    "0");
  signal cout_mem_92_22_front_din: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_back: unsigned((1 - 1) downto 0);
  signal cout_mem_92_22_push_front_pop_back_en: std_logic;
  signal prev_mode_93_22_next: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22: unsigned((3 - 1) downto 0);
  signal prev_mode_93_22_reg_i: std_logic_vector((3 - 1) downto 0);
  signal prev_mode_93_22_reg_o: std_logic_vector((3 - 1) downto 0);
  signal cast_69_18: signed((35 - 1) downto 0);
  signal cast_69_22: signed((35 - 1) downto 0);
  signal internal_s_69_5_addsub: signed((35 - 1) downto 0);
begin
  a_17_32 <= std_logic_vector_to_signed(a);
  b_17_35 <= std_logic_vector_to_signed(b);
  op_mem_91_20_back <= op_mem_91_20(3);
  proc_op_mem_91_20: process (clk)
  is
    variable i: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (op_mem_91_20_push_front_pop_back_en = '1')) then
        for i in 3 downto 1 loop 
          op_mem_91_20(i) <= op_mem_91_20(i-1);
        end loop;
        op_mem_91_20(0) <= op_mem_91_20_front_din;
      end if;
    end if;
  end process proc_op_mem_91_20;
  cout_mem_92_22_back <= cout_mem_92_22(3);
  proc_cout_mem_92_22: process (clk)
  is
    variable i_x_000000: integer;
  begin
    if (clk'event and (clk = '1')) then
      if ((ce = '1') and (cout_mem_92_22_push_front_pop_back_en = '1')) then
        for i_x_000000 in 3 downto 1 loop 
          cout_mem_92_22(i_x_000000) <= cout_mem_92_22(i_x_000000-1);
        end loop;
        cout_mem_92_22(0) <= cout_mem_92_22_front_din;
      end if;
    end if;
  end process proc_cout_mem_92_22;
  prev_mode_93_22_reg_i <= unsigned_to_std_logic_vector(prev_mode_93_22_next);
  prev_mode_93_22 <= std_logic_vector_to_unsigned(prev_mode_93_22_reg_o);
  prev_mode_93_22_reg_inst: entity work.synth_reg_w_init
    generic map (
      init_index => 2, 
      init_value => b"010", 
      latency => 1, 
      width => 3)
    port map (
      ce => ce, 
      clk => clk, 
      clr => clr, 
      i => prev_mode_93_22_reg_i, 
      o => prev_mode_93_22_reg_o);
  cast_69_18 <= s2s_cast(a_17_32, 12, 35, 12);
  cast_69_22 <= s2s_cast(b_17_35, 12, 35, 12);
  internal_s_69_5_addsub <= cast_69_18 + cast_69_22;
  op_mem_91_20_front_din <= internal_s_69_5_addsub;
  op_mem_91_20_push_front_pop_back_en <= '1';
  cout_mem_92_22_front_din <= std_logic_vector_to_unsigned("0");
  cout_mem_92_22_push_front_pop_back_en <= '1';
  prev_mode_93_22_next <= std_logic_vector_to_unsigned("000");
  s <= signed_to_std_logic_vector(op_mem_91_20_back);
end behavior;


-------------------------------------------------------------------
-- System Generator version 14.2 VHDL source file.
--
-- Copyright(C) 2012 by Xilinx, Inc.  All rights reserved.  This
-- text/file contains proprietary, confidential information of Xilinx,
-- Inc., is distributed under license from Xilinx, Inc., and may be used,
-- copied and/or disclosed only pursuant to the terms of a valid license
-- agreement with Xilinx, Inc.  Xilinx hereby grants you a license to use
-- this text/file solely for design, simulation, implementation and
-- creation of design files limited to Xilinx devices or technologies.
-- Use with non-Xilinx devices or technologies is expressly prohibited
-- and immediately terminates your license unless covered by a separate
-- agreement.
--
-- Xilinx is providing this design, code, or information "as is" solely
-- for use in developing programs and solutions for Xilinx devices.  By
-- providing this design, code, or information as one possible
-- implementation of this feature, application or standard, Xilinx is
-- making no representation that this implementation is free from any
-- claims of infringement.  You are responsible for obtaining any rights
-- you may require for your implementation.  Xilinx expressly disclaims
-- any warranty whatsoever with respect to the adequacy of the
-- implementation, including but not limited to warranties of
-- merchantability or fitness for a particular purpose.
--
-- Xilinx products are not intended for use in life support appliances,
-- devices, or systems.  Use in such applications is expressly prohibited.
--
-- Any modifications that are made to the source code are done at the user's
-- sole risk and will be unsupported.
--
-- This copyright and support notice must be retained as part of this
-- text at all times.  (c) Copyright 1995-2012 Xilinx, Inc.  All rights
-- reserved.
-------------------------------------------------------------------
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;
entity convert_func_call is
    generic (
        din_width    : integer := 16;
        din_bin_pt   : integer := 4;
        din_arith    : integer := xlUnsigned;
        dout_width   : integer := 8;
        dout_bin_pt  : integer := 2;
        dout_arith   : integer := xlUnsigned;
        quantization : integer := xlTruncate;
        overflow     : integer := xlWrap);
    port (
        din : in std_logic_vector (din_width-1 downto 0);
        result : out std_logic_vector (dout_width-1 downto 0));
end convert_func_call;
architecture behavior of convert_func_call is
begin
    result <= convert_type(din, din_width, din_bin_pt, din_arith,
                           dout_width, dout_bin_pt, dout_arith,
                           quantization, overflow);
end behavior;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;
entity xlconvert is
    generic (
        din_width    : integer := 16;
        din_bin_pt   : integer := 4;
        din_arith    : integer := xlUnsigned;
        dout_width   : integer := 8;
        dout_bin_pt  : integer := 2;
        dout_arith   : integer := xlUnsigned;
        en_width     : integer := 1;
        en_bin_pt    : integer := 0;
        en_arith     : integer := xlUnsigned;
        bool_conversion : integer :=0;
        latency      : integer := 0;
        quantization : integer := xlTruncate;
        overflow     : integer := xlWrap);
    port (
        din : in std_logic_vector (din_width-1 downto 0);
        en  : in std_logic_vector (en_width-1 downto 0);
        ce  : in std_logic;
        clr : in std_logic;
        clk : in std_logic;
        dout : out std_logic_vector (dout_width-1 downto 0));
end xlconvert;
architecture behavior of xlconvert is
    component synth_reg
        generic (width       : integer;
                 latency     : integer);
        port (i       : in std_logic_vector(width-1 downto 0);
              ce      : in std_logic;
              clr     : in std_logic;
              clk     : in std_logic;
              o       : out std_logic_vector(width-1 downto 0));
    end component;
    component convert_func_call
        generic (
            din_width    : integer := 16;
            din_bin_pt   : integer := 4;
            din_arith    : integer := xlUnsigned;
            dout_width   : integer := 8;
            dout_bin_pt  : integer := 2;
            dout_arith   : integer := xlUnsigned;
            quantization : integer := xlTruncate;
            overflow     : integer := xlWrap);
        port (
            din : in std_logic_vector (din_width-1 downto 0);
            result : out std_logic_vector (dout_width-1 downto 0));
    end component;
    -- synopsys translate_off
    -- synopsys translate_on
    signal result : std_logic_vector(dout_width-1 downto 0);
    signal internal_ce : std_logic;
begin
    -- synopsys translate_off
    -- synopsys translate_on
    internal_ce <= ce and en(0);

    bool_conversion_generate : if (bool_conversion = 1)
    generate
      result <= din;
    end generate;
    std_conversion_generate : if (bool_conversion = 0)
    generate
      convert : convert_func_call
        generic map (
          din_width   => din_width,
          din_bin_pt  => din_bin_pt,
          din_arith   => din_arith,
          dout_width  => dout_width,
          dout_bin_pt => dout_bin_pt,
          dout_arith  => dout_arith,
          quantization => quantization,
          overflow     => overflow)
        port map (
          din => din,
          result => result);
    end generate;
    latency_test : if (latency > 0) generate
        reg : synth_reg
            generic map (
              width => dout_width,
              latency => latency
            )
            port map (
              i => result,
              ce => internal_ce,
              clr => clr,
              clk => clk,
              o => dout
            );
    end generate;
    latency0 : if (latency = 0)
    generate
        dout <= result;
    end generate latency0;
end  behavior;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.conv_pkg.all;

entity reinterpret_41e5bd5e40 is
  port (
    input_port : in std_logic_vector((44 - 1) downto 0);
    output_port : out std_logic_vector((44 - 1) downto 0);
    clk : in std_logic;
    ce : in std_logic;
    clr : in std_logic);
end reinterpret_41e5bd5e40;


architecture behavior of reinterpret_41e5bd5e40 is
  signal input_port_1_40: signed((44 - 1) downto 0);
begin
  input_port_1_40 <= std_logic_vector_to_signed(input_port);
  output_port <= signed_to_std_logic_vector(input_port_1_40);
end behavior;

library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/adder_tree/addr1"

entity addr1_entity_7a7c1e47fe is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    data_a: in std_logic_vector(47 downto 0); 
    data_b: in std_logic_vector(47 downto 0); 
    en_in: in std_logic; 
    sum_ab: out std_logic_vector(48 downto 0)
  );
end addr1_entity_7a7c1e47fe;

architecture structural of addr1_entity_7a7c1e47fe is
  signal addsub_s_net_x0: std_logic_vector(48 downto 0);
  signal ce_1_sg_x0: std_logic;
  signal clk_1_sg_x0: std_logic;
  signal mult_p_net_x1: std_logic_vector(47 downto 0);
  signal mult_p_net_x2: std_logic_vector(47 downto 0);
  signal xlsub1_logical_out1_x0: std_logic;

begin
  ce_1_sg_x0 <= ce_1;
  clk_1_sg_x0 <= clk_1;
  mult_p_net_x1 <= data_a;
  mult_p_net_x2 <= data_b;
  xlsub1_logical_out1_x0 <= en_in;
  sum_ab <= addsub_s_net_x0;

  addsub: entity work.xladdsub_cichbcic_core
    generic map (
      a_arith => xlSigned,
      a_bin_pt => 32,
      a_width => 48,
      b_arith => xlSigned,
      b_bin_pt => 32,
      b_width => 48,
      c_has_c_out => 0,
      c_latency => 1,
      c_output_width => 49,
      core_name0 => "addsb_11_0_a6179ec1a236388e",
      en_arith => xlUnsigned,
      en_bin_pt => 0,
      en_width => 1,
      extra_registers => 1,
      full_s_arith => 2,
      full_s_width => 49,
      latency => 2,
      overflow => 1,
      quantization => 1,
      s_arith => xlSigned,
      s_bin_pt => 32,
      s_width => 49
    )
    port map (
      a => mult_p_net_x2,
      b => mult_p_net_x1,
      ce => ce_1_sg_x0,
      clk => clk_1_sg_x0,
      clr => '0',
      en(0) => xlsub1_logical_out1_x0,
      s => addsub_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/adder_tree/addr11"

entity addr11_entity_4dc2a4fe75 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    data_a: in std_logic_vector(50 downto 0); 
    data_b: in std_logic_vector(47 downto 0); 
    en_in: in std_logic; 
    sum_ab: out std_logic_vector(51 downto 0)
  );
end addr11_entity_4dc2a4fe75;

architecture structural of addr11_entity_4dc2a4fe75 is
  signal adder_outs3_2_x0: std_logic_vector(47 downto 0);
  signal addsub_s_net_x1: std_logic_vector(50 downto 0);
  signal addsub_s_net_x2: std_logic_vector(51 downto 0);
  signal ce_1_sg_x1: std_logic;
  signal clk_1_sg_x1: std_logic;
  signal xlsub1_logical_out1_x1: std_logic;

begin
  ce_1_sg_x1 <= ce_1;
  clk_1_sg_x1 <= clk_1;
  addsub_s_net_x1 <= data_a;
  adder_outs3_2_x0 <= data_b;
  xlsub1_logical_out1_x1 <= en_in;
  sum_ab <= addsub_s_net_x2;

  addsub: entity work.xladdsub_cichbcic_core
    generic map (
      a_arith => xlSigned,
      a_bin_pt => 32,
      a_width => 48,
      b_arith => xlSigned,
      b_bin_pt => 32,
      b_width => 51,
      c_has_c_out => 0,
      c_latency => 1,
      c_output_width => 52,
      core_name0 => "addsb_11_0_5bfb73f1589643d3",
      en_arith => xlUnsigned,
      en_bin_pt => 0,
      en_width => 1,
      extra_registers => 1,
      full_s_arith => 2,
      full_s_width => 52,
      latency => 2,
      overflow => 1,
      quantization => 1,
      s_arith => xlSigned,
      s_bin_pt => 32,
      s_width => 52
    )
    port map (
      a => adder_outs3_2_x0,
      b => addsub_s_net_x1,
      ce => ce_1_sg_x1,
      clk => clk_1_sg_x1,
      clr => '0',
      en(0) => xlsub1_logical_out1_x1,
      s => addsub_s_net_x2
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/adder_tree/addr6"

entity addr6_entity_6c203a7cff is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    data_a: in std_logic_vector(48 downto 0); 
    data_b: in std_logic_vector(48 downto 0); 
    en_in: in std_logic; 
    sum_ab: out std_logic_vector(49 downto 0)
  );
end addr6_entity_6c203a7cff;

architecture structural of addr6_entity_6c203a7cff is
  signal addsub_s_net_x2: std_logic_vector(48 downto 0);
  signal addsub_s_net_x3: std_logic_vector(48 downto 0);
  signal addsub_s_net_x4: std_logic_vector(49 downto 0);
  signal ce_1_sg_x5: std_logic;
  signal clk_1_sg_x5: std_logic;
  signal xlsub1_logical_out1_x5: std_logic;

begin
  ce_1_sg_x5 <= ce_1;
  clk_1_sg_x5 <= clk_1;
  addsub_s_net_x2 <= data_a;
  addsub_s_net_x3 <= data_b;
  xlsub1_logical_out1_x5 <= en_in;
  sum_ab <= addsub_s_net_x4;

  addsub: entity work.xladdsub_cichbcic_core
    generic map (
      a_arith => xlSigned,
      a_bin_pt => 32,
      a_width => 49,
      b_arith => xlSigned,
      b_bin_pt => 32,
      b_width => 49,
      c_has_c_out => 0,
      c_latency => 1,
      c_output_width => 50,
      core_name0 => "addsb_11_0_e35a3bf39f366fd8",
      en_arith => xlUnsigned,
      en_bin_pt => 0,
      en_width => 1,
      extra_registers => 1,
      full_s_arith => 2,
      full_s_width => 50,
      latency => 2,
      overflow => 1,
      quantization => 1,
      s_arith => xlSigned,
      s_bin_pt => 32,
      s_width => 50
    )
    port map (
      a => addsub_s_net_x3,
      b => addsub_s_net_x2,
      ce => ce_1_sg_x5,
      clk => clk_1_sg_x5,
      clr => '0',
      en(0) => xlsub1_logical_out1_x5,
      s => addsub_s_net_x4
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/adder_tree/addr9"

entity addr9_entity_9ae2508b3a is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    data_a: in std_logic_vector(49 downto 0); 
    data_b: in std_logic_vector(49 downto 0); 
    en_in: in std_logic; 
    sum_ab: out std_logic_vector(50 downto 0)
  );
end addr9_entity_9ae2508b3a;

architecture structural of addr9_entity_9ae2508b3a is
  signal addsub_s_net_x6: std_logic_vector(49 downto 0);
  signal addsub_s_net_x7: std_logic_vector(49 downto 0);
  signal addsub_s_net_x8: std_logic_vector(50 downto 0);
  signal ce_1_sg_x7: std_logic;
  signal clk_1_sg_x7: std_logic;
  signal xlsub1_logical_out1_x7: std_logic;

begin
  ce_1_sg_x7 <= ce_1;
  clk_1_sg_x7 <= clk_1;
  addsub_s_net_x6 <= data_a;
  addsub_s_net_x7 <= data_b;
  xlsub1_logical_out1_x7 <= en_in;
  sum_ab <= addsub_s_net_x8;

  addsub: entity work.xladdsub_cichbcic_core
    generic map (
      a_arith => xlSigned,
      a_bin_pt => 32,
      a_width => 50,
      b_arith => xlSigned,
      b_bin_pt => 32,
      b_width => 50,
      c_has_c_out => 0,
      c_latency => 1,
      c_output_width => 51,
      core_name0 => "addsb_11_0_5de09ee679db1560",
      en_arith => xlUnsigned,
      en_bin_pt => 0,
      en_width => 1,
      extra_registers => 1,
      full_s_arith => 2,
      full_s_width => 51,
      latency => 2,
      overflow => 1,
      quantization => 1,
      s_arith => xlSigned,
      s_bin_pt => 32,
      s_width => 51
    )
    port map (
      a => addsub_s_net_x7,
      b => addsub_s_net_x6,
      ce => ce_1_sg_x7,
      clk => clk_1_sg_x7,
      clr => '0',
      en(0) => xlsub1_logical_out1_x7,
      s => addsub_s_net_x8
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/adder_tree"

entity adder_tree_entity_418f80e00e is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din1: in std_logic_vector(47 downto 0); 
    din2: in std_logic_vector(47 downto 0); 
    din3: in std_logic_vector(47 downto 0); 
    din4: in std_logic_vector(47 downto 0); 
    din5: in std_logic_vector(47 downto 0); 
    din6: in std_logic_vector(47 downto 0); 
    din7: in std_logic_vector(47 downto 0); 
    din8: in std_logic_vector(47 downto 0); 
    din9: in std_logic_vector(47 downto 0); 
    en_in: in std_logic; 
    dout: out std_logic_vector(51 downto 0)
  );
end adder_tree_entity_418f80e00e;

architecture structural of adder_tree_entity_418f80e00e is
  signal adder_outs1_5: std_logic_vector(47 downto 0);
  signal adder_outs2_3: std_logic_vector(47 downto 0);
  signal adder_outs3_2_x0: std_logic_vector(47 downto 0);
  signal addsub_s_net_x10: std_logic_vector(51 downto 0);
  signal addsub_s_net_x2: std_logic_vector(48 downto 0);
  signal addsub_s_net_x4: std_logic_vector(48 downto 0);
  signal addsub_s_net_x5: std_logic_vector(48 downto 0);
  signal addsub_s_net_x6: std_logic_vector(48 downto 0);
  signal addsub_s_net_x7: std_logic_vector(49 downto 0);
  signal addsub_s_net_x8: std_logic_vector(49 downto 0);
  signal addsub_s_net_x9: std_logic_vector(50 downto 0);
  signal ce_1_sg_x8: std_logic;
  signal clk_1_sg_x8: std_logic;
  signal mult_p_net_x0: std_logic_vector(47 downto 0);
  signal mult_p_net_x10: std_logic_vector(47 downto 0);
  signal mult_p_net_x11: std_logic_vector(47 downto 0);
  signal mult_p_net_x12: std_logic_vector(47 downto 0);
  signal mult_p_net_x13: std_logic_vector(47 downto 0);
  signal mult_p_net_x14: std_logic_vector(47 downto 0);
  signal mult_p_net_x15: std_logic_vector(47 downto 0);
  signal mult_p_net_x16: std_logic_vector(47 downto 0);
  signal mult_p_net_x9: std_logic_vector(47 downto 0);
  signal xlsub1_logical_out1_x8: std_logic;

begin
  ce_1_sg_x8 <= ce_1;
  clk_1_sg_x8 <= clk_1;
  mult_p_net_x9 <= din1;
  mult_p_net_x12 <= din2;
  mult_p_net_x13 <= din3;
  mult_p_net_x14 <= din4;
  mult_p_net_x15 <= din5;
  mult_p_net_x16 <= din6;
  mult_p_net_x10 <= din7;
  mult_p_net_x11 <= din8;
  mult_p_net_x0 <= din9;
  xlsub1_logical_out1_x8 <= en_in;
  dout <= addsub_s_net_x10;

  addr11_4dc2a4fe75: entity work.addr11_entity_4dc2a4fe75
    port map (
      ce_1 => ce_1_sg_x8,
      clk_1 => clk_1_sg_x8,
      data_a => addsub_s_net_x9,
      data_b => adder_outs3_2_x0,
      en_in => xlsub1_logical_out1_x8,
      sum_ab => addsub_s_net_x10
    );

  addr1_7a7c1e47fe: entity work.addr1_entity_7a7c1e47fe
    port map (
      ce_1 => ce_1_sg_x8,
      clk_1 => clk_1_sg_x8,
      data_a => mult_p_net_x9,
      data_b => mult_p_net_x12,
      en_in => xlsub1_logical_out1_x8,
      sum_ab => addsub_s_net_x2
    );

  addr2_d6be946d76: entity work.addr1_entity_7a7c1e47fe
    port map (
      ce_1 => ce_1_sg_x8,
      clk_1 => clk_1_sg_x8,
      data_a => mult_p_net_x13,
      data_b => mult_p_net_x14,
      en_in => xlsub1_logical_out1_x8,
      sum_ab => addsub_s_net_x4
    );

  addr3_ef7ed51cec: entity work.addr1_entity_7a7c1e47fe
    port map (
      ce_1 => ce_1_sg_x8,
      clk_1 => clk_1_sg_x8,
      data_a => mult_p_net_x15,
      data_b => mult_p_net_x16,
      en_in => xlsub1_logical_out1_x8,
      sum_ab => addsub_s_net_x5
    );

  addr4_1902a91898: entity work.addr1_entity_7a7c1e47fe
    port map (
      ce_1 => ce_1_sg_x8,
      clk_1 => clk_1_sg_x8,
      data_a => mult_p_net_x10,
      data_b => mult_p_net_x11,
      en_in => xlsub1_logical_out1_x8,
      sum_ab => addsub_s_net_x6
    );

  addr6_6c203a7cff: entity work.addr6_entity_6c203a7cff
    port map (
      ce_1 => ce_1_sg_x8,
      clk_1 => clk_1_sg_x8,
      data_a => addsub_s_net_x2,
      data_b => addsub_s_net_x4,
      en_in => xlsub1_logical_out1_x8,
      sum_ab => addsub_s_net_x7
    );

  addr7_566f767153: entity work.addr6_entity_6c203a7cff
    port map (
      ce_1 => ce_1_sg_x8,
      clk_1 => clk_1_sg_x8,
      data_a => addsub_s_net_x5,
      data_b => addsub_s_net_x6,
      en_in => xlsub1_logical_out1_x8,
      sum_ab => addsub_s_net_x8
    );

  addr9_9ae2508b3a: entity work.addr9_entity_9ae2508b3a
    port map (
      ce_1 => ce_1_sg_x8,
      clk_1 => clk_1_sg_x8,
      data_a => addsub_s_net_x7,
      data_b => addsub_s_net_x8,
      en_in => xlsub1_logical_out1_x8,
      sum_ab => addsub_s_net_x9
    );

  dly10: entity work.delay_d5831e814a
    port map (
      ce => ce_1_sg_x8,
      clk => clk_1_sg_x8,
      clr => '0',
      d => adder_outs2_3,
      q => adder_outs3_2_x0
    );

  dly5: entity work.delay_d5831e814a
    port map (
      ce => ce_1_sg_x8,
      clk => clk_1_sg_x8,
      clr => '0',
      d => mult_p_net_x0,
      q => adder_outs1_5
    );

  dly8: entity work.delay_d5831e814a
    port map (
      ce => ce_1_sg_x8,
      clk => clk_1_sg_x8,
      clr => '0',
      d => adder_outs1_5,
      q => adder_outs2_3
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/en_gen"

entity en_gen_entity_f6c8ec71ed is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    sync_in: in std_logic; 
    en_out: out std_logic
  );
end en_gen_entity_f6c8ec71ed;

architecture structural of en_gen_entity_f6c8ec71ed is
  signal ce_1_sg_x9: std_logic;
  signal clk_1_sg_x9: std_logic;
  signal sync_delay_q_net_x0: std_logic;
  signal xlsub1_constant_out1: std_logic_vector(15 downto 0);
  signal xlsub1_counter1_out1: std_logic;
  signal xlsub1_logical_out1_x9: std_logic;
  signal xlsub1_relational_out1: std_logic;

begin
  ce_1_sg_x9 <= ce_1;
  clk_1_sg_x9 <= clk_1;
  sync_delay_q_net_x0 <= sync_in;
  en_out <= xlsub1_logical_out1_x9;

  constant_x0: entity work.constant_9f5572ba51
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      op => xlsub1_constant_out1
    );

  counter1: entity work.counter_4d9f59591b
    port map (
      ce => ce_1_sg_x9,
      clk => clk_1_sg_x9,
      clr => '0',
      rst(0) => xlsub1_logical_out1_x9,
      op(0) => xlsub1_counter1_out1
    );

  logical: entity work.logical_aacf6e1b0e
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      d0(0) => xlsub1_relational_out1,
      d1(0) => sync_delay_q_net_x0,
      y(0) => xlsub1_logical_out1_x9
    );

  relational: entity work.relational_af078c6141
    port map (
      a(0) => xlsub1_counter1_out1,
      b => xlsub1_constant_out1,
      ce => '0',
      clk => '0',
      clr => '0',
      op(0) => xlsub1_relational_out1
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/hold_fir_tap1"

entity hold_fir_tap1_entity_a0f1aa4837 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    data_in: in std_logic_vector(31 downto 0); 
    en_in: in std_logic; 
    mult_out: out std_logic_vector(47 downto 0)
  );
end hold_fir_tap1_entity_a0f1aa4837;

architecture structural of hold_fir_tap1_entity_a0f1aa4837 is
  signal ce_1_sg_x10: std_logic;
  signal clk_1_sg_x10: std_logic;
  signal convert4_dout_net_x0: std_logic_vector(31 downto 0);
  signal mult_p_net_x10: std_logic_vector(47 downto 0);
  signal xlsub1_logical_out1_x10: std_logic;
  signal xlsub2_coefficient_out1: std_logic_vector(15 downto 0);

begin
  ce_1_sg_x10 <= ce_1;
  clk_1_sg_x10 <= clk_1;
  convert4_dout_net_x0 <= data_in;
  xlsub1_logical_out1_x10 <= en_in;
  mult_out <= mult_p_net_x10;

  coefficient: entity work.constant_37b2f0b7ea
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      op => xlsub2_coefficient_out1
    );

  mult: entity work.xlmult_cichbcic_core
    generic map (
      a_arith => xlSigned,
      a_bin_pt => 14,
      a_width => 16,
      b_arith => xlSigned,
      b_bin_pt => 18,
      b_width => 32,
      c_a_type => 0,
      c_a_width => 16,
      c_b_type => 0,
      c_b_width => 32,
      c_baat => 16,
      c_output_width => 48,
      c_type => 0,
      core_name0 => "mult_11_2_893416810381d560",
      en_arith => xlUnsigned,
      en_bin_pt => 0,
      en_width => 1,
      extra_registers => 0,
      multsign => 2,
      overflow => 1,
      p_arith => xlSigned,
      p_bin_pt => 32,
      p_width => 48,
      quantization => 1
    )
    port map (
      a => xlsub2_coefficient_out1,
      b => convert4_dout_net_x0,
      ce => ce_1_sg_x10,
      clk => clk_1_sg_x10,
      clr => '0',
      core_ce => ce_1_sg_x10,
      core_clk => clk_1_sg_x10,
      core_clr => '1',
      en(0) => xlsub1_logical_out1_x10,
      rst => "0",
      p => mult_p_net_x10
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/hold_fir_tap11"

entity hold_fir_tap11_entity_907c07f1f7 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    data_in: in std_logic_vector(31 downto 0); 
    en_in: in std_logic; 
    mult_out: out std_logic_vector(47 downto 0)
  );
end hold_fir_tap11_entity_907c07f1f7;

architecture structural of hold_fir_tap11_entity_907c07f1f7 is
  signal ce_1_sg_x11: std_logic;
  signal clk_1_sg_x11: std_logic;
  signal mult_p_net_x11: std_logic_vector(47 downto 0);
  signal xlsub1_logical_out1_x11: std_logic;
  signal xlsub2_coefficient_out1: std_logic_vector(15 downto 0);
  signal xlsub2_delay_out10_x0: std_logic_vector(31 downto 0);

begin
  ce_1_sg_x11 <= ce_1;
  clk_1_sg_x11 <= clk_1;
  xlsub2_delay_out10_x0 <= data_in;
  xlsub1_logical_out1_x11 <= en_in;
  mult_out <= mult_p_net_x11;

  coefficient: entity work.constant_d4111c362e
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      op => xlsub2_coefficient_out1
    );

  mult: entity work.xlmult_cichbcic_core
    generic map (
      a_arith => xlSigned,
      a_bin_pt => 14,
      a_width => 16,
      b_arith => xlSigned,
      b_bin_pt => 18,
      b_width => 32,
      c_a_type => 0,
      c_a_width => 16,
      c_b_type => 0,
      c_b_width => 32,
      c_baat => 16,
      c_output_width => 48,
      c_type => 0,
      core_name0 => "mult_11_2_893416810381d560",
      en_arith => xlUnsigned,
      en_bin_pt => 0,
      en_width => 1,
      extra_registers => 0,
      multsign => 2,
      overflow => 1,
      p_arith => xlSigned,
      p_bin_pt => 32,
      p_width => 48,
      quantization => 1
    )
    port map (
      a => xlsub2_coefficient_out1,
      b => xlsub2_delay_out10_x0,
      ce => ce_1_sg_x11,
      clk => clk_1_sg_x11,
      clr => '0',
      core_ce => ce_1_sg_x11,
      core_clk => clk_1_sg_x11,
      core_clr => '1',
      en(0) => xlsub1_logical_out1_x11,
      rst => "0",
      p => mult_p_net_x11
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/hold_fir_tap13"

entity hold_fir_tap13_entity_9a70db5048 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    data_in: in std_logic_vector(31 downto 0); 
    en_in: in std_logic; 
    mult_out: out std_logic_vector(47 downto 0)
  );
end hold_fir_tap13_entity_9a70db5048;

architecture structural of hold_fir_tap13_entity_9a70db5048 is
  signal ce_1_sg_x12: std_logic;
  signal clk_1_sg_x12: std_logic;
  signal mult_p_net_x12: std_logic_vector(47 downto 0);
  signal xlsub1_logical_out1_x12: std_logic;
  signal xlsub2_coefficient_out1: std_logic_vector(15 downto 0);
  signal xlsub2_delay_out12_x0: std_logic_vector(31 downto 0);

begin
  ce_1_sg_x12 <= ce_1;
  clk_1_sg_x12 <= clk_1;
  xlsub2_delay_out12_x0 <= data_in;
  xlsub1_logical_out1_x12 <= en_in;
  mult_out <= mult_p_net_x12;

  coefficient: entity work.constant_b0dd5d0cf3
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      op => xlsub2_coefficient_out1
    );

  mult: entity work.xlmult_cichbcic_core
    generic map (
      a_arith => xlSigned,
      a_bin_pt => 14,
      a_width => 16,
      b_arith => xlSigned,
      b_bin_pt => 18,
      b_width => 32,
      c_a_type => 0,
      c_a_width => 16,
      c_b_type => 0,
      c_b_width => 32,
      c_baat => 16,
      c_output_width => 48,
      c_type => 0,
      core_name0 => "mult_11_2_893416810381d560",
      en_arith => xlUnsigned,
      en_bin_pt => 0,
      en_width => 1,
      extra_registers => 0,
      multsign => 2,
      overflow => 1,
      p_arith => xlSigned,
      p_bin_pt => 32,
      p_width => 48,
      quantization => 1
    )
    port map (
      a => xlsub2_coefficient_out1,
      b => xlsub2_delay_out12_x0,
      ce => ce_1_sg_x12,
      clk => clk_1_sg_x12,
      clr => '0',
      core_ce => ce_1_sg_x12,
      core_clk => clk_1_sg_x12,
      core_clr => '1',
      en(0) => xlsub1_logical_out1_x12,
      rst => "0",
      p => mult_p_net_x12
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/hold_fir_tap7"

entity hold_fir_tap7_entity_3449d19f28 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    data_in: in std_logic_vector(31 downto 0); 
    en_in: in std_logic; 
    mult_out: out std_logic_vector(47 downto 0)
  );
end hold_fir_tap7_entity_3449d19f28;

architecture structural of hold_fir_tap7_entity_3449d19f28 is
  signal ce_1_sg_x16: std_logic;
  signal clk_1_sg_x16: std_logic;
  signal mult_p_net_x15: std_logic_vector(47 downto 0);
  signal xlsub1_logical_out1_x16: std_logic;
  signal xlsub2_coefficient_out1: std_logic_vector(15 downto 0);
  signal xlsub2_delay_out6_x0: std_logic_vector(31 downto 0);

begin
  ce_1_sg_x16 <= ce_1;
  clk_1_sg_x16 <= clk_1;
  xlsub2_delay_out6_x0 <= data_in;
  xlsub1_logical_out1_x16 <= en_in;
  mult_out <= mult_p_net_x15;

  coefficient: entity work.constant_32e8526bad
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      op => xlsub2_coefficient_out1
    );

  mult: entity work.xlmult_cichbcic_core
    generic map (
      a_arith => xlSigned,
      a_bin_pt => 14,
      a_width => 16,
      b_arith => xlSigned,
      b_bin_pt => 18,
      b_width => 32,
      c_a_type => 0,
      c_a_width => 16,
      c_b_type => 0,
      c_b_width => 32,
      c_baat => 16,
      c_output_width => 48,
      c_type => 0,
      core_name0 => "mult_11_2_893416810381d560",
      en_arith => xlUnsigned,
      en_bin_pt => 0,
      en_width => 1,
      extra_registers => 0,
      multsign => 2,
      overflow => 1,
      p_arith => xlSigned,
      p_bin_pt => 32,
      p_width => 48,
      quantization => 1
    )
    port map (
      a => xlsub2_coefficient_out1,
      b => xlsub2_delay_out6_x0,
      ce => ce_1_sg_x16,
      clk => clk_1_sg_x16,
      clr => '0',
      core_ce => ce_1_sg_x16,
      core_clk => clk_1_sg_x16,
      core_clr => '1',
      en(0) => xlsub1_logical_out1_x16,
      rst => "0",
      p => mult_p_net_x15
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1/hold_fir_tap8"

entity hold_fir_tap8_entity_2e0727bf4e is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    data_in: in std_logic_vector(31 downto 0); 
    en_in: in std_logic; 
    mult_out: out std_logic_vector(47 downto 0)
  );
end hold_fir_tap8_entity_2e0727bf4e;

architecture structural of hold_fir_tap8_entity_2e0727bf4e is
  signal ce_1_sg_x17: std_logic;
  signal clk_1_sg_x17: std_logic;
  signal mult_p_net_x16: std_logic_vector(47 downto 0);
  signal xlsub1_logical_out1_x17: std_logic;
  signal xlsub2_coefficient_out1: std_logic_vector(15 downto 0);
  signal xlsub2_delay_out7_x0: std_logic_vector(31 downto 0);

begin
  ce_1_sg_x17 <= ce_1;
  clk_1_sg_x17 <= clk_1;
  xlsub2_delay_out7_x0 <= data_in;
  xlsub1_logical_out1_x17 <= en_in;
  mult_out <= mult_p_net_x16;

  coefficient: entity work.constant_6c8e1bed76
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      op => xlsub2_coefficient_out1
    );

  mult: entity work.xlmult_cichbcic_core
    generic map (
      a_arith => xlSigned,
      a_bin_pt => 14,
      a_width => 16,
      b_arith => xlSigned,
      b_bin_pt => 18,
      b_width => 32,
      c_a_type => 0,
      c_a_width => 16,
      c_b_type => 0,
      c_b_width => 32,
      c_baat => 16,
      c_output_width => 48,
      c_type => 0,
      core_name0 => "mult_11_2_893416810381d560",
      en_arith => xlUnsigned,
      en_bin_pt => 0,
      en_width => 1,
      extra_registers => 0,
      multsign => 2,
      overflow => 1,
      p_arith => xlSigned,
      p_bin_pt => 32,
      p_width => 48,
      quantization => 1
    )
    port map (
      a => xlsub2_coefficient_out1,
      b => xlsub2_delay_out7_x0,
      ce => ce_1_sg_x17,
      clk => clk_1_sg_x17,
      clr => '0',
      core_ce => ce_1_sg_x17,
      core_clk => clk_1_sg_x17,
      core_clr => '1',
      en(0) => xlsub1_logical_out1_x17,
      rst => "0",
      p => mult_p_net_x16
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/Halfband1"

entity halfband1_entity_83ad1d234b is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in_x0: in std_logic_vector(31 downto 0); 
    sync: in std_logic; 
    out_x0: out std_logic_vector(51 downto 0)
  );
end halfband1_entity_83ad1d234b;

architecture structural of halfband1_entity_83ad1d234b is
  signal addsub_s_net_x11: std_logic_vector(51 downto 0);
  signal ce_1_sg_x19: std_logic;
  signal clk_1_sg_x19: std_logic;
  signal convert4_dout_net_x1: std_logic_vector(31 downto 0);
  signal mult_p_net_x1: std_logic_vector(47 downto 0);
  signal mult_p_net_x10: std_logic_vector(47 downto 0);
  signal mult_p_net_x11: std_logic_vector(47 downto 0);
  signal mult_p_net_x12: std_logic_vector(47 downto 0);
  signal mult_p_net_x13: std_logic_vector(47 downto 0);
  signal mult_p_net_x14: std_logic_vector(47 downto 0);
  signal mult_p_net_x15: std_logic_vector(47 downto 0);
  signal mult_p_net_x16: std_logic_vector(47 downto 0);
  signal mult_p_net_x17: std_logic_vector(47 downto 0);
  signal sync_delay_q_net_x1: std_logic;
  signal xlsub1_logical_out1_x18: std_logic;
  signal xlsub2_delay_out1: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out10_x0: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out11: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out12_x0: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out13: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out14_x0: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out2_x0: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out3: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out4_x0: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out5: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out6_x0: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out7_x0: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out8_x0: std_logic_vector(31 downto 0);
  signal xlsub2_delay_out9: std_logic_vector(31 downto 0);

begin
  ce_1_sg_x19 <= ce_1;
  clk_1_sg_x19 <= clk_1;
  convert4_dout_net_x1 <= in_x0;
  sync_delay_q_net_x1 <= sync;
  out_x0 <= addsub_s_net_x11;

  adder_tree_418f80e00e: entity work.adder_tree_entity_418f80e00e
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      din1 => mult_p_net_x10,
      din2 => mult_p_net_x13,
      din3 => mult_p_net_x14,
      din4 => mult_p_net_x15,
      din5 => mult_p_net_x16,
      din6 => mult_p_net_x17,
      din7 => mult_p_net_x11,
      din8 => mult_p_net_x12,
      din9 => mult_p_net_x1,
      en_in => xlsub1_logical_out1_x18,
      dout => addsub_s_net_x11
    );

  delay1: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => convert4_dout_net_x1,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out1
    );

  delay10: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out9,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out10_x0
    );

  delay11: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out10_x0,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out11
    );

  delay12: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out11,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out12_x0
    );

  delay13: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out12_x0,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out13
    );

  delay14: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out13,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out14_x0
    );

  delay2: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out1,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out2_x0
    );

  delay3: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out2_x0,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out3
    );

  delay4: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out3,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out4_x0
    );

  delay5: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out4_x0,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out5
    );

  delay6: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out5,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out6_x0
    );

  delay7: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out6_x0,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out7_x0
    );

  delay8: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out7_x0,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out8_x0
    );

  delay9: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x19,
      clk => clk_1_sg_x19,
      d => xlsub2_delay_out8_x0,
      en => '1',
      rst => '1',
      q => xlsub2_delay_out9
    );

  en_gen_f6c8ec71ed: entity work.en_gen_entity_f6c8ec71ed
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      sync_in => sync_delay_q_net_x1,
      en_out => xlsub1_logical_out1_x18
    );

  hold_fir_tap11_907c07f1f7: entity work.hold_fir_tap11_entity_907c07f1f7
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      data_in => xlsub2_delay_out10_x0,
      en_in => xlsub1_logical_out1_x18,
      mult_out => mult_p_net_x11
    );

  hold_fir_tap13_9a70db5048: entity work.hold_fir_tap13_entity_9a70db5048
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      data_in => xlsub2_delay_out12_x0,
      en_in => xlsub1_logical_out1_x18,
      mult_out => mult_p_net_x12
    );

  hold_fir_tap15_066783c214: entity work.hold_fir_tap1_entity_a0f1aa4837
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      data_in => xlsub2_delay_out14_x0,
      en_in => xlsub1_logical_out1_x18,
      mult_out => mult_p_net_x1
    );

  hold_fir_tap1_a0f1aa4837: entity work.hold_fir_tap1_entity_a0f1aa4837
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      data_in => convert4_dout_net_x1,
      en_in => xlsub1_logical_out1_x18,
      mult_out => mult_p_net_x10
    );

  hold_fir_tap3_6ea4b8a3b4: entity work.hold_fir_tap13_entity_9a70db5048
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      data_in => xlsub2_delay_out2_x0,
      en_in => xlsub1_logical_out1_x18,
      mult_out => mult_p_net_x13
    );

  hold_fir_tap5_451375555c: entity work.hold_fir_tap11_entity_907c07f1f7
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      data_in => xlsub2_delay_out4_x0,
      en_in => xlsub1_logical_out1_x18,
      mult_out => mult_p_net_x14
    );

  hold_fir_tap7_3449d19f28: entity work.hold_fir_tap7_entity_3449d19f28
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      data_in => xlsub2_delay_out6_x0,
      en_in => xlsub1_logical_out1_x18,
      mult_out => mult_p_net_x15
    );

  hold_fir_tap8_2e0727bf4e: entity work.hold_fir_tap8_entity_2e0727bf4e
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      data_in => xlsub2_delay_out7_x0,
      en_in => xlsub1_logical_out1_x18,
      mult_out => mult_p_net_x16
    );

  hold_fir_tap9_f283c53eb8: entity work.hold_fir_tap7_entity_3449d19f28
    port map (
      ce_1 => ce_1_sg_x19,
      clk_1 => clk_1_sg_x19,
      data_in => xlsub2_delay_out8_x0,
      en_in => xlsub1_logical_out1_x18,
      mult_out => mult_p_net_x17
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage1_dec2/p_adder1/adder_tree1"

entity adder_tree1_entity_d35da23fe2 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(23 downto 0); 
    din_2: in std_logic_vector(23 downto 0); 
    dout: out std_logic_vector(24 downto 0)
  );
end adder_tree1_entity_d35da23fe2;

architecture structural of adder_tree1_entity_d35da23fe2 is
  signal adder_0_s_net_x0: std_logic_vector(24 downto 0);
  signal ce_1_sg_x20: std_logic;
  signal clk_1_sg_x20: std_logic;
  signal d16_x0: std_logic_vector(23 downto 0);
  signal in0_net_x0: std_logic_vector(23 downto 0);

begin
  ce_1_sg_x20 <= ce_1;
  clk_1_sg_x20 <= clk_1;
  in0_net_x0 <= din_1;
  d16_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_6eb5f4740f
    port map (
      a => d16_x0,
      b => in0_net_x0,
      ce => ce_1_sg_x20,
      clk => clk_1_sg_x20,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage1_dec2/p_adder1"

entity p_adder1_entity_7661d8894d is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(23 downto 0); 
    in10: in std_logic_vector(23 downto 0); 
    in11: in std_logic_vector(23 downto 0); 
    in12: in std_logic_vector(23 downto 0); 
    in13: in std_logic_vector(23 downto 0); 
    in14: in std_logic_vector(23 downto 0); 
    in15: in std_logic_vector(23 downto 0); 
    in16: in std_logic_vector(23 downto 0); 
    in2: in std_logic_vector(23 downto 0); 
    in3: in std_logic_vector(23 downto 0); 
    in4: in std_logic_vector(23 downto 0); 
    in5: in std_logic_vector(23 downto 0); 
    in6: in std_logic_vector(23 downto 0); 
    in7: in std_logic_vector(23 downto 0); 
    in8: in std_logic_vector(23 downto 0); 
    in9: in std_logic_vector(23 downto 0); 
    out1: out std_logic_vector(24 downto 0); 
    out10: out std_logic_vector(24 downto 0); 
    out11: out std_logic_vector(24 downto 0); 
    out12: out std_logic_vector(24 downto 0); 
    out13: out std_logic_vector(24 downto 0); 
    out14: out std_logic_vector(24 downto 0); 
    out15: out std_logic_vector(24 downto 0); 
    out16: out std_logic_vector(24 downto 0); 
    out2: out std_logic_vector(24 downto 0); 
    out3: out std_logic_vector(24 downto 0); 
    out4: out std_logic_vector(24 downto 0); 
    out5: out std_logic_vector(24 downto 0); 
    out6: out std_logic_vector(24 downto 0); 
    out7: out std_logic_vector(24 downto 0); 
    out8: out std_logic_vector(24 downto 0); 
    out9: out std_logic_vector(24 downto 0)
  );
end p_adder1_entity_7661d8894d;

architecture structural of p_adder1_entity_7661d8894d is
  signal adder_0_s_net_x16: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x17: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x18: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x19: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x20: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x21: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x22: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x23: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x24: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x25: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x26: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x27: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x28: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x29: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x30: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x31: std_logic_vector(24 downto 0);
  signal ce_1_sg_x36: std_logic;
  signal clk_1_sg_x36: std_logic;
  signal d16_x0: std_logic_vector(23 downto 0);
  signal in0_net_x2: std_logic_vector(23 downto 0);
  signal in10_net_x2: std_logic_vector(23 downto 0);
  signal in11_net_x2: std_logic_vector(23 downto 0);
  signal in12_net_x2: std_logic_vector(23 downto 0);
  signal in13_net_x2: std_logic_vector(23 downto 0);
  signal in14_net_x2: std_logic_vector(23 downto 0);
  signal in15_net_x1: std_logic_vector(23 downto 0);
  signal in1_net_x2: std_logic_vector(23 downto 0);
  signal in2_net_x2: std_logic_vector(23 downto 0);
  signal in3_net_x2: std_logic_vector(23 downto 0);
  signal in4_net_x2: std_logic_vector(23 downto 0);
  signal in5_net_x2: std_logic_vector(23 downto 0);
  signal in6_net_x2: std_logic_vector(23 downto 0);
  signal in7_net_x2: std_logic_vector(23 downto 0);
  signal in8_net_x2: std_logic_vector(23 downto 0);
  signal in9_net_x2: std_logic_vector(23 downto 0);

begin
  ce_1_sg_x36 <= ce_1;
  clk_1_sg_x36 <= clk_1;
  in0_net_x2 <= in1;
  in9_net_x2 <= in10;
  in10_net_x2 <= in11;
  in11_net_x2 <= in12;
  in12_net_x2 <= in13;
  in13_net_x2 <= in14;
  in14_net_x2 <= in15;
  in15_net_x1 <= in16;
  in1_net_x2 <= in2;
  in2_net_x2 <= in3;
  in3_net_x2 <= in4;
  in4_net_x2 <= in5;
  in5_net_x2 <= in6;
  in6_net_x2 <= in7;
  in7_net_x2 <= in8;
  in8_net_x2 <= in9;
  out1 <= adder_0_s_net_x16;
  out10 <= adder_0_s_net_x17;
  out11 <= adder_0_s_net_x18;
  out12 <= adder_0_s_net_x19;
  out13 <= adder_0_s_net_x20;
  out14 <= adder_0_s_net_x21;
  out15 <= adder_0_s_net_x22;
  out16 <= adder_0_s_net_x23;
  out2 <= adder_0_s_net_x24;
  out3 <= adder_0_s_net_x25;
  out4 <= adder_0_s_net_x26;
  out5 <= adder_0_s_net_x27;
  out6 <= adder_0_s_net_x28;
  out7 <= adder_0_s_net_x29;
  out8 <= adder_0_s_net_x30;
  out9 <= adder_0_s_net_x31;

  adder_tree10_b9fe764fa5: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in9_net_x2,
      din_2 => in8_net_x2,
      dout => adder_0_s_net_x17
    );

  adder_tree11_3a456232ee: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in10_net_x2,
      din_2 => in9_net_x2,
      dout => adder_0_s_net_x18
    );

  adder_tree12_aeb18274f6: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in11_net_x2,
      din_2 => in10_net_x2,
      dout => adder_0_s_net_x19
    );

  adder_tree13_083fe0b0e6: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in12_net_x2,
      din_2 => in11_net_x2,
      dout => adder_0_s_net_x20
    );

  adder_tree14_32b7da6bba: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in13_net_x2,
      din_2 => in12_net_x2,
      dout => adder_0_s_net_x21
    );

  adder_tree15_8ec7f672ff: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in14_net_x2,
      din_2 => in13_net_x2,
      dout => adder_0_s_net_x22
    );

  adder_tree16_eaecd7aacd: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in15_net_x1,
      din_2 => in14_net_x2,
      dout => adder_0_s_net_x23
    );

  adder_tree1_d35da23fe2: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in0_net_x2,
      din_2 => d16_x0,
      dout => adder_0_s_net_x16
    );

  adder_tree2_b5887d0a69: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in1_net_x2,
      din_2 => in0_net_x2,
      dout => adder_0_s_net_x24
    );

  adder_tree3_f615e36a15: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in2_net_x2,
      din_2 => in1_net_x2,
      dout => adder_0_s_net_x25
    );

  adder_tree4_71aee21cd2: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in3_net_x2,
      din_2 => in2_net_x2,
      dout => adder_0_s_net_x26
    );

  adder_tree5_c0ca6e8436: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in4_net_x2,
      din_2 => in3_net_x2,
      dout => adder_0_s_net_x27
    );

  adder_tree6_593693c3d6: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in5_net_x2,
      din_2 => in4_net_x2,
      dout => adder_0_s_net_x28
    );

  adder_tree7_66290f969b: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in6_net_x2,
      din_2 => in5_net_x2,
      dout => adder_0_s_net_x29
    );

  adder_tree8_544a037d9e: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in7_net_x2,
      din_2 => in6_net_x2,
      dout => adder_0_s_net_x30
    );

  adder_tree9_3b3c4c377d: entity work.adder_tree1_entity_d35da23fe2
    port map (
      ce_1 => ce_1_sg_x36,
      clk_1 => clk_1_sg_x36,
      din_1 => in8_net_x2,
      din_2 => in7_net_x2,
      dout => adder_0_s_net_x31
    );

  delay16: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 24
    )
    port map (
      ce => ce_1_sg_x36,
      clk => clk_1_sg_x36,
      d => in15_net_x1,
      en => '1',
      rst => '1',
      q => d16_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage1_dec2/p_adder2/adder_tree1"

entity adder_tree1_entity_e78d1e76b6 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(24 downto 0); 
    din_2: in std_logic_vector(24 downto 0); 
    dout: out std_logic_vector(25 downto 0)
  );
end adder_tree1_entity_e78d1e76b6;

architecture structural of adder_tree1_entity_e78d1e76b6 is
  signal adder_0_s_net_x0: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x17: std_logic_vector(24 downto 0);
  signal ce_1_sg_x37: std_logic;
  signal clk_1_sg_x37: std_logic;
  signal d16_x0: std_logic_vector(24 downto 0);

begin
  ce_1_sg_x37 <= ce_1;
  clk_1_sg_x37 <= clk_1;
  adder_0_s_net_x17 <= din_1;
  d16_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_b4fd6cc060
    port map (
      a => d16_x0,
      b => adder_0_s_net_x17,
      ce => ce_1_sg_x37,
      clk => clk_1_sg_x37,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage1_dec2/p_adder2"

entity p_adder2_entity_252974f8c2 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(24 downto 0); 
    in10: in std_logic_vector(24 downto 0); 
    in11: in std_logic_vector(24 downto 0); 
    in12: in std_logic_vector(24 downto 0); 
    in13: in std_logic_vector(24 downto 0); 
    in14: in std_logic_vector(24 downto 0); 
    in15: in std_logic_vector(24 downto 0); 
    in16: in std_logic_vector(24 downto 0); 
    in2: in std_logic_vector(24 downto 0); 
    in3: in std_logic_vector(24 downto 0); 
    in4: in std_logic_vector(24 downto 0); 
    in5: in std_logic_vector(24 downto 0); 
    in6: in std_logic_vector(24 downto 0); 
    in7: in std_logic_vector(24 downto 0); 
    in8: in std_logic_vector(24 downto 0); 
    in9: in std_logic_vector(24 downto 0); 
    out1: out std_logic_vector(25 downto 0); 
    out10: out std_logic_vector(25 downto 0); 
    out11: out std_logic_vector(25 downto 0); 
    out12: out std_logic_vector(25 downto 0); 
    out13: out std_logic_vector(25 downto 0); 
    out14: out std_logic_vector(25 downto 0); 
    out15: out std_logic_vector(25 downto 0); 
    out16: out std_logic_vector(25 downto 0); 
    out2: out std_logic_vector(25 downto 0); 
    out3: out std_logic_vector(25 downto 0); 
    out4: out std_logic_vector(25 downto 0); 
    out5: out std_logic_vector(25 downto 0); 
    out6: out std_logic_vector(25 downto 0); 
    out7: out std_logic_vector(25 downto 0); 
    out8: out std_logic_vector(25 downto 0); 
    out9: out std_logic_vector(25 downto 0)
  );
end p_adder2_entity_252974f8c2;

architecture structural of p_adder2_entity_252974f8c2 is
  signal adder_0_s_net_x53: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x54: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x55: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x56: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x57: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x58: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x59: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x60: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x61: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x62: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x63: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x64: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x65: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x66: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x67: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x68: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x69: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x70: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x71: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x72: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x73: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x74: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x75: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x76: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x77: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x78: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x79: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x80: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x81: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x82: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x83: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x84: std_logic_vector(25 downto 0);
  signal ce_1_sg_x53: std_logic;
  signal clk_1_sg_x53: std_logic;
  signal d16_x0: std_logic_vector(24 downto 0);

begin
  ce_1_sg_x53 <= ce_1;
  clk_1_sg_x53 <= clk_1;
  adder_0_s_net_x53 <= in1;
  adder_0_s_net_x54 <= in10;
  adder_0_s_net_x55 <= in11;
  adder_0_s_net_x56 <= in12;
  adder_0_s_net_x57 <= in13;
  adder_0_s_net_x58 <= in14;
  adder_0_s_net_x59 <= in15;
  adder_0_s_net_x60 <= in16;
  adder_0_s_net_x61 <= in2;
  adder_0_s_net_x62 <= in3;
  adder_0_s_net_x63 <= in4;
  adder_0_s_net_x64 <= in5;
  adder_0_s_net_x65 <= in6;
  adder_0_s_net_x66 <= in7;
  adder_0_s_net_x67 <= in8;
  adder_0_s_net_x68 <= in9;
  out1 <= adder_0_s_net_x69;
  out10 <= adder_0_s_net_x70;
  out11 <= adder_0_s_net_x71;
  out12 <= adder_0_s_net_x72;
  out13 <= adder_0_s_net_x73;
  out14 <= adder_0_s_net_x74;
  out15 <= adder_0_s_net_x75;
  out16 <= adder_0_s_net_x76;
  out2 <= adder_0_s_net_x77;
  out3 <= adder_0_s_net_x78;
  out4 <= adder_0_s_net_x79;
  out5 <= adder_0_s_net_x80;
  out6 <= adder_0_s_net_x81;
  out7 <= adder_0_s_net_x82;
  out8 <= adder_0_s_net_x83;
  out9 <= adder_0_s_net_x84;

  adder_tree10_967948ff0c: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x54,
      din_2 => adder_0_s_net_x68,
      dout => adder_0_s_net_x70
    );

  adder_tree11_f89f10e45c: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x55,
      din_2 => adder_0_s_net_x54,
      dout => adder_0_s_net_x71
    );

  adder_tree12_fb6ac476f7: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x56,
      din_2 => adder_0_s_net_x55,
      dout => adder_0_s_net_x72
    );

  adder_tree13_5a14e6a5bd: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x57,
      din_2 => adder_0_s_net_x56,
      dout => adder_0_s_net_x73
    );

  adder_tree14_c78a02fd93: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x58,
      din_2 => adder_0_s_net_x57,
      dout => adder_0_s_net_x74
    );

  adder_tree15_89ad06e1d9: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x59,
      din_2 => adder_0_s_net_x58,
      dout => adder_0_s_net_x75
    );

  adder_tree16_d6669a3049: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x60,
      din_2 => adder_0_s_net_x59,
      dout => adder_0_s_net_x76
    );

  adder_tree1_e78d1e76b6: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x53,
      din_2 => d16_x0,
      dout => adder_0_s_net_x69
    );

  adder_tree2_72e2a15525: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x61,
      din_2 => adder_0_s_net_x53,
      dout => adder_0_s_net_x77
    );

  adder_tree3_2fa18602ae: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x62,
      din_2 => adder_0_s_net_x61,
      dout => adder_0_s_net_x78
    );

  adder_tree4_ff6cd1ed21: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x63,
      din_2 => adder_0_s_net_x62,
      dout => adder_0_s_net_x79
    );

  adder_tree5_702d090325: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x64,
      din_2 => adder_0_s_net_x63,
      dout => adder_0_s_net_x80
    );

  adder_tree6_a38ce31f3d: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x65,
      din_2 => adder_0_s_net_x64,
      dout => adder_0_s_net_x81
    );

  adder_tree7_2fd7efa680: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x66,
      din_2 => adder_0_s_net_x65,
      dout => adder_0_s_net_x82
    );

  adder_tree8_8abdebecb8: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x67,
      din_2 => adder_0_s_net_x66,
      dout => adder_0_s_net_x83
    );

  adder_tree9_dd42dc42c1: entity work.adder_tree1_entity_e78d1e76b6
    port map (
      ce_1 => ce_1_sg_x53,
      clk_1 => clk_1_sg_x53,
      din_1 => adder_0_s_net_x68,
      din_2 => adder_0_s_net_x67,
      dout => adder_0_s_net_x84
    );

  delay16: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 25
    )
    port map (
      ce => ce_1_sg_x53,
      clk => clk_1_sg_x53,
      d => adder_0_s_net_x60,
      en => '1',
      rst => '1',
      q => d16_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage1_dec2/p_adder3/adder_tree1"

entity adder_tree1_entity_f971f51b01 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(25 downto 0); 
    din_2: in std_logic_vector(25 downto 0); 
    dout: out std_logic_vector(26 downto 0)
  );
end adder_tree1_entity_f971f51b01;

architecture structural of adder_tree1_entity_f971f51b01 is
  signal adder_0_s_net_x0: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x70: std_logic_vector(25 downto 0);
  signal ce_1_sg_x54: std_logic;
  signal clk_1_sg_x54: std_logic;
  signal d16_x0: std_logic_vector(25 downto 0);

begin
  ce_1_sg_x54 <= ce_1;
  clk_1_sg_x54 <= clk_1;
  adder_0_s_net_x70 <= din_1;
  d16_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_f6988cbd01
    port map (
      a => d16_x0,
      b => adder_0_s_net_x70,
      ce => ce_1_sg_x54,
      clk => clk_1_sg_x54,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage1_dec2/p_adder3"

entity p_adder3_entity_cdc32c2ca9 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(25 downto 0); 
    in10: in std_logic_vector(25 downto 0); 
    in11: in std_logic_vector(25 downto 0); 
    in12: in std_logic_vector(25 downto 0); 
    in13: in std_logic_vector(25 downto 0); 
    in14: in std_logic_vector(25 downto 0); 
    in15: in std_logic_vector(25 downto 0); 
    in16: in std_logic_vector(25 downto 0); 
    in2: in std_logic_vector(25 downto 0); 
    in3: in std_logic_vector(25 downto 0); 
    in4: in std_logic_vector(25 downto 0); 
    in5: in std_logic_vector(25 downto 0); 
    in6: in std_logic_vector(25 downto 0); 
    in7: in std_logic_vector(25 downto 0); 
    in8: in std_logic_vector(25 downto 0); 
    in9: in std_logic_vector(25 downto 0); 
    out1: out std_logic_vector(26 downto 0); 
    out10: out std_logic_vector(26 downto 0); 
    out11: out std_logic_vector(26 downto 0); 
    out12: out std_logic_vector(26 downto 0); 
    out13: out std_logic_vector(26 downto 0); 
    out14: out std_logic_vector(26 downto 0); 
    out15: out std_logic_vector(26 downto 0); 
    out16: out std_logic_vector(26 downto 0); 
    out2: out std_logic_vector(26 downto 0); 
    out3: out std_logic_vector(26 downto 0); 
    out4: out std_logic_vector(26 downto 0); 
    out5: out std_logic_vector(26 downto 0); 
    out6: out std_logic_vector(26 downto 0); 
    out7: out std_logic_vector(26 downto 0); 
    out8: out std_logic_vector(26 downto 0); 
    out9: out std_logic_vector(26 downto 0)
  );
end p_adder3_entity_cdc32c2ca9;

architecture structural of p_adder3_entity_cdc32c2ca9 is
  signal adder_0_s_net_x100: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x101: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x102: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x103: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x104: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x105: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x106: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x107: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x108: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x109: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x16: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x17: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x18: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x19: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x20: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x21: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x22: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x23: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x24: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x25: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x26: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x27: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x28: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x91: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x92: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x93: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x94: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x95: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x96: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x97: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x98: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x99: std_logic_vector(25 downto 0);
  signal ce_1_sg_x70: std_logic;
  signal clk_1_sg_x70: std_logic;
  signal d16_x0: std_logic_vector(25 downto 0);

begin
  ce_1_sg_x70 <= ce_1;
  clk_1_sg_x70 <= clk_1;
  adder_0_s_net_x91 <= in1;
  adder_0_s_net_x92 <= in10;
  adder_0_s_net_x93 <= in11;
  adder_0_s_net_x94 <= in12;
  adder_0_s_net_x95 <= in13;
  adder_0_s_net_x96 <= in14;
  adder_0_s_net_x97 <= in15;
  adder_0_s_net_x98 <= in16;
  adder_0_s_net_x99 <= in2;
  adder_0_s_net_x100 <= in3;
  adder_0_s_net_x101 <= in4;
  adder_0_s_net_x102 <= in5;
  adder_0_s_net_x103 <= in6;
  adder_0_s_net_x104 <= in7;
  adder_0_s_net_x105 <= in8;
  adder_0_s_net_x106 <= in9;
  out1 <= adder_0_s_net_x16;
  out10 <= adder_0_s_net_x17;
  out11 <= adder_0_s_net_x18;
  out12 <= adder_0_s_net_x19;
  out13 <= adder_0_s_net_x20;
  out14 <= adder_0_s_net_x21;
  out15 <= adder_0_s_net_x22;
  out16 <= adder_0_s_net_x23;
  out2 <= adder_0_s_net_x24;
  out3 <= adder_0_s_net_x25;
  out4 <= adder_0_s_net_x26;
  out5 <= adder_0_s_net_x107;
  out6 <= adder_0_s_net_x27;
  out7 <= adder_0_s_net_x108;
  out8 <= adder_0_s_net_x28;
  out9 <= adder_0_s_net_x109;

  adder_tree10_f1bf6e90a0: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x92,
      din_2 => adder_0_s_net_x106,
      dout => adder_0_s_net_x17
    );

  adder_tree11_8b0919cb77: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x93,
      din_2 => adder_0_s_net_x92,
      dout => adder_0_s_net_x18
    );

  adder_tree12_6eb5e76407: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x94,
      din_2 => adder_0_s_net_x93,
      dout => adder_0_s_net_x19
    );

  adder_tree13_2f0caafb4b: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x95,
      din_2 => adder_0_s_net_x94,
      dout => adder_0_s_net_x20
    );

  adder_tree14_440b0d96c8: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x96,
      din_2 => adder_0_s_net_x95,
      dout => adder_0_s_net_x21
    );

  adder_tree15_36dc6b3d62: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x97,
      din_2 => adder_0_s_net_x96,
      dout => adder_0_s_net_x22
    );

  adder_tree16_0bf81da524: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x98,
      din_2 => adder_0_s_net_x97,
      dout => adder_0_s_net_x23
    );

  adder_tree1_f971f51b01: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x91,
      din_2 => d16_x0,
      dout => adder_0_s_net_x16
    );

  adder_tree2_9a7f690c87: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x99,
      din_2 => adder_0_s_net_x91,
      dout => adder_0_s_net_x24
    );

  adder_tree3_b9508e8adf: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x100,
      din_2 => adder_0_s_net_x99,
      dout => adder_0_s_net_x25
    );

  adder_tree4_986da3c7d5: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x101,
      din_2 => adder_0_s_net_x100,
      dout => adder_0_s_net_x26
    );

  adder_tree5_8f54c91d17: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x102,
      din_2 => adder_0_s_net_x101,
      dout => adder_0_s_net_x107
    );

  adder_tree6_d8d8e75d9f: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x103,
      din_2 => adder_0_s_net_x102,
      dout => adder_0_s_net_x27
    );

  adder_tree7_5e30398bc2: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x104,
      din_2 => adder_0_s_net_x103,
      dout => adder_0_s_net_x108
    );

  adder_tree8_bec60f1354: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x105,
      din_2 => adder_0_s_net_x104,
      dout => adder_0_s_net_x28
    );

  adder_tree9_3d142897fb: entity work.adder_tree1_entity_f971f51b01
    port map (
      ce_1 => ce_1_sg_x70,
      clk_1 => clk_1_sg_x70,
      din_1 => adder_0_s_net_x106,
      din_2 => adder_0_s_net_x105,
      dout => adder_0_s_net_x109
    );

  delay16: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 26
    )
    port map (
      ce => ce_1_sg_x70,
      clk => clk_1_sg_x70,
      d => adder_0_s_net_x98,
      en => '1',
      rst => '1',
      q => d16_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage1_dec2/p_adder4/adder_tree10"

entity adder_tree10_entity_eeb237a269 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(26 downto 0); 
    din_2: in std_logic_vector(26 downto 0); 
    dout: out std_logic_vector(27 downto 0)
  );
end adder_tree10_entity_eeb237a269;

architecture structural of adder_tree10_entity_eeb237a269 is
  signal adder_0_s_net_x0: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x110: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x18: std_logic_vector(26 downto 0);
  signal ce_1_sg_x71: std_logic;
  signal clk_1_sg_x71: std_logic;

begin
  ce_1_sg_x71 <= ce_1;
  clk_1_sg_x71 <= clk_1;
  adder_0_s_net_x18 <= din_1;
  adder_0_s_net_x110 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_7d1974bd4f
    port map (
      a => adder_0_s_net_x110,
      b => adder_0_s_net_x18,
      ce => ce_1_sg_x71,
      clk => clk_1_sg_x71,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage1_dec2/p_adder4"

entity p_adder4_entity_9312cee7d8 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(26 downto 0); 
    in10: in std_logic_vector(26 downto 0); 
    in11: in std_logic_vector(26 downto 0); 
    in12: in std_logic_vector(26 downto 0); 
    in13: in std_logic_vector(26 downto 0); 
    in14: in std_logic_vector(26 downto 0); 
    in15: in std_logic_vector(26 downto 0); 
    in16: in std_logic_vector(26 downto 0); 
    in2: in std_logic_vector(26 downto 0); 
    in3: in std_logic_vector(26 downto 0); 
    in4: in std_logic_vector(26 downto 0); 
    in5: in std_logic_vector(26 downto 0); 
    in6: in std_logic_vector(26 downto 0); 
    in7: in std_logic_vector(26 downto 0); 
    in8: in std_logic_vector(26 downto 0); 
    in9: in std_logic_vector(26 downto 0); 
    out10: out std_logic_vector(27 downto 0); 
    out12: out std_logic_vector(27 downto 0); 
    out14: out std_logic_vector(27 downto 0); 
    out16: out std_logic_vector(27 downto 0); 
    out2: out std_logic_vector(27 downto 0); 
    out4: out std_logic_vector(27 downto 0); 
    out6: out std_logic_vector(27 downto 0); 
    out8: out std_logic_vector(27 downto 0)
  );
end p_adder4_entity_9312cee7d8;

architecture structural of p_adder4_entity_9312cee7d8 is
  signal adder_0_s_net_x111: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x112: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x113: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x31: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x32: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x33: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x34: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x35: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x36: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x37: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x38: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x39: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x40: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x41: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x42: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x43: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x44: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x45: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x46: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x47: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x48: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x49: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x50: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x51: std_logic_vector(27 downto 0);
  signal ce_1_sg_x79: std_logic;
  signal clk_1_sg_x79: std_logic;

begin
  ce_1_sg_x79 <= ce_1;
  clk_1_sg_x79 <= clk_1;
  adder_0_s_net_x31 <= in1;
  adder_0_s_net_x32 <= in10;
  adder_0_s_net_x33 <= in11;
  adder_0_s_net_x34 <= in12;
  adder_0_s_net_x35 <= in13;
  adder_0_s_net_x36 <= in14;
  adder_0_s_net_x37 <= in15;
  adder_0_s_net_x38 <= in16;
  adder_0_s_net_x39 <= in2;
  adder_0_s_net_x40 <= in3;
  adder_0_s_net_x41 <= in4;
  adder_0_s_net_x111 <= in5;
  adder_0_s_net_x42 <= in6;
  adder_0_s_net_x112 <= in7;
  adder_0_s_net_x43 <= in8;
  adder_0_s_net_x113 <= in9;
  out10 <= adder_0_s_net_x44;
  out12 <= adder_0_s_net_x45;
  out14 <= adder_0_s_net_x46;
  out16 <= adder_0_s_net_x47;
  out2 <= adder_0_s_net_x48;
  out4 <= adder_0_s_net_x49;
  out6 <= adder_0_s_net_x50;
  out8 <= adder_0_s_net_x51;

  adder_tree10_eeb237a269: entity work.adder_tree10_entity_eeb237a269
    port map (
      ce_1 => ce_1_sg_x79,
      clk_1 => clk_1_sg_x79,
      din_1 => adder_0_s_net_x32,
      din_2 => adder_0_s_net_x113,
      dout => adder_0_s_net_x44
    );

  adder_tree12_e94a681297: entity work.adder_tree10_entity_eeb237a269
    port map (
      ce_1 => ce_1_sg_x79,
      clk_1 => clk_1_sg_x79,
      din_1 => adder_0_s_net_x34,
      din_2 => adder_0_s_net_x33,
      dout => adder_0_s_net_x45
    );

  adder_tree14_11f616223c: entity work.adder_tree10_entity_eeb237a269
    port map (
      ce_1 => ce_1_sg_x79,
      clk_1 => clk_1_sg_x79,
      din_1 => adder_0_s_net_x36,
      din_2 => adder_0_s_net_x35,
      dout => adder_0_s_net_x46
    );

  adder_tree16_63a9406dec: entity work.adder_tree10_entity_eeb237a269
    port map (
      ce_1 => ce_1_sg_x79,
      clk_1 => clk_1_sg_x79,
      din_1 => adder_0_s_net_x38,
      din_2 => adder_0_s_net_x37,
      dout => adder_0_s_net_x47
    );

  adder_tree2_b7f37c926d: entity work.adder_tree10_entity_eeb237a269
    port map (
      ce_1 => ce_1_sg_x79,
      clk_1 => clk_1_sg_x79,
      din_1 => adder_0_s_net_x39,
      din_2 => adder_0_s_net_x31,
      dout => adder_0_s_net_x48
    );

  adder_tree4_a2e9d1ea3c: entity work.adder_tree10_entity_eeb237a269
    port map (
      ce_1 => ce_1_sg_x79,
      clk_1 => clk_1_sg_x79,
      din_1 => adder_0_s_net_x41,
      din_2 => adder_0_s_net_x40,
      dout => adder_0_s_net_x49
    );

  adder_tree6_1970ed789d: entity work.adder_tree10_entity_eeb237a269
    port map (
      ce_1 => ce_1_sg_x79,
      clk_1 => clk_1_sg_x79,
      din_1 => adder_0_s_net_x42,
      din_2 => adder_0_s_net_x111,
      dout => adder_0_s_net_x50
    );

  adder_tree8_a328fd35b7: entity work.adder_tree10_entity_eeb237a269
    port map (
      ce_1 => ce_1_sg_x79,
      clk_1 => clk_1_sg_x79,
      din_1 => adder_0_s_net_x43,
      din_2 => adder_0_s_net_x112,
      dout => adder_0_s_net_x51
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage1_dec2"

entity stage1_dec2_entity_3e8ab687de is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(23 downto 0); 
    in10: in std_logic_vector(23 downto 0); 
    in11: in std_logic_vector(23 downto 0); 
    in12: in std_logic_vector(23 downto 0); 
    in13: in std_logic_vector(23 downto 0); 
    in14: in std_logic_vector(23 downto 0); 
    in15: in std_logic_vector(23 downto 0); 
    in16: in std_logic_vector(23 downto 0); 
    in2: in std_logic_vector(23 downto 0); 
    in3: in std_logic_vector(23 downto 0); 
    in4: in std_logic_vector(23 downto 0); 
    in5: in std_logic_vector(23 downto 0); 
    in6: in std_logic_vector(23 downto 0); 
    in7: in std_logic_vector(23 downto 0); 
    in8: in std_logic_vector(23 downto 0); 
    in9: in std_logic_vector(23 downto 0); 
    out1: out std_logic_vector(27 downto 0); 
    out2: out std_logic_vector(27 downto 0); 
    out3: out std_logic_vector(27 downto 0); 
    out4: out std_logic_vector(27 downto 0); 
    out5: out std_logic_vector(27 downto 0); 
    out6: out std_logic_vector(27 downto 0); 
    out7: out std_logic_vector(27 downto 0); 
    out8: out std_logic_vector(27 downto 0)
  );
end stage1_dec2_entity_3e8ab687de;

architecture structural of stage1_dec2_entity_3e8ab687de is
  signal adder_0_s_net_x100: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x101: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x102: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x103: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x104: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x105: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x106: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x111: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x112: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x113: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x114: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x115: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x116: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x117: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x31: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x32: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x33: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x34: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x35: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x36: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x37: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x38: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x39: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x40: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x41: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x42: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x43: std_logic_vector(26 downto 0);
  signal adder_0_s_net_x53: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x54: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x55: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x56: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x57: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x58: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x59: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x60: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x61: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x62: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x63: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x64: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x65: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x66: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x67: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x68: std_logic_vector(24 downto 0);
  signal adder_0_s_net_x69: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x70: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x71: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x72: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x91: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x92: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x93: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x94: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x95: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x96: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x97: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x98: std_logic_vector(25 downto 0);
  signal adder_0_s_net_x99: std_logic_vector(25 downto 0);
  signal ce_1_sg_x80: std_logic;
  signal clk_1_sg_x80: std_logic;
  signal in0_net_x3: std_logic_vector(23 downto 0);
  signal in10_net_x3: std_logic_vector(23 downto 0);
  signal in11_net_x3: std_logic_vector(23 downto 0);
  signal in12_net_x3: std_logic_vector(23 downto 0);
  signal in13_net_x3: std_logic_vector(23 downto 0);
  signal in14_net_x3: std_logic_vector(23 downto 0);
  signal in15_net_x2: std_logic_vector(23 downto 0);
  signal in1_net_x3: std_logic_vector(23 downto 0);
  signal in2_net_x3: std_logic_vector(23 downto 0);
  signal in3_net_x3: std_logic_vector(23 downto 0);
  signal in4_net_x3: std_logic_vector(23 downto 0);
  signal in5_net_x3: std_logic_vector(23 downto 0);
  signal in6_net_x3: std_logic_vector(23 downto 0);
  signal in7_net_x3: std_logic_vector(23 downto 0);
  signal in8_net_x3: std_logic_vector(23 downto 0);
  signal in9_net_x3: std_logic_vector(23 downto 0);

begin
  ce_1_sg_x80 <= ce_1;
  clk_1_sg_x80 <= clk_1;
  in0_net_x3 <= in1;
  in9_net_x3 <= in10;
  in10_net_x3 <= in11;
  in11_net_x3 <= in12;
  in12_net_x3 <= in13;
  in13_net_x3 <= in14;
  in14_net_x3 <= in15;
  in15_net_x2 <= in16;
  in1_net_x3 <= in2;
  in2_net_x3 <= in3;
  in3_net_x3 <= in4;
  in4_net_x3 <= in5;
  in5_net_x3 <= in6;
  in6_net_x3 <= in7;
  in7_net_x3 <= in8;
  in8_net_x3 <= in9;
  out1 <= adder_0_s_net_x72;
  out2 <= adder_0_s_net_x115;
  out3 <= adder_0_s_net_x116;
  out4 <= adder_0_s_net_x117;
  out5 <= adder_0_s_net_x114;
  out6 <= adder_0_s_net_x69;
  out7 <= adder_0_s_net_x70;
  out8 <= adder_0_s_net_x71;

  p_adder1_7661d8894d: entity work.p_adder1_entity_7661d8894d
    port map (
      ce_1 => ce_1_sg_x80,
      clk_1 => clk_1_sg_x80,
      in1 => in0_net_x3,
      in10 => in9_net_x3,
      in11 => in10_net_x3,
      in12 => in11_net_x3,
      in13 => in12_net_x3,
      in14 => in13_net_x3,
      in15 => in14_net_x3,
      in16 => in15_net_x2,
      in2 => in1_net_x3,
      in3 => in2_net_x3,
      in4 => in3_net_x3,
      in5 => in4_net_x3,
      in6 => in5_net_x3,
      in7 => in6_net_x3,
      in8 => in7_net_x3,
      in9 => in8_net_x3,
      out1 => adder_0_s_net_x53,
      out10 => adder_0_s_net_x54,
      out11 => adder_0_s_net_x55,
      out12 => adder_0_s_net_x56,
      out13 => adder_0_s_net_x57,
      out14 => adder_0_s_net_x58,
      out15 => adder_0_s_net_x59,
      out16 => adder_0_s_net_x60,
      out2 => adder_0_s_net_x61,
      out3 => adder_0_s_net_x62,
      out4 => adder_0_s_net_x63,
      out5 => adder_0_s_net_x64,
      out6 => adder_0_s_net_x65,
      out7 => adder_0_s_net_x66,
      out8 => adder_0_s_net_x67,
      out9 => adder_0_s_net_x68
    );

  p_adder2_252974f8c2: entity work.p_adder2_entity_252974f8c2
    port map (
      ce_1 => ce_1_sg_x80,
      clk_1 => clk_1_sg_x80,
      in1 => adder_0_s_net_x53,
      in10 => adder_0_s_net_x54,
      in11 => adder_0_s_net_x55,
      in12 => adder_0_s_net_x56,
      in13 => adder_0_s_net_x57,
      in14 => adder_0_s_net_x58,
      in15 => adder_0_s_net_x59,
      in16 => adder_0_s_net_x60,
      in2 => adder_0_s_net_x61,
      in3 => adder_0_s_net_x62,
      in4 => adder_0_s_net_x63,
      in5 => adder_0_s_net_x64,
      in6 => adder_0_s_net_x65,
      in7 => adder_0_s_net_x66,
      in8 => adder_0_s_net_x67,
      in9 => adder_0_s_net_x68,
      out1 => adder_0_s_net_x91,
      out10 => adder_0_s_net_x92,
      out11 => adder_0_s_net_x93,
      out12 => adder_0_s_net_x94,
      out13 => adder_0_s_net_x95,
      out14 => adder_0_s_net_x96,
      out15 => adder_0_s_net_x97,
      out16 => adder_0_s_net_x98,
      out2 => adder_0_s_net_x99,
      out3 => adder_0_s_net_x100,
      out4 => adder_0_s_net_x101,
      out5 => adder_0_s_net_x102,
      out6 => adder_0_s_net_x103,
      out7 => adder_0_s_net_x104,
      out8 => adder_0_s_net_x105,
      out9 => adder_0_s_net_x106
    );

  p_adder3_cdc32c2ca9: entity work.p_adder3_entity_cdc32c2ca9
    port map (
      ce_1 => ce_1_sg_x80,
      clk_1 => clk_1_sg_x80,
      in1 => adder_0_s_net_x91,
      in10 => adder_0_s_net_x92,
      in11 => adder_0_s_net_x93,
      in12 => adder_0_s_net_x94,
      in13 => adder_0_s_net_x95,
      in14 => adder_0_s_net_x96,
      in15 => adder_0_s_net_x97,
      in16 => adder_0_s_net_x98,
      in2 => adder_0_s_net_x99,
      in3 => adder_0_s_net_x100,
      in4 => adder_0_s_net_x101,
      in5 => adder_0_s_net_x102,
      in6 => adder_0_s_net_x103,
      in7 => adder_0_s_net_x104,
      in8 => adder_0_s_net_x105,
      in9 => adder_0_s_net_x106,
      out1 => adder_0_s_net_x31,
      out10 => adder_0_s_net_x32,
      out11 => adder_0_s_net_x33,
      out12 => adder_0_s_net_x34,
      out13 => adder_0_s_net_x35,
      out14 => adder_0_s_net_x36,
      out15 => adder_0_s_net_x37,
      out16 => adder_0_s_net_x38,
      out2 => adder_0_s_net_x39,
      out3 => adder_0_s_net_x40,
      out4 => adder_0_s_net_x41,
      out5 => adder_0_s_net_x111,
      out6 => adder_0_s_net_x42,
      out7 => adder_0_s_net_x112,
      out8 => adder_0_s_net_x43,
      out9 => adder_0_s_net_x113
    );

  p_adder4_9312cee7d8: entity work.p_adder4_entity_9312cee7d8
    port map (
      ce_1 => ce_1_sg_x80,
      clk_1 => clk_1_sg_x80,
      in1 => adder_0_s_net_x31,
      in10 => adder_0_s_net_x32,
      in11 => adder_0_s_net_x33,
      in12 => adder_0_s_net_x34,
      in13 => adder_0_s_net_x35,
      in14 => adder_0_s_net_x36,
      in15 => adder_0_s_net_x37,
      in16 => adder_0_s_net_x38,
      in2 => adder_0_s_net_x39,
      in3 => adder_0_s_net_x40,
      in4 => adder_0_s_net_x41,
      in5 => adder_0_s_net_x111,
      in6 => adder_0_s_net_x42,
      in7 => adder_0_s_net_x112,
      in8 => adder_0_s_net_x43,
      in9 => adder_0_s_net_x113,
      out10 => adder_0_s_net_x114,
      out12 => adder_0_s_net_x69,
      out14 => adder_0_s_net_x70,
      out16 => adder_0_s_net_x71,
      out2 => adder_0_s_net_x72,
      out4 => adder_0_s_net_x115,
      out6 => adder_0_s_net_x116,
      out8 => adder_0_s_net_x117
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage2_dec2/p_adder1/adder_tree1"

entity adder_tree1_entity_7d910c0945 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(27 downto 0); 
    din_2: in std_logic_vector(27 downto 0); 
    dout: out std_logic_vector(28 downto 0)
  );
end adder_tree1_entity_7d910c0945;

architecture structural of adder_tree1_entity_7d910c0945 is
  signal adder_0_s_net_x0: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x73: std_logic_vector(27 downto 0);
  signal ce_1_sg_x81: std_logic;
  signal clk_1_sg_x81: std_logic;
  signal d8_x0: std_logic_vector(27 downto 0);

begin
  ce_1_sg_x81 <= ce_1;
  clk_1_sg_x81 <= clk_1;
  adder_0_s_net_x73 <= din_1;
  d8_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_a6d6abc1fb
    port map (
      a => d8_x0,
      b => adder_0_s_net_x73,
      ce => ce_1_sg_x81,
      clk => clk_1_sg_x81,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage2_dec2/p_adder1"

entity p_adder1_entity_3c6ffc487b is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(27 downto 0); 
    in2: in std_logic_vector(27 downto 0); 
    in3: in std_logic_vector(27 downto 0); 
    in4: in std_logic_vector(27 downto 0); 
    in5: in std_logic_vector(27 downto 0); 
    in6: in std_logic_vector(27 downto 0); 
    in7: in std_logic_vector(27 downto 0); 
    in8: in std_logic_vector(27 downto 0); 
    out1: out std_logic_vector(28 downto 0); 
    out2: out std_logic_vector(28 downto 0); 
    out3: out std_logic_vector(28 downto 0); 
    out4: out std_logic_vector(28 downto 0); 
    out5: out std_logic_vector(28 downto 0); 
    out6: out std_logic_vector(28 downto 0); 
    out7: out std_logic_vector(28 downto 0); 
    out8: out std_logic_vector(28 downto 0)
  );
end p_adder1_entity_3c6ffc487b;

architecture structural of p_adder1_entity_3c6ffc487b is
  signal adder_0_s_net_x125: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x126: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x127: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x128: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x129: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x130: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x131: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x132: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x133: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x134: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x135: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x136: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x77: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x78: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x79: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x80: std_logic_vector(27 downto 0);
  signal ce_1_sg_x89: std_logic;
  signal clk_1_sg_x89: std_logic;
  signal d8_x0: std_logic_vector(27 downto 0);

begin
  ce_1_sg_x89 <= ce_1;
  clk_1_sg_x89 <= clk_1;
  adder_0_s_net_x80 <= in1;
  adder_0_s_net_x126 <= in2;
  adder_0_s_net_x127 <= in3;
  adder_0_s_net_x128 <= in4;
  adder_0_s_net_x125 <= in5;
  adder_0_s_net_x77 <= in6;
  adder_0_s_net_x78 <= in7;
  adder_0_s_net_x79 <= in8;
  out1 <= adder_0_s_net_x129;
  out2 <= adder_0_s_net_x130;
  out3 <= adder_0_s_net_x131;
  out4 <= adder_0_s_net_x132;
  out5 <= adder_0_s_net_x133;
  out6 <= adder_0_s_net_x134;
  out7 <= adder_0_s_net_x135;
  out8 <= adder_0_s_net_x136;

  adder_tree1_7d910c0945: entity work.adder_tree1_entity_7d910c0945
    port map (
      ce_1 => ce_1_sg_x89,
      clk_1 => clk_1_sg_x89,
      din_1 => adder_0_s_net_x80,
      din_2 => d8_x0,
      dout => adder_0_s_net_x129
    );

  adder_tree2_d412ea32ce: entity work.adder_tree1_entity_7d910c0945
    port map (
      ce_1 => ce_1_sg_x89,
      clk_1 => clk_1_sg_x89,
      din_1 => adder_0_s_net_x126,
      din_2 => adder_0_s_net_x80,
      dout => adder_0_s_net_x130
    );

  adder_tree3_3e1d61a054: entity work.adder_tree1_entity_7d910c0945
    port map (
      ce_1 => ce_1_sg_x89,
      clk_1 => clk_1_sg_x89,
      din_1 => adder_0_s_net_x127,
      din_2 => adder_0_s_net_x126,
      dout => adder_0_s_net_x131
    );

  adder_tree4_a787865502: entity work.adder_tree1_entity_7d910c0945
    port map (
      ce_1 => ce_1_sg_x89,
      clk_1 => clk_1_sg_x89,
      din_1 => adder_0_s_net_x128,
      din_2 => adder_0_s_net_x127,
      dout => adder_0_s_net_x132
    );

  adder_tree5_6a7e8deed6: entity work.adder_tree1_entity_7d910c0945
    port map (
      ce_1 => ce_1_sg_x89,
      clk_1 => clk_1_sg_x89,
      din_1 => adder_0_s_net_x125,
      din_2 => adder_0_s_net_x128,
      dout => adder_0_s_net_x133
    );

  adder_tree6_40e7f1cf0f: entity work.adder_tree1_entity_7d910c0945
    port map (
      ce_1 => ce_1_sg_x89,
      clk_1 => clk_1_sg_x89,
      din_1 => adder_0_s_net_x77,
      din_2 => adder_0_s_net_x125,
      dout => adder_0_s_net_x134
    );

  adder_tree7_7141b4ee2a: entity work.adder_tree1_entity_7d910c0945
    port map (
      ce_1 => ce_1_sg_x89,
      clk_1 => clk_1_sg_x89,
      din_1 => adder_0_s_net_x78,
      din_2 => adder_0_s_net_x77,
      dout => adder_0_s_net_x135
    );

  adder_tree8_204fd407b7: entity work.adder_tree1_entity_7d910c0945
    port map (
      ce_1 => ce_1_sg_x89,
      clk_1 => clk_1_sg_x89,
      din_1 => adder_0_s_net_x79,
      din_2 => adder_0_s_net_x78,
      dout => adder_0_s_net_x136
    );

  delay8: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 28
    )
    port map (
      ce => ce_1_sg_x89,
      clk => clk_1_sg_x89,
      d => adder_0_s_net_x79,
      en => '1',
      rst => '1',
      q => d8_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage2_dec2/p_adder2/adder_tree1"

entity adder_tree1_entity_63a86f4f14 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(28 downto 0); 
    din_2: in std_logic_vector(28 downto 0); 
    dout: out std_logic_vector(29 downto 0)
  );
end adder_tree1_entity_63a86f4f14;

architecture structural of adder_tree1_entity_63a86f4f14 is
  signal adder_0_s_net_x0: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x130: std_logic_vector(28 downto 0);
  signal ce_1_sg_x90: std_logic;
  signal clk_1_sg_x90: std_logic;
  signal d8_x0: std_logic_vector(28 downto 0);

begin
  ce_1_sg_x90 <= ce_1;
  clk_1_sg_x90 <= clk_1;
  adder_0_s_net_x130 <= din_1;
  d8_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_2de4d085a7
    port map (
      a => d8_x0,
      b => adder_0_s_net_x130,
      ce => ce_1_sg_x90,
      clk => clk_1_sg_x90,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage2_dec2/p_adder2"

entity p_adder2_entity_4a295385cb is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(28 downto 0); 
    in2: in std_logic_vector(28 downto 0); 
    in3: in std_logic_vector(28 downto 0); 
    in4: in std_logic_vector(28 downto 0); 
    in5: in std_logic_vector(28 downto 0); 
    in6: in std_logic_vector(28 downto 0); 
    in7: in std_logic_vector(28 downto 0); 
    in8: in std_logic_vector(28 downto 0); 
    out1: out std_logic_vector(29 downto 0); 
    out2: out std_logic_vector(29 downto 0); 
    out3: out std_logic_vector(29 downto 0); 
    out4: out std_logic_vector(29 downto 0); 
    out5: out std_logic_vector(29 downto 0); 
    out6: out std_logic_vector(29 downto 0); 
    out7: out std_logic_vector(29 downto 0); 
    out8: out std_logic_vector(29 downto 0)
  );
end p_adder2_entity_4a295385cb;

architecture structural of p_adder2_entity_4a295385cb is
  signal adder_0_s_net_x10: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x11: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x12: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x13: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x14: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x142: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x143: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x144: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x145: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x146: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x147: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x148: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x149: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x150: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x8: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x9: std_logic_vector(29 downto 0);
  signal ce_1_sg_x98: std_logic;
  signal clk_1_sg_x98: std_logic;
  signal d8_x0: std_logic_vector(28 downto 0);

begin
  ce_1_sg_x98 <= ce_1;
  clk_1_sg_x98 <= clk_1;
  adder_0_s_net_x142 <= in1;
  adder_0_s_net_x143 <= in2;
  adder_0_s_net_x144 <= in3;
  adder_0_s_net_x145 <= in4;
  adder_0_s_net_x146 <= in5;
  adder_0_s_net_x147 <= in6;
  adder_0_s_net_x148 <= in7;
  adder_0_s_net_x149 <= in8;
  out1 <= adder_0_s_net_x8;
  out2 <= adder_0_s_net_x9;
  out3 <= adder_0_s_net_x10;
  out4 <= adder_0_s_net_x11;
  out5 <= adder_0_s_net_x12;
  out6 <= adder_0_s_net_x13;
  out7 <= adder_0_s_net_x14;
  out8 <= adder_0_s_net_x150;

  adder_tree1_63a86f4f14: entity work.adder_tree1_entity_63a86f4f14
    port map (
      ce_1 => ce_1_sg_x98,
      clk_1 => clk_1_sg_x98,
      din_1 => adder_0_s_net_x142,
      din_2 => d8_x0,
      dout => adder_0_s_net_x8
    );

  adder_tree2_73fc4c9507: entity work.adder_tree1_entity_63a86f4f14
    port map (
      ce_1 => ce_1_sg_x98,
      clk_1 => clk_1_sg_x98,
      din_1 => adder_0_s_net_x143,
      din_2 => adder_0_s_net_x142,
      dout => adder_0_s_net_x9
    );

  adder_tree3_979ae67a92: entity work.adder_tree1_entity_63a86f4f14
    port map (
      ce_1 => ce_1_sg_x98,
      clk_1 => clk_1_sg_x98,
      din_1 => adder_0_s_net_x144,
      din_2 => adder_0_s_net_x143,
      dout => adder_0_s_net_x10
    );

  adder_tree4_704d3158e8: entity work.adder_tree1_entity_63a86f4f14
    port map (
      ce_1 => ce_1_sg_x98,
      clk_1 => clk_1_sg_x98,
      din_1 => adder_0_s_net_x145,
      din_2 => adder_0_s_net_x144,
      dout => adder_0_s_net_x11
    );

  adder_tree5_7a237bb74c: entity work.adder_tree1_entity_63a86f4f14
    port map (
      ce_1 => ce_1_sg_x98,
      clk_1 => clk_1_sg_x98,
      din_1 => adder_0_s_net_x146,
      din_2 => adder_0_s_net_x145,
      dout => adder_0_s_net_x12
    );

  adder_tree6_102587b5d3: entity work.adder_tree1_entity_63a86f4f14
    port map (
      ce_1 => ce_1_sg_x98,
      clk_1 => clk_1_sg_x98,
      din_1 => adder_0_s_net_x147,
      din_2 => adder_0_s_net_x146,
      dout => adder_0_s_net_x13
    );

  adder_tree7_51f7a38670: entity work.adder_tree1_entity_63a86f4f14
    port map (
      ce_1 => ce_1_sg_x98,
      clk_1 => clk_1_sg_x98,
      din_1 => adder_0_s_net_x148,
      din_2 => adder_0_s_net_x147,
      dout => adder_0_s_net_x14
    );

  adder_tree8_abad0302e7: entity work.adder_tree1_entity_63a86f4f14
    port map (
      ce_1 => ce_1_sg_x98,
      clk_1 => clk_1_sg_x98,
      din_1 => adder_0_s_net_x149,
      din_2 => adder_0_s_net_x148,
      dout => adder_0_s_net_x150
    );

  delay8: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 29
    )
    port map (
      ce => ce_1_sg_x98,
      clk => clk_1_sg_x98,
      d => adder_0_s_net_x149,
      en => '1',
      rst => '1',
      q => d8_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage2_dec2/p_adder3/adder_tree1"

entity adder_tree1_entity_578796265e is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(29 downto 0); 
    din_2: in std_logic_vector(29 downto 0); 
    dout: out std_logic_vector(30 downto 0)
  );
end adder_tree1_entity_578796265e;

architecture structural of adder_tree1_entity_578796265e is
  signal adder_0_s_net_x0: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x9: std_logic_vector(29 downto 0);
  signal ce_1_sg_x99: std_logic;
  signal clk_1_sg_x99: std_logic;
  signal d8_x0: std_logic_vector(29 downto 0);

begin
  ce_1_sg_x99 <= ce_1;
  clk_1_sg_x99 <= clk_1;
  adder_0_s_net_x9 <= din_1;
  d8_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_78fc83082e
    port map (
      a => d8_x0,
      b => adder_0_s_net_x9,
      ce => ce_1_sg_x99,
      clk => clk_1_sg_x99,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage2_dec2/p_adder3"

entity p_adder3_entity_6a24f9e40a is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(29 downto 0); 
    in2: in std_logic_vector(29 downto 0); 
    in3: in std_logic_vector(29 downto 0); 
    in4: in std_logic_vector(29 downto 0); 
    in5: in std_logic_vector(29 downto 0); 
    in6: in std_logic_vector(29 downto 0); 
    in7: in std_logic_vector(29 downto 0); 
    in8: in std_logic_vector(29 downto 0); 
    out1: out std_logic_vector(30 downto 0); 
    out2: out std_logic_vector(30 downto 0); 
    out3: out std_logic_vector(30 downto 0); 
    out4: out std_logic_vector(30 downto 0); 
    out5: out std_logic_vector(30 downto 0); 
    out6: out std_logic_vector(30 downto 0); 
    out7: out std_logic_vector(30 downto 0); 
    out8: out std_logic_vector(30 downto 0)
  );
end p_adder3_entity_6a24f9e40a;

architecture structural of p_adder3_entity_6a24f9e40a is
  signal adder_0_s_net_x152: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x153: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x27: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x28: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x29: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x30: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x31: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x32: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x33: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x34: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x35: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x36: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x37: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x38: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x39: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x40: std_logic_vector(30 downto 0);
  signal ce_1_sg_x107: std_logic;
  signal clk_1_sg_x107: std_logic;
  signal d8_x0: std_logic_vector(29 downto 0);

begin
  ce_1_sg_x107 <= ce_1;
  clk_1_sg_x107 <= clk_1;
  adder_0_s_net_x27 <= in1;
  adder_0_s_net_x28 <= in2;
  adder_0_s_net_x29 <= in3;
  adder_0_s_net_x30 <= in4;
  adder_0_s_net_x31 <= in5;
  adder_0_s_net_x32 <= in6;
  adder_0_s_net_x33 <= in7;
  adder_0_s_net_x152 <= in8;
  out1 <= adder_0_s_net_x34;
  out2 <= adder_0_s_net_x35;
  out3 <= adder_0_s_net_x36;
  out4 <= adder_0_s_net_x37;
  out5 <= adder_0_s_net_x38;
  out6 <= adder_0_s_net_x39;
  out7 <= adder_0_s_net_x40;
  out8 <= adder_0_s_net_x153;

  adder_tree1_578796265e: entity work.adder_tree1_entity_578796265e
    port map (
      ce_1 => ce_1_sg_x107,
      clk_1 => clk_1_sg_x107,
      din_1 => adder_0_s_net_x27,
      din_2 => d8_x0,
      dout => adder_0_s_net_x34
    );

  adder_tree2_26191ce105: entity work.adder_tree1_entity_578796265e
    port map (
      ce_1 => ce_1_sg_x107,
      clk_1 => clk_1_sg_x107,
      din_1 => adder_0_s_net_x28,
      din_2 => adder_0_s_net_x27,
      dout => adder_0_s_net_x35
    );

  adder_tree3_85b4eca831: entity work.adder_tree1_entity_578796265e
    port map (
      ce_1 => ce_1_sg_x107,
      clk_1 => clk_1_sg_x107,
      din_1 => adder_0_s_net_x29,
      din_2 => adder_0_s_net_x28,
      dout => adder_0_s_net_x36
    );

  adder_tree4_72a5a00334: entity work.adder_tree1_entity_578796265e
    port map (
      ce_1 => ce_1_sg_x107,
      clk_1 => clk_1_sg_x107,
      din_1 => adder_0_s_net_x30,
      din_2 => adder_0_s_net_x29,
      dout => adder_0_s_net_x37
    );

  adder_tree5_4b1f19d7fe: entity work.adder_tree1_entity_578796265e
    port map (
      ce_1 => ce_1_sg_x107,
      clk_1 => clk_1_sg_x107,
      din_1 => adder_0_s_net_x31,
      din_2 => adder_0_s_net_x30,
      dout => adder_0_s_net_x38
    );

  adder_tree6_63c9ecdd20: entity work.adder_tree1_entity_578796265e
    port map (
      ce_1 => ce_1_sg_x107,
      clk_1 => clk_1_sg_x107,
      din_1 => adder_0_s_net_x32,
      din_2 => adder_0_s_net_x31,
      dout => adder_0_s_net_x39
    );

  adder_tree7_42eb138323: entity work.adder_tree1_entity_578796265e
    port map (
      ce_1 => ce_1_sg_x107,
      clk_1 => clk_1_sg_x107,
      din_1 => adder_0_s_net_x33,
      din_2 => adder_0_s_net_x32,
      dout => adder_0_s_net_x40
    );

  adder_tree8_4f6aa57901: entity work.adder_tree1_entity_578796265e
    port map (
      ce_1 => ce_1_sg_x107,
      clk_1 => clk_1_sg_x107,
      din_1 => adder_0_s_net_x152,
      din_2 => adder_0_s_net_x33,
      dout => adder_0_s_net_x153
    );

  delay8: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 30
    )
    port map (
      ce => ce_1_sg_x107,
      clk => clk_1_sg_x107,
      d => adder_0_s_net_x152,
      en => '1',
      rst => '1',
      q => d8_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage2_dec2/p_adder4/adder_tree2"

entity adder_tree2_entity_5c5381b772 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(30 downto 0); 
    din_2: in std_logic_vector(30 downto 0); 
    dout: out std_logic_vector(31 downto 0)
  );
end adder_tree2_entity_5c5381b772;

architecture structural of adder_tree2_entity_5c5381b772 is
  signal adder_0_s_net_x0: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x36: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x37: std_logic_vector(30 downto 0);
  signal ce_1_sg_x108: std_logic;
  signal clk_1_sg_x108: std_logic;

begin
  ce_1_sg_x108 <= ce_1;
  clk_1_sg_x108 <= clk_1;
  adder_0_s_net_x37 <= din_1;
  adder_0_s_net_x36 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_2a7fe08e67
    port map (
      a => adder_0_s_net_x36,
      b => adder_0_s_net_x37,
      ce => ce_1_sg_x108,
      clk => clk_1_sg_x108,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage2_dec2/p_adder4"

entity p_adder4_entity_a5092f85da is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(30 downto 0); 
    in2: in std_logic_vector(30 downto 0); 
    in3: in std_logic_vector(30 downto 0); 
    in4: in std_logic_vector(30 downto 0); 
    in5: in std_logic_vector(30 downto 0); 
    in6: in std_logic_vector(30 downto 0); 
    in7: in std_logic_vector(30 downto 0); 
    in8: in std_logic_vector(30 downto 0); 
    out2: out std_logic_vector(31 downto 0); 
    out4: out std_logic_vector(31 downto 0); 
    out6: out std_logic_vector(31 downto 0); 
    out8: out std_logic_vector(31 downto 0)
  );
end p_adder4_entity_a5092f85da;

architecture structural of p_adder4_entity_a5092f85da is
  signal adder_0_s_net_x155: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x4: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x43: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x44: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x45: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x46: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x47: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x48: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x49: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x5: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x6: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x7: std_logic_vector(31 downto 0);
  signal ce_1_sg_x112: std_logic;
  signal clk_1_sg_x112: std_logic;

begin
  ce_1_sg_x112 <= ce_1;
  clk_1_sg_x112 <= clk_1;
  adder_0_s_net_x43 <= in1;
  adder_0_s_net_x44 <= in2;
  adder_0_s_net_x45 <= in3;
  adder_0_s_net_x46 <= in4;
  adder_0_s_net_x47 <= in5;
  adder_0_s_net_x48 <= in6;
  adder_0_s_net_x49 <= in7;
  adder_0_s_net_x155 <= in8;
  out2 <= adder_0_s_net_x4;
  out4 <= adder_0_s_net_x5;
  out6 <= adder_0_s_net_x6;
  out8 <= adder_0_s_net_x7;

  adder_tree2_5c5381b772: entity work.adder_tree2_entity_5c5381b772
    port map (
      ce_1 => ce_1_sg_x112,
      clk_1 => clk_1_sg_x112,
      din_1 => adder_0_s_net_x44,
      din_2 => adder_0_s_net_x43,
      dout => adder_0_s_net_x4
    );

  adder_tree4_1579e19049: entity work.adder_tree2_entity_5c5381b772
    port map (
      ce_1 => ce_1_sg_x112,
      clk_1 => clk_1_sg_x112,
      din_1 => adder_0_s_net_x46,
      din_2 => adder_0_s_net_x45,
      dout => adder_0_s_net_x5
    );

  adder_tree6_39c8f3079e: entity work.adder_tree2_entity_5c5381b772
    port map (
      ce_1 => ce_1_sg_x112,
      clk_1 => clk_1_sg_x112,
      din_1 => adder_0_s_net_x48,
      din_2 => adder_0_s_net_x47,
      dout => adder_0_s_net_x6
    );

  adder_tree8_b0adb61240: entity work.adder_tree2_entity_5c5381b772
    port map (
      ce_1 => ce_1_sg_x112,
      clk_1 => clk_1_sg_x112,
      din_1 => adder_0_s_net_x155,
      din_2 => adder_0_s_net_x49,
      dout => adder_0_s_net_x7
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage2_dec2"

entity stage2_dec2_entity_0b5dc5c0dd is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(27 downto 0); 
    in2: in std_logic_vector(27 downto 0); 
    in3: in std_logic_vector(27 downto 0); 
    in4: in std_logic_vector(27 downto 0); 
    in5: in std_logic_vector(27 downto 0); 
    in6: in std_logic_vector(27 downto 0); 
    in7: in std_logic_vector(27 downto 0); 
    in8: in std_logic_vector(27 downto 0); 
    out1: out std_logic_vector(31 downto 0); 
    out2: out std_logic_vector(31 downto 0); 
    out3: out std_logic_vector(31 downto 0); 
    out4: out std_logic_vector(31 downto 0)
  );
end stage2_dec2_entity_0b5dc5c0dd;

architecture structural of stage2_dec2_entity_0b5dc5c0dd is
  signal adder_0_s_net_x129: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x130: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x142: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x143: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x144: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x145: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x146: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x147: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x148: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x149: std_logic_vector(28 downto 0);
  signal adder_0_s_net_x150: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x151: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x152: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x153: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x154: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x155: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x156: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x27: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x28: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x29: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x30: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x31: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x32: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x33: std_logic_vector(29 downto 0);
  signal adder_0_s_net_x43: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x44: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x45: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x46: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x47: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x48: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x49: std_logic_vector(30 downto 0);
  signal adder_0_s_net_x8: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x81: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x82: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x83: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x9: std_logic_vector(31 downto 0);
  signal ce_1_sg_x113: std_logic;
  signal clk_1_sg_x113: std_logic;

begin
  ce_1_sg_x113 <= ce_1;
  clk_1_sg_x113 <= clk_1;
  adder_0_s_net_x150 <= in1;
  adder_0_s_net_x130 <= in2;
  adder_0_s_net_x153 <= in3;
  adder_0_s_net_x151 <= in4;
  adder_0_s_net_x129 <= in5;
  adder_0_s_net_x81 <= in6;
  adder_0_s_net_x82 <= in7;
  adder_0_s_net_x83 <= in8;
  out1 <= adder_0_s_net_x154;
  out2 <= adder_0_s_net_x156;
  out3 <= adder_0_s_net_x8;
  out4 <= adder_0_s_net_x9;

  p_adder1_3c6ffc487b: entity work.p_adder1_entity_3c6ffc487b
    port map (
      ce_1 => ce_1_sg_x113,
      clk_1 => clk_1_sg_x113,
      in1 => adder_0_s_net_x150,
      in2 => adder_0_s_net_x130,
      in3 => adder_0_s_net_x153,
      in4 => adder_0_s_net_x151,
      in5 => adder_0_s_net_x129,
      in6 => adder_0_s_net_x81,
      in7 => adder_0_s_net_x82,
      in8 => adder_0_s_net_x83,
      out1 => adder_0_s_net_x142,
      out2 => adder_0_s_net_x143,
      out3 => adder_0_s_net_x144,
      out4 => adder_0_s_net_x145,
      out5 => adder_0_s_net_x146,
      out6 => adder_0_s_net_x147,
      out7 => adder_0_s_net_x148,
      out8 => adder_0_s_net_x149
    );

  p_adder2_4a295385cb: entity work.p_adder2_entity_4a295385cb
    port map (
      ce_1 => ce_1_sg_x113,
      clk_1 => clk_1_sg_x113,
      in1 => adder_0_s_net_x142,
      in2 => adder_0_s_net_x143,
      in3 => adder_0_s_net_x144,
      in4 => adder_0_s_net_x145,
      in5 => adder_0_s_net_x146,
      in6 => adder_0_s_net_x147,
      in7 => adder_0_s_net_x148,
      in8 => adder_0_s_net_x149,
      out1 => adder_0_s_net_x27,
      out2 => adder_0_s_net_x28,
      out3 => adder_0_s_net_x29,
      out4 => adder_0_s_net_x30,
      out5 => adder_0_s_net_x31,
      out6 => adder_0_s_net_x32,
      out7 => adder_0_s_net_x33,
      out8 => adder_0_s_net_x152
    );

  p_adder3_6a24f9e40a: entity work.p_adder3_entity_6a24f9e40a
    port map (
      ce_1 => ce_1_sg_x113,
      clk_1 => clk_1_sg_x113,
      in1 => adder_0_s_net_x27,
      in2 => adder_0_s_net_x28,
      in3 => adder_0_s_net_x29,
      in4 => adder_0_s_net_x30,
      in5 => adder_0_s_net_x31,
      in6 => adder_0_s_net_x32,
      in7 => adder_0_s_net_x33,
      in8 => adder_0_s_net_x152,
      out1 => adder_0_s_net_x43,
      out2 => adder_0_s_net_x44,
      out3 => adder_0_s_net_x45,
      out4 => adder_0_s_net_x46,
      out5 => adder_0_s_net_x47,
      out6 => adder_0_s_net_x48,
      out7 => adder_0_s_net_x49,
      out8 => adder_0_s_net_x155
    );

  p_adder4_a5092f85da: entity work.p_adder4_entity_a5092f85da
    port map (
      ce_1 => ce_1_sg_x113,
      clk_1 => clk_1_sg_x113,
      in1 => adder_0_s_net_x43,
      in2 => adder_0_s_net_x44,
      in3 => adder_0_s_net_x45,
      in4 => adder_0_s_net_x46,
      in5 => adder_0_s_net_x47,
      in6 => adder_0_s_net_x48,
      in7 => adder_0_s_net_x49,
      in8 => adder_0_s_net_x155,
      out2 => adder_0_s_net_x154,
      out4 => adder_0_s_net_x156,
      out6 => adder_0_s_net_x8,
      out8 => adder_0_s_net_x9
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage3_dec2/p_adder1/adder_tree1"

entity adder_tree1_entity_bf524f6f6a is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(31 downto 0); 
    din_2: in std_logic_vector(31 downto 0); 
    dout: out std_logic_vector(32 downto 0)
  );
end adder_tree1_entity_bf524f6f6a;

architecture structural of adder_tree1_entity_bf524f6f6a is
  signal adder_0_s_net_x0: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x155: std_logic_vector(31 downto 0);
  signal ce_1_sg_x114: std_logic;
  signal clk_1_sg_x114: std_logic;
  signal d4_x0: std_logic_vector(31 downto 0);

begin
  ce_1_sg_x114 <= ce_1;
  clk_1_sg_x114 <= clk_1;
  adder_0_s_net_x155 <= din_1;
  d4_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_f1849463e9
    port map (
      a => d4_x0,
      b => adder_0_s_net_x155,
      ce => ce_1_sg_x114,
      clk => clk_1_sg_x114,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage3_dec2/p_adder1"

entity p_adder1_entity_a818d6e68b is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(31 downto 0); 
    in2: in std_logic_vector(31 downto 0); 
    in3: in std_logic_vector(31 downto 0); 
    in4: in std_logic_vector(31 downto 0); 
    out1: out std_logic_vector(32 downto 0); 
    out2: out std_logic_vector(32 downto 0); 
    out3: out std_logic_vector(32 downto 0); 
    out4: out std_logic_vector(32 downto 0)
  );
end p_adder1_entity_a818d6e68b;

architecture structural of p_adder1_entity_a818d6e68b is
  signal adder_0_s_net_x13: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x14: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x158: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x160: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x4: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x5: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x6: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x7: std_logic_vector(32 downto 0);
  signal ce_1_sg_x118: std_logic;
  signal clk_1_sg_x118: std_logic;
  signal d4_x0: std_logic_vector(31 downto 0);

begin
  ce_1_sg_x118 <= ce_1;
  clk_1_sg_x118 <= clk_1;
  adder_0_s_net_x158 <= in1;
  adder_0_s_net_x160 <= in2;
  adder_0_s_net_x13 <= in3;
  adder_0_s_net_x14 <= in4;
  out1 <= adder_0_s_net_x4;
  out2 <= adder_0_s_net_x5;
  out3 <= adder_0_s_net_x6;
  out4 <= adder_0_s_net_x7;

  adder_tree1_bf524f6f6a: entity work.adder_tree1_entity_bf524f6f6a
    port map (
      ce_1 => ce_1_sg_x118,
      clk_1 => clk_1_sg_x118,
      din_1 => adder_0_s_net_x158,
      din_2 => d4_x0,
      dout => adder_0_s_net_x4
    );

  adder_tree2_482ffc1f87: entity work.adder_tree1_entity_bf524f6f6a
    port map (
      ce_1 => ce_1_sg_x118,
      clk_1 => clk_1_sg_x118,
      din_1 => adder_0_s_net_x160,
      din_2 => adder_0_s_net_x158,
      dout => adder_0_s_net_x5
    );

  adder_tree3_07d9ce4dd7: entity work.adder_tree1_entity_bf524f6f6a
    port map (
      ce_1 => ce_1_sg_x118,
      clk_1 => clk_1_sg_x118,
      din_1 => adder_0_s_net_x13,
      din_2 => adder_0_s_net_x160,
      dout => adder_0_s_net_x6
    );

  adder_tree4_9f2016bc46: entity work.adder_tree1_entity_bf524f6f6a
    port map (
      ce_1 => ce_1_sg_x118,
      clk_1 => clk_1_sg_x118,
      din_1 => adder_0_s_net_x14,
      din_2 => adder_0_s_net_x13,
      dout => adder_0_s_net_x7
    );

  delay4: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x118,
      clk => clk_1_sg_x118,
      d => adder_0_s_net_x14,
      en => '1',
      rst => '1',
      q => d4_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage3_dec2/p_adder2/adder_tree1"

entity adder_tree1_entity_c6fca0fcfb is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(32 downto 0); 
    din_2: in std_logic_vector(32 downto 0); 
    dout: out std_logic_vector(33 downto 0)
  );
end adder_tree1_entity_c6fca0fcfb;

architecture structural of adder_tree1_entity_c6fca0fcfb is
  signal adder_0_s_net_x0: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x5: std_logic_vector(32 downto 0);
  signal ce_1_sg_x119: std_logic;
  signal clk_1_sg_x119: std_logic;
  signal d4_x0: std_logic_vector(32 downto 0);

begin
  ce_1_sg_x119 <= ce_1;
  clk_1_sg_x119 <= clk_1;
  adder_0_s_net_x5 <= din_1;
  d4_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_e63f1751f5
    port map (
      a => d4_x0,
      b => adder_0_s_net_x5,
      ce => ce_1_sg_x119,
      clk => clk_1_sg_x119,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage3_dec2/p_adder2"

entity p_adder2_entity_bb24f12742 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(32 downto 0); 
    in2: in std_logic_vector(32 downto 0); 
    in3: in std_logic_vector(32 downto 0); 
    in4: in std_logic_vector(32 downto 0); 
    out1: out std_logic_vector(33 downto 0); 
    out2: out std_logic_vector(33 downto 0); 
    out3: out std_logic_vector(33 downto 0); 
    out4: out std_logic_vector(33 downto 0)
  );
end p_adder2_entity_bb24f12742;

architecture structural of p_adder2_entity_bb24f12742 is
  signal adder_0_s_net_x16: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x17: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x18: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x19: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x20: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x21: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x22: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x23: std_logic_vector(33 downto 0);
  signal ce_1_sg_x123: std_logic;
  signal clk_1_sg_x123: std_logic;
  signal d4_x0: std_logic_vector(32 downto 0);

begin
  ce_1_sg_x123 <= ce_1;
  clk_1_sg_x123 <= clk_1;
  adder_0_s_net_x16 <= in1;
  adder_0_s_net_x17 <= in2;
  adder_0_s_net_x18 <= in3;
  adder_0_s_net_x19 <= in4;
  out1 <= adder_0_s_net_x20;
  out2 <= adder_0_s_net_x21;
  out3 <= adder_0_s_net_x22;
  out4 <= adder_0_s_net_x23;

  adder_tree1_c6fca0fcfb: entity work.adder_tree1_entity_c6fca0fcfb
    port map (
      ce_1 => ce_1_sg_x123,
      clk_1 => clk_1_sg_x123,
      din_1 => adder_0_s_net_x16,
      din_2 => d4_x0,
      dout => adder_0_s_net_x20
    );

  adder_tree2_09e4231899: entity work.adder_tree1_entity_c6fca0fcfb
    port map (
      ce_1 => ce_1_sg_x123,
      clk_1 => clk_1_sg_x123,
      din_1 => adder_0_s_net_x17,
      din_2 => adder_0_s_net_x16,
      dout => adder_0_s_net_x21
    );

  adder_tree3_f3f25ff4ec: entity work.adder_tree1_entity_c6fca0fcfb
    port map (
      ce_1 => ce_1_sg_x123,
      clk_1 => clk_1_sg_x123,
      din_1 => adder_0_s_net_x18,
      din_2 => adder_0_s_net_x17,
      dout => adder_0_s_net_x22
    );

  adder_tree4_1110c2967f: entity work.adder_tree1_entity_c6fca0fcfb
    port map (
      ce_1 => ce_1_sg_x123,
      clk_1 => clk_1_sg_x123,
      din_1 => adder_0_s_net_x19,
      din_2 => adder_0_s_net_x18,
      dout => adder_0_s_net_x23
    );

  delay4: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 33
    )
    port map (
      ce => ce_1_sg_x123,
      clk => clk_1_sg_x123,
      d => adder_0_s_net_x19,
      en => '1',
      rst => '1',
      q => d4_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage3_dec2/p_adder3/adder_tree1"

entity adder_tree1_entity_48b7c822d0 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(33 downto 0); 
    din_2: in std_logic_vector(33 downto 0); 
    dout: out std_logic_vector(34 downto 0)
  );
end adder_tree1_entity_48b7c822d0;

architecture structural of adder_tree1_entity_48b7c822d0 is
  signal adder_0_s_net_x0: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x21: std_logic_vector(33 downto 0);
  signal ce_1_sg_x124: std_logic;
  signal clk_1_sg_x124: std_logic;
  signal d4_x0: std_logic_vector(33 downto 0);

begin
  ce_1_sg_x124 <= ce_1;
  clk_1_sg_x124 <= clk_1;
  adder_0_s_net_x21 <= din_1;
  d4_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_5d08e5a27e
    port map (
      a => d4_x0,
      b => adder_0_s_net_x21,
      ce => ce_1_sg_x124,
      clk => clk_1_sg_x124,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage3_dec2/p_adder3"

entity p_adder3_entity_aed56babbd is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(33 downto 0); 
    in2: in std_logic_vector(33 downto 0); 
    in3: in std_logic_vector(33 downto 0); 
    in4: in std_logic_vector(33 downto 0); 
    out1: out std_logic_vector(34 downto 0); 
    out2: out std_logic_vector(34 downto 0); 
    out3: out std_logic_vector(34 downto 0); 
    out4: out std_logic_vector(34 downto 0)
  );
end p_adder3_entity_aed56babbd;

architecture structural of p_adder3_entity_aed56babbd is
  signal adder_0_s_net_x24: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x29: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x30: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x31: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x32: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x33: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x34: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x35: std_logic_vector(34 downto 0);
  signal ce_1_sg_x128: std_logic;
  signal clk_1_sg_x128: std_logic;
  signal d4_x0: std_logic_vector(33 downto 0);

begin
  ce_1_sg_x128 <= ce_1;
  clk_1_sg_x128 <= clk_1;
  adder_0_s_net_x24 <= in1;
  adder_0_s_net_x29 <= in2;
  adder_0_s_net_x30 <= in3;
  adder_0_s_net_x31 <= in4;
  out1 <= adder_0_s_net_x32;
  out2 <= adder_0_s_net_x33;
  out3 <= adder_0_s_net_x34;
  out4 <= adder_0_s_net_x35;

  adder_tree1_48b7c822d0: entity work.adder_tree1_entity_48b7c822d0
    port map (
      ce_1 => ce_1_sg_x128,
      clk_1 => clk_1_sg_x128,
      din_1 => adder_0_s_net_x24,
      din_2 => d4_x0,
      dout => adder_0_s_net_x32
    );

  adder_tree2_4160717c08: entity work.adder_tree1_entity_48b7c822d0
    port map (
      ce_1 => ce_1_sg_x128,
      clk_1 => clk_1_sg_x128,
      din_1 => adder_0_s_net_x29,
      din_2 => adder_0_s_net_x24,
      dout => adder_0_s_net_x33
    );

  adder_tree3_9e72ba98ea: entity work.adder_tree1_entity_48b7c822d0
    port map (
      ce_1 => ce_1_sg_x128,
      clk_1 => clk_1_sg_x128,
      din_1 => adder_0_s_net_x30,
      din_2 => adder_0_s_net_x29,
      dout => adder_0_s_net_x34
    );

  adder_tree4_d9f39834f6: entity work.adder_tree1_entity_48b7c822d0
    port map (
      ce_1 => ce_1_sg_x128,
      clk_1 => clk_1_sg_x128,
      din_1 => adder_0_s_net_x31,
      din_2 => adder_0_s_net_x30,
      dout => adder_0_s_net_x35
    );

  delay4: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 34
    )
    port map (
      ce => ce_1_sg_x128,
      clk => clk_1_sg_x128,
      d => adder_0_s_net_x31,
      en => '1',
      rst => '1',
      q => d4_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage3_dec2/p_adder4/adder_tree2"

entity adder_tree2_entity_9be1cbe10a is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(34 downto 0); 
    din_2: in std_logic_vector(34 downto 0); 
    dout: out std_logic_vector(35 downto 0)
  );
end adder_tree2_entity_9be1cbe10a;

architecture structural of adder_tree2_entity_9be1cbe10a is
  signal adder_0_s_net_x0: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x34: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x35: std_logic_vector(34 downto 0);
  signal ce_1_sg_x129: std_logic;
  signal clk_1_sg_x129: std_logic;

begin
  ce_1_sg_x129 <= ce_1;
  clk_1_sg_x129 <= clk_1;
  adder_0_s_net_x35 <= din_1;
  adder_0_s_net_x34 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_849323e8f6
    port map (
      a => adder_0_s_net_x34,
      b => adder_0_s_net_x35,
      ce => ce_1_sg_x129,
      clk => clk_1_sg_x129,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage3_dec2/p_adder4"

entity p_adder4_entity_4081ffc9c0 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(34 downto 0); 
    in2: in std_logic_vector(34 downto 0); 
    in3: in std_logic_vector(34 downto 0); 
    in4: in std_logic_vector(34 downto 0); 
    out2: out std_logic_vector(35 downto 0); 
    out4: out std_logic_vector(35 downto 0)
  );
end p_adder4_entity_4081ffc9c0;

architecture structural of p_adder4_entity_4081ffc9c0 is
  signal adder_0_s_net_x2: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x3: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x38: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x39: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x40: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x41: std_logic_vector(34 downto 0);
  signal ce_1_sg_x131: std_logic;
  signal clk_1_sg_x131: std_logic;

begin
  ce_1_sg_x131 <= ce_1;
  clk_1_sg_x131 <= clk_1;
  adder_0_s_net_x38 <= in1;
  adder_0_s_net_x39 <= in2;
  adder_0_s_net_x40 <= in3;
  adder_0_s_net_x41 <= in4;
  out2 <= adder_0_s_net_x2;
  out4 <= adder_0_s_net_x3;

  adder_tree2_9be1cbe10a: entity work.adder_tree2_entity_9be1cbe10a
    port map (
      ce_1 => ce_1_sg_x131,
      clk_1 => clk_1_sg_x131,
      din_1 => adder_0_s_net_x39,
      din_2 => adder_0_s_net_x38,
      dout => adder_0_s_net_x2
    );

  adder_tree4_0fd4508c12: entity work.adder_tree2_entity_9be1cbe10a
    port map (
      ce_1 => ce_1_sg_x131,
      clk_1 => clk_1_sg_x131,
      din_1 => adder_0_s_net_x41,
      din_2 => adder_0_s_net_x40,
      dout => adder_0_s_net_x3
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage3_dec2"

entity stage3_dec2_entity_4774a9beac is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(31 downto 0); 
    in2: in std_logic_vector(31 downto 0); 
    in3: in std_logic_vector(31 downto 0); 
    in4: in std_logic_vector(31 downto 0); 
    out1: out std_logic_vector(35 downto 0); 
    out2: out std_logic_vector(35 downto 0)
  );
end stage3_dec2_entity_4774a9beac;

architecture structural of stage3_dec2_entity_4774a9beac is
  signal adder_0_s_net_x159: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x16: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x161: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x17: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x18: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x19: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x20: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x21: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x22: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x24: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x29: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x30: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x31: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x32: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x38: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x39: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x40: std_logic_vector(34 downto 0);
  signal adder_0_s_net_x41: std_logic_vector(34 downto 0);
  signal ce_1_sg_x132: std_logic;
  signal clk_1_sg_x132: std_logic;

begin
  ce_1_sg_x132 <= ce_1;
  clk_1_sg_x132 <= clk_1;
  adder_0_s_net_x159 <= in1;
  adder_0_s_net_x161 <= in2;
  adder_0_s_net_x20 <= in3;
  adder_0_s_net_x32 <= in4;
  out1 <= adder_0_s_net_x21;
  out2 <= adder_0_s_net_x22;

  p_adder1_a818d6e68b: entity work.p_adder1_entity_a818d6e68b
    port map (
      ce_1 => ce_1_sg_x132,
      clk_1 => clk_1_sg_x132,
      in1 => adder_0_s_net_x159,
      in2 => adder_0_s_net_x161,
      in3 => adder_0_s_net_x20,
      in4 => adder_0_s_net_x32,
      out1 => adder_0_s_net_x16,
      out2 => adder_0_s_net_x17,
      out3 => adder_0_s_net_x18,
      out4 => adder_0_s_net_x19
    );

  p_adder2_bb24f12742: entity work.p_adder2_entity_bb24f12742
    port map (
      ce_1 => ce_1_sg_x132,
      clk_1 => clk_1_sg_x132,
      in1 => adder_0_s_net_x16,
      in2 => adder_0_s_net_x17,
      in3 => adder_0_s_net_x18,
      in4 => adder_0_s_net_x19,
      out1 => adder_0_s_net_x24,
      out2 => adder_0_s_net_x29,
      out3 => adder_0_s_net_x30,
      out4 => adder_0_s_net_x31
    );

  p_adder3_aed56babbd: entity work.p_adder3_entity_aed56babbd
    port map (
      ce_1 => ce_1_sg_x132,
      clk_1 => clk_1_sg_x132,
      in1 => adder_0_s_net_x24,
      in2 => adder_0_s_net_x29,
      in3 => adder_0_s_net_x30,
      in4 => adder_0_s_net_x31,
      out1 => adder_0_s_net_x38,
      out2 => adder_0_s_net_x39,
      out3 => adder_0_s_net_x40,
      out4 => adder_0_s_net_x41
    );

  p_adder4_4081ffc9c0: entity work.p_adder4_entity_4081ffc9c0
    port map (
      ce_1 => ce_1_sg_x132,
      clk_1 => clk_1_sg_x132,
      in1 => adder_0_s_net_x38,
      in2 => adder_0_s_net_x39,
      in3 => adder_0_s_net_x40,
      in4 => adder_0_s_net_x41,
      out2 => adder_0_s_net_x21,
      out4 => adder_0_s_net_x22
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage4_dec2/p_adder1/adder_tree1"

entity adder_tree1_entity_34871ad021 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(35 downto 0); 
    din_2: in std_logic_vector(35 downto 0); 
    dout: out std_logic_vector(36 downto 0)
  );
end adder_tree1_entity_34871ad021;

architecture structural of adder_tree1_entity_34871ad021 is
  signal adder_0_s_net_x0: std_logic_vector(36 downto 0);
  signal adder_0_s_net_x22: std_logic_vector(35 downto 0);
  signal ce_1_sg_x133: std_logic;
  signal clk_1_sg_x133: std_logic;
  signal d2_x0: std_logic_vector(35 downto 0);

begin
  ce_1_sg_x133 <= ce_1;
  clk_1_sg_x133 <= clk_1;
  adder_0_s_net_x22 <= din_1;
  d2_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_f62aecf512
    port map (
      a => d2_x0,
      b => adder_0_s_net_x22,
      ce => ce_1_sg_x133,
      clk => clk_1_sg_x133,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage4_dec2/p_adder1"

entity p_adder1_entity_bbc5e9a942 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(35 downto 0); 
    in2: in std_logic_vector(35 downto 0); 
    out1: out std_logic_vector(36 downto 0); 
    out2: out std_logic_vector(36 downto 0)
  );
end p_adder1_entity_bbc5e9a942;

architecture structural of p_adder1_entity_bbc5e9a942 is
  signal adder_0_s_net_x2: std_logic_vector(36 downto 0);
  signal adder_0_s_net_x26: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x27: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x3: std_logic_vector(36 downto 0);
  signal ce_1_sg_x135: std_logic;
  signal clk_1_sg_x135: std_logic;
  signal d2_x0: std_logic_vector(35 downto 0);

begin
  ce_1_sg_x135 <= ce_1;
  clk_1_sg_x135 <= clk_1;
  adder_0_s_net_x26 <= in1;
  adder_0_s_net_x27 <= in2;
  out1 <= adder_0_s_net_x2;
  out2 <= adder_0_s_net_x3;

  adder_tree1_34871ad021: entity work.adder_tree1_entity_34871ad021
    port map (
      ce_1 => ce_1_sg_x135,
      clk_1 => clk_1_sg_x135,
      din_1 => adder_0_s_net_x26,
      din_2 => d2_x0,
      dout => adder_0_s_net_x2
    );

  adder_tree2_14ba0870da: entity work.adder_tree1_entity_34871ad021
    port map (
      ce_1 => ce_1_sg_x135,
      clk_1 => clk_1_sg_x135,
      din_1 => adder_0_s_net_x27,
      din_2 => adder_0_s_net_x26,
      dout => adder_0_s_net_x3
    );

  delay2: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 36
    )
    port map (
      ce => ce_1_sg_x135,
      clk => clk_1_sg_x135,
      d => adder_0_s_net_x27,
      en => '1',
      rst => '1',
      q => d2_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage4_dec2/p_adder2/adder_tree1"

entity adder_tree1_entity_7dd3bbbaf1 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(36 downto 0); 
    din_2: in std_logic_vector(36 downto 0); 
    dout: out std_logic_vector(37 downto 0)
  );
end adder_tree1_entity_7dd3bbbaf1;

architecture structural of adder_tree1_entity_7dd3bbbaf1 is
  signal adder_0_s_net_x0: std_logic_vector(37 downto 0);
  signal adder_0_s_net_x3: std_logic_vector(36 downto 0);
  signal ce_1_sg_x136: std_logic;
  signal clk_1_sg_x136: std_logic;
  signal d2_x0: std_logic_vector(36 downto 0);

begin
  ce_1_sg_x136 <= ce_1;
  clk_1_sg_x136 <= clk_1;
  adder_0_s_net_x3 <= din_1;
  d2_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_fa63be3dfe
    port map (
      a => d2_x0,
      b => adder_0_s_net_x3,
      ce => ce_1_sg_x136,
      clk => clk_1_sg_x136,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage4_dec2/p_adder2"

entity p_adder2_entity_5994a06339 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(36 downto 0); 
    in2: in std_logic_vector(36 downto 0); 
    out1: out std_logic_vector(37 downto 0); 
    out2: out std_logic_vector(37 downto 0)
  );
end p_adder2_entity_5994a06339;

architecture structural of p_adder2_entity_5994a06339 is
  signal adder_0_s_net_x10: std_logic_vector(37 downto 0);
  signal adder_0_s_net_x7: std_logic_vector(36 downto 0);
  signal adder_0_s_net_x8: std_logic_vector(36 downto 0);
  signal adder_0_s_net_x9: std_logic_vector(37 downto 0);
  signal ce_1_sg_x138: std_logic;
  signal clk_1_sg_x138: std_logic;
  signal d2_x0: std_logic_vector(36 downto 0);

begin
  ce_1_sg_x138 <= ce_1;
  clk_1_sg_x138 <= clk_1;
  adder_0_s_net_x7 <= in1;
  adder_0_s_net_x8 <= in2;
  out1 <= adder_0_s_net_x9;
  out2 <= adder_0_s_net_x10;

  adder_tree1_7dd3bbbaf1: entity work.adder_tree1_entity_7dd3bbbaf1
    port map (
      ce_1 => ce_1_sg_x138,
      clk_1 => clk_1_sg_x138,
      din_1 => adder_0_s_net_x7,
      din_2 => d2_x0,
      dout => adder_0_s_net_x9
    );

  adder_tree2_6e12bd810f: entity work.adder_tree1_entity_7dd3bbbaf1
    port map (
      ce_1 => ce_1_sg_x138,
      clk_1 => clk_1_sg_x138,
      din_1 => adder_0_s_net_x8,
      din_2 => adder_0_s_net_x7,
      dout => adder_0_s_net_x10
    );

  delay2: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 37
    )
    port map (
      ce => ce_1_sg_x138,
      clk => clk_1_sg_x138,
      d => adder_0_s_net_x8,
      en => '1',
      rst => '1',
      q => d2_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage4_dec2/p_adder3/adder_tree1"

entity adder_tree1_entity_4cd7ebfdc8 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(37 downto 0); 
    din_2: in std_logic_vector(37 downto 0); 
    dout: out std_logic_vector(38 downto 0)
  );
end adder_tree1_entity_4cd7ebfdc8;

architecture structural of adder_tree1_entity_4cd7ebfdc8 is
  signal adder_0_s_net_x0: std_logic_vector(38 downto 0);
  signal adder_0_s_net_x10: std_logic_vector(37 downto 0);
  signal ce_1_sg_x139: std_logic;
  signal clk_1_sg_x139: std_logic;
  signal d2_x0: std_logic_vector(37 downto 0);

begin
  ce_1_sg_x139 <= ce_1;
  clk_1_sg_x139 <= clk_1;
  adder_0_s_net_x10 <= din_1;
  d2_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_8f8925e093
    port map (
      a => d2_x0,
      b => adder_0_s_net_x10,
      ce => ce_1_sg_x139,
      clk => clk_1_sg_x139,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage4_dec2/p_adder3"

entity p_adder3_entity_2993de1819 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(37 downto 0); 
    in2: in std_logic_vector(37 downto 0); 
    out1: out std_logic_vector(38 downto 0); 
    out2: out std_logic_vector(38 downto 0)
  );
end p_adder3_entity_2993de1819;

architecture structural of p_adder3_entity_2993de1819 is
  signal adder_0_s_net_x14: std_logic_vector(37 downto 0);
  signal adder_0_s_net_x15: std_logic_vector(37 downto 0);
  signal adder_0_s_net_x16: std_logic_vector(38 downto 0);
  signal adder_0_s_net_x17: std_logic_vector(38 downto 0);
  signal ce_1_sg_x141: std_logic;
  signal clk_1_sg_x141: std_logic;
  signal d2_x0: std_logic_vector(37 downto 0);

begin
  ce_1_sg_x141 <= ce_1;
  clk_1_sg_x141 <= clk_1;
  adder_0_s_net_x14 <= in1;
  adder_0_s_net_x15 <= in2;
  out1 <= adder_0_s_net_x16;
  out2 <= adder_0_s_net_x17;

  adder_tree1_4cd7ebfdc8: entity work.adder_tree1_entity_4cd7ebfdc8
    port map (
      ce_1 => ce_1_sg_x141,
      clk_1 => clk_1_sg_x141,
      din_1 => adder_0_s_net_x14,
      din_2 => d2_x0,
      dout => adder_0_s_net_x16
    );

  adder_tree2_48817cd60d: entity work.adder_tree1_entity_4cd7ebfdc8
    port map (
      ce_1 => ce_1_sg_x141,
      clk_1 => clk_1_sg_x141,
      din_1 => adder_0_s_net_x15,
      din_2 => adder_0_s_net_x14,
      dout => adder_0_s_net_x17
    );

  delay2: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 38
    )
    port map (
      ce => ce_1_sg_x141,
      clk => clk_1_sg_x141,
      d => adder_0_s_net_x15,
      en => '1',
      rst => '1',
      q => d2_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage4_dec2/p_adder4/adder_tree2"

entity adder_tree2_entity_fc900cfa40 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(38 downto 0); 
    din_2: in std_logic_vector(38 downto 0); 
    dout: out std_logic_vector(39 downto 0)
  );
end adder_tree2_entity_fc900cfa40;

architecture structural of adder_tree2_entity_fc900cfa40 is
  signal adder_0_s_net_x0: std_logic_vector(39 downto 0);
  signal adder_0_s_net_x18: std_logic_vector(38 downto 0);
  signal adder_0_s_net_x19: std_logic_vector(38 downto 0);
  signal ce_1_sg_x142: std_logic;
  signal clk_1_sg_x142: std_logic;

begin
  ce_1_sg_x142 <= ce_1;
  clk_1_sg_x142 <= clk_1;
  adder_0_s_net_x19 <= din_1;
  adder_0_s_net_x18 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_958f59c040
    port map (
      a => adder_0_s_net_x18,
      b => adder_0_s_net_x19,
      ce => ce_1_sg_x142,
      clk => clk_1_sg_x142,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage4_dec2/p_adder4"

entity p_adder4_entity_9a926c16f6 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(38 downto 0); 
    in2: in std_logic_vector(38 downto 0); 
    out2: out std_logic_vector(39 downto 0)
  );
end p_adder4_entity_9a926c16f6;

architecture structural of p_adder4_entity_9a926c16f6 is
  signal adder_0_s_net_x1: std_logic_vector(39 downto 0);
  signal adder_0_s_net_x20: std_logic_vector(38 downto 0);
  signal adder_0_s_net_x21: std_logic_vector(38 downto 0);
  signal ce_1_sg_x143: std_logic;
  signal clk_1_sg_x143: std_logic;

begin
  ce_1_sg_x143 <= ce_1;
  clk_1_sg_x143 <= clk_1;
  adder_0_s_net_x20 <= in1;
  adder_0_s_net_x21 <= in2;
  out2 <= adder_0_s_net_x1;

  adder_tree2_fc900cfa40: entity work.adder_tree2_entity_fc900cfa40
    port map (
      ce_1 => ce_1_sg_x143,
      clk_1 => clk_1_sg_x143,
      din_1 => adder_0_s_net_x21,
      din_2 => adder_0_s_net_x20,
      dout => adder_0_s_net_x1
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage4_dec2"

entity stage4_dec2_entity_2bbbcaeeb8 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(35 downto 0); 
    in2: in std_logic_vector(35 downto 0); 
    out1: out std_logic_vector(39 downto 0)
  );
end stage4_dec2_entity_2bbbcaeeb8;

architecture structural of stage4_dec2_entity_2bbbcaeeb8 is
  signal adder_0_s_net_x14: std_logic_vector(37 downto 0);
  signal adder_0_s_net_x15: std_logic_vector(37 downto 0);
  signal adder_0_s_net_x20: std_logic_vector(38 downto 0);
  signal adder_0_s_net_x21: std_logic_vector(38 downto 0);
  signal adder_0_s_net_x28: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x29: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x7: std_logic_vector(36 downto 0);
  signal adder_0_s_net_x8: std_logic_vector(36 downto 0);
  signal adder_0_s_net_x9: std_logic_vector(39 downto 0);
  signal ce_1_sg_x144: std_logic;
  signal clk_1_sg_x144: std_logic;

begin
  ce_1_sg_x144 <= ce_1;
  clk_1_sg_x144 <= clk_1;
  adder_0_s_net_x28 <= in1;
  adder_0_s_net_x29 <= in2;
  out1 <= adder_0_s_net_x9;

  p_adder1_bbc5e9a942: entity work.p_adder1_entity_bbc5e9a942
    port map (
      ce_1 => ce_1_sg_x144,
      clk_1 => clk_1_sg_x144,
      in1 => adder_0_s_net_x28,
      in2 => adder_0_s_net_x29,
      out1 => adder_0_s_net_x7,
      out2 => adder_0_s_net_x8
    );

  p_adder2_5994a06339: entity work.p_adder2_entity_5994a06339
    port map (
      ce_1 => ce_1_sg_x144,
      clk_1 => clk_1_sg_x144,
      in1 => adder_0_s_net_x7,
      in2 => adder_0_s_net_x8,
      out1 => adder_0_s_net_x14,
      out2 => adder_0_s_net_x15
    );

  p_adder3_2993de1819: entity work.p_adder3_entity_2993de1819
    port map (
      ce_1 => ce_1_sg_x144,
      clk_1 => clk_1_sg_x144,
      in1 => adder_0_s_net_x14,
      in2 => adder_0_s_net_x15,
      out1 => adder_0_s_net_x20,
      out2 => adder_0_s_net_x21
    );

  p_adder4_9a926c16f6: entity work.p_adder4_entity_9a926c16f6
    port map (
      ce_1 => ce_1_sg_x144,
      clk_1 => clk_1_sg_x144,
      in1 => adder_0_s_net_x20,
      in2 => adder_0_s_net_x21,
      out2 => adder_0_s_net_x9
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage5_dec2/Down_sample1"

entity down_sample1_entity_624e61f010 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in_x0: in std_logic_vector(43 downto 0); 
    sync_in: in std_logic; 
    out_x0: out std_logic_vector(43 downto 0); 
    sync_out: out std_logic
  );
end down_sample1_entity_624e61f010;

architecture structural of down_sample1_entity_624e61f010 is
  signal adder_0_s_net_x0: std_logic_vector(43 downto 0);
  signal ce_1_sg_x146: std_logic;
  signal clk_1_sg_x146: std_logic;
  signal register_q_net_x0: std_logic_vector(43 downto 0);
  signal sync_delay_q_net_x2: std_logic;
  signal sync_in_net_x1: std_logic;
  signal xlsub1_logical_out1_x0: std_logic;

begin
  ce_1_sg_x146 <= ce_1;
  clk_1_sg_x146 <= clk_1;
  adder_0_s_net_x0 <= in_x0;
  sync_in_net_x1 <= sync_in;
  out_x0 <= register_q_net_x0;
  sync_out <= sync_delay_q_net_x2;

  en_gen_9eb0979a23: entity work.en_gen_entity_f6c8ec71ed
    port map (
      ce_1 => ce_1_sg_x146,
      clk_1 => clk_1_sg_x146,
      sync_in => sync_in_net_x1,
      en_out => xlsub1_logical_out1_x0
    );

  register_x0: entity work.xlregister
    generic map (
      d_width => 44,
      init_value => b"00000000000000000000000000000000000000000000"
    )
    port map (
      ce => ce_1_sg_x146,
      clk => clk_1_sg_x146,
      d => adder_0_s_net_x0,
      en(0) => xlsub1_logical_out1_x0,
      rst => "0",
      q => register_q_net_x0
    );

  sync_delay: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 1
    )
    port map (
      ce => ce_1_sg_x146,
      clk => clk_1_sg_x146,
      d(0) => sync_in_net_x1,
      en => '1',
      rst => '1',
      q(0) => sync_delay_q_net_x2
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage5_dec2/adder_tree1"

entity adder_tree1_entity_31b1833c76 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_2: in std_logic_vector(39 downto 0); 
    sync: in std_logic_vector(39 downto 0); 
    dout: out std_logic_vector(40 downto 0)
  );
end adder_tree1_entity_31b1833c76;

architecture structural of adder_tree1_entity_31b1833c76 is
  signal adder_0_s_net_x0: std_logic_vector(40 downto 0);
  signal adder_0_s_net_x10: std_logic_vector(39 downto 0);
  signal ce_1_sg_x147: std_logic;
  signal clk_1_sg_x147: std_logic;
  signal delay_out1_2_x0: std_logic_vector(39 downto 0);

begin
  ce_1_sg_x147 <= ce_1;
  clk_1_sg_x147 <= clk_1;
  delay_out1_2_x0 <= din_2;
  adder_0_s_net_x10 <= sync;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_4b2650d0e6
    port map (
      a => delay_out1_2_x0,
      b => adder_0_s_net_x10,
      ce => ce_1_sg_x147,
      clk => clk_1_sg_x147,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage5_dec2/adder_tree2"

entity adder_tree2_entity_1691d44791 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(40 downto 0); 
    din_2: in std_logic_vector(40 downto 0); 
    dout: out std_logic_vector(41 downto 0)
  );
end adder_tree2_entity_1691d44791;

architecture structural of adder_tree2_entity_1691d44791 is
  signal adder_0_s_net_x1: std_logic_vector(40 downto 0);
  signal adder_0_s_net_x2: std_logic_vector(41 downto 0);
  signal ce_1_sg_x148: std_logic;
  signal clk_1_sg_x148: std_logic;
  signal delay_out2_2_x0: std_logic_vector(40 downto 0);

begin
  ce_1_sg_x148 <= ce_1;
  clk_1_sg_x148 <= clk_1;
  adder_0_s_net_x1 <= din_1;
  delay_out2_2_x0 <= din_2;
  dout <= adder_0_s_net_x2;

  adder_0: entity work.addsub_614a30f654
    port map (
      a => delay_out2_2_x0,
      b => adder_0_s_net_x1,
      ce => ce_1_sg_x148,
      clk => clk_1_sg_x148,
      clr => '0',
      s => adder_0_s_net_x2
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage5_dec2/adder_tree3"

entity adder_tree3_entity_3a13a2d488 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(41 downto 0); 
    din_2: in std_logic_vector(41 downto 0); 
    dout: out std_logic_vector(42 downto 0)
  );
end adder_tree3_entity_3a13a2d488;

architecture structural of adder_tree3_entity_3a13a2d488 is
  signal adder_0_s_net_x0: std_logic_vector(42 downto 0);
  signal adder_0_s_net_x3: std_logic_vector(41 downto 0);
  signal ce_1_sg_x149: std_logic;
  signal clk_1_sg_x149: std_logic;
  signal delay_out3_2_x0: std_logic_vector(41 downto 0);

begin
  ce_1_sg_x149 <= ce_1;
  clk_1_sg_x149 <= clk_1;
  adder_0_s_net_x3 <= din_1;
  delay_out3_2_x0 <= din_2;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_b829aecccd
    port map (
      a => delay_out3_2_x0,
      b => adder_0_s_net_x3,
      ce => ce_1_sg_x149,
      clk => clk_1_sg_x149,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage5_dec2/adder_tree4"

entity adder_tree4_entity_86755ab36c is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(42 downto 0); 
    din_2: in std_logic_vector(42 downto 0); 
    dout: out std_logic_vector(43 downto 0)
  );
end adder_tree4_entity_86755ab36c;

architecture structural of adder_tree4_entity_86755ab36c is
  signal adder_0_s_net_x2: std_logic_vector(42 downto 0);
  signal adder_0_s_net_x3: std_logic_vector(43 downto 0);
  signal ce_1_sg_x150: std_logic;
  signal clk_1_sg_x150: std_logic;
  signal delay_out4_2_x0: std_logic_vector(42 downto 0);

begin
  ce_1_sg_x150 <= ce_1;
  clk_1_sg_x150 <= clk_1;
  adder_0_s_net_x2 <= din_1;
  delay_out4_2_x0 <= din_2;
  dout <= adder_0_s_net_x3;

  adder_0: entity work.addsub_d65218df4d
    port map (
      a => delay_out4_2_x0,
      b => adder_0_s_net_x2,
      ce => ce_1_sg_x150,
      clk => clk_1_sg_x150,
      clr => '0',
      s => adder_0_s_net_x3
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1/Stage5_dec2"

entity stage5_dec2_entity_b7d1cdd330 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in_x0: in std_logic_vector(39 downto 0); 
    sync_in: in std_logic; 
    out_x0: out std_logic_vector(43 downto 0); 
    sync_out: out std_logic
  );
end stage5_dec2_entity_b7d1cdd330;

architecture structural of stage5_dec2_entity_b7d1cdd330 is
  signal adder_0_s_net_x1: std_logic_vector(40 downto 0);
  signal adder_0_s_net_x11: std_logic_vector(43 downto 0);
  signal adder_0_s_net_x12: std_logic_vector(39 downto 0);
  signal adder_0_s_net_x2: std_logic_vector(42 downto 0);
  signal adder_0_s_net_x3: std_logic_vector(41 downto 0);
  signal ce_1_sg_x151: std_logic;
  signal clk_1_sg_x151: std_logic;
  signal delay_out1_2_x0: std_logic_vector(39 downto 0);
  signal delay_out2_2_x0: std_logic_vector(40 downto 0);
  signal delay_out3_2_x0: std_logic_vector(41 downto 0);
  signal delay_out4_2_x0: std_logic_vector(42 downto 0);
  signal register_q_net_x1: std_logic_vector(43 downto 0);
  signal sync_delay_q_net_x3: std_logic;
  signal sync_in_net_x2: std_logic;

begin
  ce_1_sg_x151 <= ce_1;
  clk_1_sg_x151 <= clk_1;
  adder_0_s_net_x12 <= in_x0;
  sync_in_net_x2 <= sync_in;
  out_x0 <= register_q_net_x1;
  sync_out <= sync_delay_q_net_x3;

  adder_tree1_31b1833c76: entity work.adder_tree1_entity_31b1833c76
    port map (
      ce_1 => ce_1_sg_x151,
      clk_1 => clk_1_sg_x151,
      din_2 => delay_out1_2_x0,
      sync => adder_0_s_net_x12,
      dout => adder_0_s_net_x1
    );

  adder_tree2_1691d44791: entity work.adder_tree2_entity_1691d44791
    port map (
      ce_1 => ce_1_sg_x151,
      clk_1 => clk_1_sg_x151,
      din_1 => adder_0_s_net_x1,
      din_2 => delay_out2_2_x0,
      dout => adder_0_s_net_x3
    );

  adder_tree3_3a13a2d488: entity work.adder_tree3_entity_3a13a2d488
    port map (
      ce_1 => ce_1_sg_x151,
      clk_1 => clk_1_sg_x151,
      din_1 => adder_0_s_net_x3,
      din_2 => delay_out3_2_x0,
      dout => adder_0_s_net_x2
    );

  adder_tree4_86755ab36c: entity work.adder_tree4_entity_86755ab36c
    port map (
      ce_1 => ce_1_sg_x151,
      clk_1 => clk_1_sg_x151,
      din_1 => adder_0_s_net_x2,
      din_2 => delay_out4_2_x0,
      dout => adder_0_s_net_x11
    );

  delay1_1: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 40
    )
    port map (
      ce => ce_1_sg_x151,
      clk => clk_1_sg_x151,
      d => adder_0_s_net_x12,
      en => '1',
      rst => '1',
      q => delay_out1_2_x0
    );

  delay2_1: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 41
    )
    port map (
      ce => ce_1_sg_x151,
      clk => clk_1_sg_x151,
      d => adder_0_s_net_x1,
      en => '1',
      rst => '1',
      q => delay_out2_2_x0
    );

  delay3_1: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 42
    )
    port map (
      ce => ce_1_sg_x151,
      clk => clk_1_sg_x151,
      d => adder_0_s_net_x3,
      en => '1',
      rst => '1',
      q => delay_out3_2_x0
    );

  delay4_1: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 43
    )
    port map (
      ce => ce_1_sg_x151,
      clk => clk_1_sg_x151,
      d => adder_0_s_net_x2,
      en => '1',
      rst => '1',
      q => delay_out4_2_x0
    );

  down_sample1_624e61f010: entity work.down_sample1_entity_624e61f010
    port map (
      ce_1 => ce_1_sg_x151,
      clk_1 => clk_1_sg_x151,
      in_x0 => adder_0_s_net_x11,
      sync_in => sync_in_net_x2,
      out_x0 => register_q_net_x1,
      sync_out => sync_delay_q_net_x3
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_pow2_1"

entity cic_pow2_1_entity_ce4f7e0900 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in1: in std_logic_vector(23 downto 0); 
    in10: in std_logic_vector(23 downto 0); 
    in11: in std_logic_vector(23 downto 0); 
    in12: in std_logic_vector(23 downto 0); 
    in13: in std_logic_vector(23 downto 0); 
    in14: in std_logic_vector(23 downto 0); 
    in15: in std_logic_vector(23 downto 0); 
    in16: in std_logic_vector(23 downto 0); 
    in2: in std_logic_vector(23 downto 0); 
    in3: in std_logic_vector(23 downto 0); 
    in4: in std_logic_vector(23 downto 0); 
    in5: in std_logic_vector(23 downto 0); 
    in6: in std_logic_vector(23 downto 0); 
    in7: in std_logic_vector(23 downto 0); 
    in8: in std_logic_vector(23 downto 0); 
    in9: in std_logic_vector(23 downto 0); 
    sync_in: in std_logic; 
    out1: out std_logic_vector(43 downto 0); 
    sync_out: out std_logic
  );
end cic_pow2_1_entity_ce4f7e0900;

architecture structural of cic_pow2_1_entity_ce4f7e0900 is
  signal adder_0_s_net_x12: std_logic_vector(39 downto 0);
  signal adder_0_s_net_x129: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x130: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x150: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x151: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x153: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x159: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x161: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x20: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x28: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x29: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x32: std_logic_vector(31 downto 0);
  signal adder_0_s_net_x81: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x82: std_logic_vector(27 downto 0);
  signal adder_0_s_net_x83: std_logic_vector(27 downto 0);
  signal ce_1_sg_x152: std_logic;
  signal clk_1_sg_x152: std_logic;
  signal in0_net_x4: std_logic_vector(23 downto 0);
  signal in10_net_x4: std_logic_vector(23 downto 0);
  signal in11_net_x4: std_logic_vector(23 downto 0);
  signal in12_net_x4: std_logic_vector(23 downto 0);
  signal in13_net_x4: std_logic_vector(23 downto 0);
  signal in14_net_x4: std_logic_vector(23 downto 0);
  signal in15_net_x3: std_logic_vector(23 downto 0);
  signal in1_net_x4: std_logic_vector(23 downto 0);
  signal in2_net_x4: std_logic_vector(23 downto 0);
  signal in3_net_x4: std_logic_vector(23 downto 0);
  signal in4_net_x4: std_logic_vector(23 downto 0);
  signal in5_net_x4: std_logic_vector(23 downto 0);
  signal in6_net_x4: std_logic_vector(23 downto 0);
  signal in7_net_x4: std_logic_vector(23 downto 0);
  signal in8_net_x4: std_logic_vector(23 downto 0);
  signal in9_net_x4: std_logic_vector(23 downto 0);
  signal register_q_net_x2: std_logic_vector(43 downto 0);
  signal sync_delay_q_net_x4: std_logic;
  signal sync_in_net_x3: std_logic;

begin
  ce_1_sg_x152 <= ce_1;
  clk_1_sg_x152 <= clk_1;
  in0_net_x4 <= in1;
  in9_net_x4 <= in10;
  in10_net_x4 <= in11;
  in11_net_x4 <= in12;
  in12_net_x4 <= in13;
  in13_net_x4 <= in14;
  in14_net_x4 <= in15;
  in15_net_x3 <= in16;
  in1_net_x4 <= in2;
  in2_net_x4 <= in3;
  in3_net_x4 <= in4;
  in4_net_x4 <= in5;
  in5_net_x4 <= in6;
  in6_net_x4 <= in7;
  in7_net_x4 <= in8;
  in8_net_x4 <= in9;
  sync_in_net_x3 <= sync_in;
  out1 <= register_q_net_x2;
  sync_out <= sync_delay_q_net_x4;

  stage1_dec2_3e8ab687de: entity work.stage1_dec2_entity_3e8ab687de
    port map (
      ce_1 => ce_1_sg_x152,
      clk_1 => clk_1_sg_x152,
      in1 => in0_net_x4,
      in10 => in9_net_x4,
      in11 => in10_net_x4,
      in12 => in11_net_x4,
      in13 => in12_net_x4,
      in14 => in13_net_x4,
      in15 => in14_net_x4,
      in16 => in15_net_x3,
      in2 => in1_net_x4,
      in3 => in2_net_x4,
      in4 => in3_net_x4,
      in5 => in4_net_x4,
      in6 => in5_net_x4,
      in7 => in6_net_x4,
      in8 => in7_net_x4,
      in9 => in8_net_x4,
      out1 => adder_0_s_net_x150,
      out2 => adder_0_s_net_x130,
      out3 => adder_0_s_net_x153,
      out4 => adder_0_s_net_x151,
      out5 => adder_0_s_net_x129,
      out6 => adder_0_s_net_x81,
      out7 => adder_0_s_net_x82,
      out8 => adder_0_s_net_x83
    );

  stage2_dec2_0b5dc5c0dd: entity work.stage2_dec2_entity_0b5dc5c0dd
    port map (
      ce_1 => ce_1_sg_x152,
      clk_1 => clk_1_sg_x152,
      in1 => adder_0_s_net_x150,
      in2 => adder_0_s_net_x130,
      in3 => adder_0_s_net_x153,
      in4 => adder_0_s_net_x151,
      in5 => adder_0_s_net_x129,
      in6 => adder_0_s_net_x81,
      in7 => adder_0_s_net_x82,
      in8 => adder_0_s_net_x83,
      out1 => adder_0_s_net_x159,
      out2 => adder_0_s_net_x161,
      out3 => adder_0_s_net_x20,
      out4 => adder_0_s_net_x32
    );

  stage3_dec2_4774a9beac: entity work.stage3_dec2_entity_4774a9beac
    port map (
      ce_1 => ce_1_sg_x152,
      clk_1 => clk_1_sg_x152,
      in1 => adder_0_s_net_x159,
      in2 => adder_0_s_net_x161,
      in3 => adder_0_s_net_x20,
      in4 => adder_0_s_net_x32,
      out1 => adder_0_s_net_x28,
      out2 => adder_0_s_net_x29
    );

  stage4_dec2_2bbbcaeeb8: entity work.stage4_dec2_entity_2bbbcaeeb8
    port map (
      ce_1 => ce_1_sg_x152,
      clk_1 => clk_1_sg_x152,
      in1 => adder_0_s_net_x28,
      in2 => adder_0_s_net_x29,
      out1 => adder_0_s_net_x12
    );

  stage5_dec2_b7d1cdd330: entity work.stage5_dec2_entity_b7d1cdd330
    port map (
      ce_1 => ce_1_sg_x152,
      clk_1 => clk_1_sg_x152,
      in_x0 => adder_0_s_net_x12,
      sync_in => sync_in_net_x3,
      out_x0 => register_q_net_x2,
      sync_out => sync_delay_q_net_x4
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec3/Down_sample1/en_gen"

entity en_gen_entity_81dc93b558 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    sync_in: in std_logic; 
    en_out: out std_logic
  );
end en_gen_entity_81dc93b558;

architecture structural of en_gen_entity_81dc93b558 is
  signal ce_1_sg_x153: std_logic;
  signal clk_1_sg_x153: std_logic;
  signal sync_delay_q_net_x0: std_logic;
  signal xlsub1_constant_out1: std_logic_vector(15 downto 0);
  signal xlsub1_counter1_out1: std_logic_vector(2 downto 0);
  signal xlsub1_logical_out1_x0: std_logic;
  signal xlsub1_relational_out1: std_logic;

begin
  ce_1_sg_x153 <= ce_1;
  clk_1_sg_x153 <= clk_1;
  sync_delay_q_net_x0 <= sync_in;
  en_out <= xlsub1_logical_out1_x0;

  constant_x0: entity work.constant_9f5572ba51
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      op => xlsub1_constant_out1
    );

  counter1: entity work.xlcounter_limit_cichbcic_core
    generic map (
      cnt_15_0 => 0,
      cnt_31_16 => 0,
      cnt_47_32 => 0,
      cnt_63_48 => 0,
      core_name0 => "cntr_11_0_6400a835b899f648",
      count_limited => 0,
      op_arith => xlUnsigned,
      op_width => 3
    )
    port map (
      ce => ce_1_sg_x153,
      clk => clk_1_sg_x153,
      clr => '0',
      en => "1",
      rst(0) => xlsub1_logical_out1_x0,
      op => xlsub1_counter1_out1
    );

  logical: entity work.logical_aacf6e1b0e
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      d0(0) => xlsub1_relational_out1,
      d1(0) => sync_delay_q_net_x0,
      y(0) => xlsub1_logical_out1_x0
    );

  relational: entity work.relational_0c602bf0fd
    port map (
      a => xlsub1_counter1_out1,
      b => xlsub1_constant_out1,
      ce => '0',
      clk => '0',
      clr => '0',
      op(0) => xlsub1_relational_out1
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec3/Down_sample1"

entity down_sample1_entity_fc13de9a72 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in_x0: in std_logic_vector(37 downto 0); 
    sync_in: in std_logic; 
    out_x0: out std_logic_vector(37 downto 0)
  );
end down_sample1_entity_fc13de9a72;

architecture structural of down_sample1_entity_fc13de9a72 is
  signal adder_0_s_net_x0: std_logic_vector(37 downto 0);
  signal ce_1_sg_x154: std_logic;
  signal clk_1_sg_x154: std_logic;
  signal register_q_net_x0: std_logic_vector(37 downto 0);
  signal sync_delay_q_net_x1: std_logic;
  signal xlsub1_logical_out1_x0: std_logic;

begin
  ce_1_sg_x154 <= ce_1;
  clk_1_sg_x154 <= clk_1;
  adder_0_s_net_x0 <= in_x0;
  sync_delay_q_net_x1 <= sync_in;
  out_x0 <= register_q_net_x0;

  en_gen_81dc93b558: entity work.en_gen_entity_81dc93b558
    port map (
      ce_1 => ce_1_sg_x154,
      clk_1 => clk_1_sg_x154,
      sync_in => sync_delay_q_net_x1,
      en_out => xlsub1_logical_out1_x0
    );

  register_x0: entity work.xlregister
    generic map (
      d_width => 38,
      init_value => b"00000000000000000000000000000000000000"
    )
    port map (
      ce => ce_1_sg_x154,
      clk => clk_1_sg_x154,
      d => adder_0_s_net_x0,
      en(0) => xlsub1_logical_out1_x0,
      rst => "0",
      q => register_q_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec3/adder_tree1"

entity adder_tree1_entity_3c800f7509 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_2: in std_logic_vector(34 downto 0); 
    sync: in std_logic_vector(34 downto 0); 
    dout: out std_logic_vector(35 downto 0)
  );
end adder_tree1_entity_3c800f7509;

architecture structural of adder_tree1_entity_3c800f7509 is
  signal adder_0_s_net_x0: std_logic_vector(35 downto 0);
  signal ce_1_sg_x155: std_logic;
  signal clk_1_sg_x155: std_logic;
  signal delay_out1_2_x0: std_logic_vector(34 downto 0);
  signal register_q_net_x0: std_logic_vector(34 downto 0);

begin
  ce_1_sg_x155 <= ce_1;
  clk_1_sg_x155 <= clk_1;
  delay_out1_2_x0 <= din_2;
  register_q_net_x0 <= sync;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_b17f382b30
    port map (
      a => delay_out1_2_x0,
      b => register_q_net_x0,
      ce => ce_1_sg_x155,
      clk => clk_1_sg_x155,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec3/adder_tree2"

entity adder_tree2_entity_b339f32084 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(35 downto 0); 
    din_2: in std_logic_vector(35 downto 0); 
    dout: out std_logic_vector(36 downto 0)
  );
end adder_tree2_entity_b339f32084;

architecture structural of adder_tree2_entity_b339f32084 is
  signal adder_0_s_net_x1: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x2: std_logic_vector(36 downto 0);
  signal ce_1_sg_x156: std_logic;
  signal clk_1_sg_x156: std_logic;
  signal delay_out2_2_x0: std_logic_vector(35 downto 0);

begin
  ce_1_sg_x156 <= ce_1;
  clk_1_sg_x156 <= clk_1;
  adder_0_s_net_x1 <= din_1;
  delay_out2_2_x0 <= din_2;
  dout <= adder_0_s_net_x2;

  adder_0: entity work.addsub_1f7f5d69b1
    port map (
      a => delay_out2_2_x0,
      b => adder_0_s_net_x1,
      ce => ce_1_sg_x156,
      clk => clk_1_sg_x156,
      clr => '0',
      s => adder_0_s_net_x2
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec3/adder_tree3"

entity adder_tree3_entity_aa7fdc046c is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(36 downto 0); 
    din_2: in std_logic_vector(36 downto 0); 
    dout: out std_logic_vector(37 downto 0)
  );
end adder_tree3_entity_aa7fdc046c;

architecture structural of adder_tree3_entity_aa7fdc046c is
  signal adder_0_s_net_x3: std_logic_vector(36 downto 0);
  signal adder_0_s_net_x4: std_logic_vector(37 downto 0);
  signal ce_1_sg_x157: std_logic;
  signal clk_1_sg_x157: std_logic;
  signal delay_out3_2_x0: std_logic_vector(36 downto 0);

begin
  ce_1_sg_x157 <= ce_1;
  clk_1_sg_x157 <= clk_1;
  adder_0_s_net_x3 <= din_1;
  delay_out3_2_x0 <= din_2;
  dout <= adder_0_s_net_x4;

  adder_0: entity work.addsub_619c589516
    port map (
      a => delay_out3_2_x0,
      b => adder_0_s_net_x3,
      ce => ce_1_sg_x157,
      clk => clk_1_sg_x157,
      clr => '0',
      s => adder_0_s_net_x4
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec3"

entity cic_stage_dec3_entity_441469ff8f is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in_x0: in std_logic_vector(34 downto 0); 
    sync_in: in std_logic; 
    out_x0: out std_logic_vector(37 downto 0)
  );
end cic_stage_dec3_entity_441469ff8f;

architecture structural of cic_stage_dec3_entity_441469ff8f is
  signal adder_0_s_net_x1: std_logic_vector(35 downto 0);
  signal adder_0_s_net_x3: std_logic_vector(36 downto 0);
  signal adder_0_s_net_x4: std_logic_vector(37 downto 0);
  signal ce_1_sg_x158: std_logic;
  signal clk_1_sg_x158: std_logic;
  signal delay_out1_2_x0: std_logic_vector(34 downto 0);
  signal delay_out2_2_x0: std_logic_vector(35 downto 0);
  signal delay_out3_2_x0: std_logic_vector(36 downto 0);
  signal register_q_net_x2: std_logic_vector(34 downto 0);
  signal register_q_net_x3: std_logic_vector(37 downto 0);
  signal sync_delay_q_net_x2: std_logic;

begin
  ce_1_sg_x158 <= ce_1;
  clk_1_sg_x158 <= clk_1;
  register_q_net_x2 <= in_x0;
  sync_delay_q_net_x2 <= sync_in;
  out_x0 <= register_q_net_x3;

  adder_tree1_3c800f7509: entity work.adder_tree1_entity_3c800f7509
    port map (
      ce_1 => ce_1_sg_x158,
      clk_1 => clk_1_sg_x158,
      din_2 => delay_out1_2_x0,
      sync => register_q_net_x2,
      dout => adder_0_s_net_x1
    );

  adder_tree2_b339f32084: entity work.adder_tree2_entity_b339f32084
    port map (
      ce_1 => ce_1_sg_x158,
      clk_1 => clk_1_sg_x158,
      din_1 => adder_0_s_net_x1,
      din_2 => delay_out2_2_x0,
      dout => adder_0_s_net_x3
    );

  adder_tree3_aa7fdc046c: entity work.adder_tree3_entity_aa7fdc046c
    port map (
      ce_1 => ce_1_sg_x158,
      clk_1 => clk_1_sg_x158,
      din_1 => adder_0_s_net_x3,
      din_2 => delay_out3_2_x0,
      dout => adder_0_s_net_x4
    );

  delay1_1: entity work.xldelay
    generic map (
      latency => 4,
      reg_retiming => 0,
      reset => 0,
      width => 35
    )
    port map (
      ce => ce_1_sg_x158,
      clk => clk_1_sg_x158,
      d => register_q_net_x2,
      en => '1',
      rst => '1',
      q => delay_out1_2_x0
    );

  delay2_1: entity work.xldelay
    generic map (
      latency => 4,
      reg_retiming => 0,
      reset => 0,
      width => 36
    )
    port map (
      ce => ce_1_sg_x158,
      clk => clk_1_sg_x158,
      d => adder_0_s_net_x1,
      en => '1',
      rst => '1',
      q => delay_out2_2_x0
    );

  delay3_1: entity work.xldelay
    generic map (
      latency => 4,
      reg_retiming => 0,
      reset => 0,
      width => 37
    )
    port map (
      ce => ce_1_sg_x158,
      clk => clk_1_sg_x158,
      d => adder_0_s_net_x3,
      en => '1',
      rst => '1',
      q => delay_out3_2_x0
    );

  down_sample1_fc13de9a72: entity work.down_sample1_entity_fc13de9a72
    port map (
      ce_1 => ce_1_sg_x158,
      clk_1 => clk_1_sg_x158,
      in_x0 => adder_0_s_net_x4,
      sync_in => sync_delay_q_net_x2,
      out_x0 => register_q_net_x3
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec4/Down_sample1/en_gen"

entity en_gen_entity_0997d4c2e5 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    sync_in: in std_logic; 
    en_out: out std_logic
  );
end en_gen_entity_0997d4c2e5;

architecture structural of en_gen_entity_0997d4c2e5 is
  signal ce_1_sg_x159: std_logic;
  signal clk_1_sg_x159: std_logic;
  signal sync_in_net_x4: std_logic;
  signal xlsub1_constant_out1: std_logic_vector(15 downto 0);
  signal xlsub1_counter1_out1: std_logic_vector(1 downto 0);
  signal xlsub1_logical_out1_x0: std_logic;
  signal xlsub1_relational_out1: std_logic;

begin
  ce_1_sg_x159 <= ce_1;
  clk_1_sg_x159 <= clk_1;
  sync_in_net_x4 <= sync_in;
  en_out <= xlsub1_logical_out1_x0;

  constant_x0: entity work.constant_9f5572ba51
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      op => xlsub1_constant_out1
    );

  counter1: entity work.xlcounter_limit_cichbcic_core
    generic map (
      cnt_15_0 => 0,
      cnt_31_16 => 0,
      cnt_47_32 => 0,
      cnt_63_48 => 0,
      core_name0 => "cntr_11_0_a0d692c18ceb283e",
      count_limited => 0,
      op_arith => xlUnsigned,
      op_width => 2
    )
    port map (
      ce => ce_1_sg_x159,
      clk => clk_1_sg_x159,
      clr => '0',
      en => "1",
      rst(0) => xlsub1_logical_out1_x0,
      op => xlsub1_counter1_out1
    );

  logical: entity work.logical_aacf6e1b0e
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      d0(0) => xlsub1_relational_out1,
      d1(0) => sync_in_net_x4,
      y(0) => xlsub1_logical_out1_x0
    );

  relational: entity work.relational_fda93ba512
    port map (
      a => xlsub1_counter1_out1,
      b => xlsub1_constant_out1,
      ce => '0',
      clk => '0',
      clr => '0',
      op(0) => xlsub1_relational_out1
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec4/Down_sample1"

entity down_sample1_entity_ccf3fb6119 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in_x0: in std_logic_vector(34 downto 0); 
    sync_in: in std_logic; 
    out_x0: out std_logic_vector(34 downto 0); 
    sync_out: out std_logic
  );
end down_sample1_entity_ccf3fb6119;

architecture structural of down_sample1_entity_ccf3fb6119 is
  signal adder_0_s_net_x0: std_logic_vector(34 downto 0);
  signal ce_1_sg_x160: std_logic;
  signal clk_1_sg_x160: std_logic;
  signal register_q_net_x3: std_logic_vector(34 downto 0);
  signal sync_delay_q_net_x3: std_logic;
  signal sync_in_net_x5: std_logic;
  signal xlsub1_logical_out1_x0: std_logic;

begin
  ce_1_sg_x160 <= ce_1;
  clk_1_sg_x160 <= clk_1;
  adder_0_s_net_x0 <= in_x0;
  sync_in_net_x5 <= sync_in;
  out_x0 <= register_q_net_x3;
  sync_out <= sync_delay_q_net_x3;

  en_gen_0997d4c2e5: entity work.en_gen_entity_0997d4c2e5
    port map (
      ce_1 => ce_1_sg_x160,
      clk_1 => clk_1_sg_x160,
      sync_in => sync_in_net_x5,
      en_out => xlsub1_logical_out1_x0
    );

  register_x0: entity work.xlregister
    generic map (
      d_width => 35,
      init_value => b"00000000000000000000000000000000000"
    )
    port map (
      ce => ce_1_sg_x160,
      clk => clk_1_sg_x160,
      d => adder_0_s_net_x0,
      en(0) => xlsub1_logical_out1_x0,
      rst => "0",
      q => register_q_net_x3
    );

  sync_delay: entity work.xldelay
    generic map (
      latency => 1,
      reg_retiming => 0,
      reset => 0,
      width => 1
    )
    port map (
      ce => ce_1_sg_x160,
      clk => clk_1_sg_x160,
      d(0) => sync_in_net_x5,
      en => '1',
      rst => '1',
      q(0) => sync_delay_q_net_x3
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec4/adder_tree1"

entity adder_tree1_entity_9d526316a6 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_2: in std_logic_vector(31 downto 0); 
    sync: in std_logic_vector(31 downto 0); 
    dout: out std_logic_vector(32 downto 0)
  );
end adder_tree1_entity_9d526316a6;

architecture structural of adder_tree1_entity_9d526316a6 is
  signal adder_0_s_net_x0: std_logic_vector(32 downto 0);
  signal ce_1_sg_x161: std_logic;
  signal clk_1_sg_x161: std_logic;
  signal convert5_dout_net_x0: std_logic_vector(31 downto 0);
  signal delay_out1_2_x0: std_logic_vector(31 downto 0);

begin
  ce_1_sg_x161 <= ce_1;
  clk_1_sg_x161 <= clk_1;
  delay_out1_2_x0 <= din_2;
  convert5_dout_net_x0 <= sync;
  dout <= adder_0_s_net_x0;

  adder_0: entity work.addsub_38883d04b4
    port map (
      a => delay_out1_2_x0,
      b => convert5_dout_net_x0,
      ce => ce_1_sg_x161,
      clk => clk_1_sg_x161,
      clr => '0',
      s => adder_0_s_net_x0
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec4/adder_tree2"

entity adder_tree2_entity_ea67e82413 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(32 downto 0); 
    din_2: in std_logic_vector(32 downto 0); 
    dout: out std_logic_vector(33 downto 0)
  );
end adder_tree2_entity_ea67e82413;

architecture structural of adder_tree2_entity_ea67e82413 is
  signal adder_0_s_net_x1: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x2: std_logic_vector(33 downto 0);
  signal ce_1_sg_x162: std_logic;
  signal clk_1_sg_x162: std_logic;
  signal delay_out2_2_x0: std_logic_vector(32 downto 0);

begin
  ce_1_sg_x162 <= ce_1;
  clk_1_sg_x162 <= clk_1;
  adder_0_s_net_x1 <= din_1;
  delay_out2_2_x0 <= din_2;
  dout <= adder_0_s_net_x2;

  adder_0: entity work.addsub_20d573ab12
    port map (
      a => delay_out2_2_x0,
      b => adder_0_s_net_x1,
      ce => ce_1_sg_x162,
      clk => clk_1_sg_x162,
      clr => '0',
      s => adder_0_s_net_x2
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec4/adder_tree3"

entity adder_tree3_entity_dcacedd84b is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    din_1: in std_logic_vector(33 downto 0); 
    din_2: in std_logic_vector(33 downto 0); 
    dout: out std_logic_vector(34 downto 0)
  );
end adder_tree3_entity_dcacedd84b;

architecture structural of adder_tree3_entity_dcacedd84b is
  signal adder_0_s_net_x3: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x4: std_logic_vector(34 downto 0);
  signal ce_1_sg_x163: std_logic;
  signal clk_1_sg_x163: std_logic;
  signal delay_out3_2_x0: std_logic_vector(33 downto 0);

begin
  ce_1_sg_x163 <= ce_1;
  clk_1_sg_x163 <= clk_1;
  adder_0_s_net_x3 <= din_1;
  delay_out3_2_x0 <= din_2;
  dout <= adder_0_s_net_x4;

  adder_0: entity work.addsub_a3fafab502
    port map (
      a => delay_out3_2_x0,
      b => adder_0_s_net_x3,
      ce => ce_1_sg_x163,
      clk => clk_1_sg_x163,
      clr => '0',
      s => adder_0_s_net_x4
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core/cic_stage_dec4"

entity cic_stage_dec4_entity_b6f00e3969 is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in_x0: in std_logic_vector(31 downto 0); 
    sync_in: in std_logic; 
    out_x0: out std_logic_vector(34 downto 0); 
    sync_out: out std_logic
  );
end cic_stage_dec4_entity_b6f00e3969;

architecture structural of cic_stage_dec4_entity_b6f00e3969 is
  signal adder_0_s_net_x1: std_logic_vector(32 downto 0);
  signal adder_0_s_net_x3: std_logic_vector(33 downto 0);
  signal adder_0_s_net_x4: std_logic_vector(34 downto 0);
  signal ce_1_sg_x164: std_logic;
  signal clk_1_sg_x164: std_logic;
  signal convert5_dout_net_x1: std_logic_vector(31 downto 0);
  signal delay_out1_2_x0: std_logic_vector(31 downto 0);
  signal delay_out2_2_x0: std_logic_vector(32 downto 0);
  signal delay_out3_2_x0: std_logic_vector(33 downto 0);
  signal register_q_net_x4: std_logic_vector(34 downto 0);
  signal sync_delay_q_net_x4: std_logic;
  signal sync_in_net_x6: std_logic;

begin
  ce_1_sg_x164 <= ce_1;
  clk_1_sg_x164 <= clk_1;
  convert5_dout_net_x1 <= in_x0;
  sync_in_net_x6 <= sync_in;
  out_x0 <= register_q_net_x4;
  sync_out <= sync_delay_q_net_x4;

  adder_tree1_9d526316a6: entity work.adder_tree1_entity_9d526316a6
    port map (
      ce_1 => ce_1_sg_x164,
      clk_1 => clk_1_sg_x164,
      din_2 => delay_out1_2_x0,
      sync => convert5_dout_net_x1,
      dout => adder_0_s_net_x1
    );

  adder_tree2_ea67e82413: entity work.adder_tree2_entity_ea67e82413
    port map (
      ce_1 => ce_1_sg_x164,
      clk_1 => clk_1_sg_x164,
      din_1 => adder_0_s_net_x1,
      din_2 => delay_out2_2_x0,
      dout => adder_0_s_net_x3
    );

  adder_tree3_dcacedd84b: entity work.adder_tree3_entity_dcacedd84b
    port map (
      ce_1 => ce_1_sg_x164,
      clk_1 => clk_1_sg_x164,
      din_1 => adder_0_s_net_x3,
      din_2 => delay_out3_2_x0,
      dout => adder_0_s_net_x4
    );

  delay1_1: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 32
    )
    port map (
      ce => ce_1_sg_x164,
      clk => clk_1_sg_x164,
      d => convert5_dout_net_x1,
      en => '1',
      rst => '1',
      q => delay_out1_2_x0
    );

  delay2_1: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 33
    )
    port map (
      ce => ce_1_sg_x164,
      clk => clk_1_sg_x164,
      d => adder_0_s_net_x1,
      en => '1',
      rst => '1',
      q => delay_out2_2_x0
    );

  delay3_1: entity work.xldelay
    generic map (
      latency => 2,
      reg_retiming => 0,
      reset => 0,
      width => 34
    )
    port map (
      ce => ce_1_sg_x164,
      clk => clk_1_sg_x164,
      d => adder_0_s_net_x3,
      en => '1',
      rst => '1',
      q => delay_out3_2_x0
    );

  down_sample1_ccf3fb6119: entity work.down_sample1_entity_ccf3fb6119
    port map (
      ce_1 => ce_1_sg_x164,
      clk_1 => clk_1_sg_x164,
      in_x0 => adder_0_s_net_x4,
      sync_in => sync_in_net_x6,
      out_x0 => register_q_net_x4,
      sync_out => sync_delay_q_net_x4
    );

end structural;
library IEEE;
use IEEE.std_logic_1164.all;
use work.conv_pkg.all;

-- Generated from Simulink block "cichbcic_core"

entity cichbcic_core is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    in0: in std_logic_vector(23 downto 0); 
    in1: in std_logic_vector(23 downto 0); 
    in10: in std_logic_vector(23 downto 0); 
    in11: in std_logic_vector(23 downto 0); 
    in12: in std_logic_vector(23 downto 0); 
    in13: in std_logic_vector(23 downto 0); 
    in14: in std_logic_vector(23 downto 0); 
    in15: in std_logic_vector(23 downto 0); 
    in2: in std_logic_vector(23 downto 0); 
    in3: in std_logic_vector(23 downto 0); 
    in4: in std_logic_vector(23 downto 0); 
    in5: in std_logic_vector(23 downto 0); 
    in6: in std_logic_vector(23 downto 0); 
    in7: in std_logic_vector(23 downto 0); 
    in8: in std_logic_vector(23 downto 0); 
    in9: in std_logic_vector(23 downto 0); 
    sync_in: in std_logic; 
    out_x0: out std_logic_vector(31 downto 0)
  );
end cichbcic_core;

architecture structural of cichbcic_core is
  attribute core_generation_info: string;
  attribute core_generation_info of structural : architecture is "cichbcic_core,sysgen_core,{clock_period=6.66670000,clocking=Clock_Enables,compilation=NGC_Netlist,sample_periods=1.00000000000,testbench=0,total_blocks=1695,xilinx_adder_subtracter_block=138,xilinx_arithmetic_relational_operator_block=4,xilinx_constant_block_block=13,xilinx_counter_block=4,xilinx_delay_block=177,xilinx_gateway_in_block=17,xilinx_gateway_out_block=1,xilinx_logical_block_block=4,xilinx_multiplier_block=9,xilinx_register_block=3,xilinx_system_generator_block=1,xilinx_type_converter_block=3,xilinx_type_reinterpreter_block=1,}";

  signal addsub_s_net_x11: std_logic_vector(51 downto 0);
  signal ce_1_sg_x165: std_logic;
  signal clk_1_sg_x165: std_logic;
  signal convert4_dout_net_x1: std_logic_vector(31 downto 0);
  signal convert5_dout_net_x1: std_logic_vector(31 downto 0);
  signal in0_net: std_logic_vector(23 downto 0);
  signal in10_net: std_logic_vector(23 downto 0);
  signal in11_net: std_logic_vector(23 downto 0);
  signal in12_net: std_logic_vector(23 downto 0);
  signal in13_net: std_logic_vector(23 downto 0);
  signal in14_net: std_logic_vector(23 downto 0);
  signal in15_net: std_logic_vector(23 downto 0);
  signal in1_net: std_logic_vector(23 downto 0);
  signal in2_net: std_logic_vector(23 downto 0);
  signal in3_net: std_logic_vector(23 downto 0);
  signal in4_net: std_logic_vector(23 downto 0);
  signal in5_net: std_logic_vector(23 downto 0);
  signal in6_net: std_logic_vector(23 downto 0);
  signal in7_net: std_logic_vector(23 downto 0);
  signal in8_net: std_logic_vector(23 downto 0);
  signal in9_net: std_logic_vector(23 downto 0);
  signal out_x0_net: std_logic_vector(31 downto 0);
  signal register_q_net_x2: std_logic_vector(43 downto 0);
  signal register_q_net_x3: std_logic_vector(37 downto 0);
  signal register_q_net_x4: std_logic_vector(34 downto 0);
  signal reinterpret2_output_port_net: std_logic_vector(43 downto 0);
  signal sync_delay_q_net_x4: std_logic;
  signal sync_delay_q_net_x5: std_logic;
  signal sync_in_net: std_logic;

begin
  ce_1_sg_x165 <= ce_1;
  clk_1_sg_x165 <= clk_1;
  in0_net <= in0;
  in1_net <= in1;
  in10_net <= in10;
  in11_net <= in11;
  in12_net <= in12;
  in13_net <= in13;
  in14_net <= in14;
  in15_net <= in15;
  in2_net <= in2;
  in3_net <= in3;
  in4_net <= in4;
  in5_net <= in5;
  in6_net <= in6;
  in7_net <= in7;
  in8_net <= in8;
  in9_net <= in9;
  sync_in_net <= sync_in;
  out_x0 <= out_x0_net;

  cic_pow2_1_ce4f7e0900: entity work.cic_pow2_1_entity_ce4f7e0900
    port map (
      ce_1 => ce_1_sg_x165,
      clk_1 => clk_1_sg_x165,
      in1 => in0_net,
      in10 => in9_net,
      in11 => in10_net,
      in12 => in11_net,
      in13 => in12_net,
      in14 => in13_net,
      in15 => in14_net,
      in16 => in15_net,
      in2 => in1_net,
      in3 => in2_net,
      in4 => in3_net,
      in5 => in4_net,
      in6 => in5_net,
      in7 => in6_net,
      in8 => in7_net,
      in9 => in8_net,
      sync_in => sync_in_net,
      out1 => register_q_net_x2,
      sync_out => sync_delay_q_net_x4
    );

  cic_stage_dec3_441469ff8f: entity work.cic_stage_dec3_entity_441469ff8f
    port map (
      ce_1 => ce_1_sg_x165,
      clk_1 => clk_1_sg_x165,
      in_x0 => register_q_net_x4,
      sync_in => sync_delay_q_net_x5,
      out_x0 => register_q_net_x3
    );

  cic_stage_dec4_b6f00e3969: entity work.cic_stage_dec4_entity_b6f00e3969
    port map (
      ce_1 => ce_1_sg_x165,
      clk_1 => clk_1_sg_x165,
      in_x0 => convert5_dout_net_x1,
      sync_in => sync_in_net,
      out_x0 => register_q_net_x4,
      sync_out => sync_delay_q_net_x5
    );

  convert4: entity work.xlconvert
    generic map (
      bool_conversion => 0,
      din_arith => 2,
      din_bin_pt => 30,
      din_width => 44,
      dout_arith => 2,
      dout_bin_pt => 18,
      dout_width => 32,
      latency => 1,
      overflow => xlWrap,
      quantization => xlTruncate
    )
    port map (
      ce => ce_1_sg_x165,
      clk => clk_1_sg_x165,
      clr => '0',
      din => reinterpret2_output_port_net,
      en => "1",
      dout => convert4_dout_net_x1
    );

  convert5: entity work.xlconvert
    generic map (
      bool_conversion => 0,
      din_arith => 2,
      din_bin_pt => 32,
      din_width => 52,
      dout_arith => 2,
      dout_bin_pt => 12,
      dout_width => 32,
      latency => 1,
      overflow => xlWrap,
      quantization => xlTruncate
    )
    port map (
      ce => ce_1_sg_x165,
      clk => clk_1_sg_x165,
      clr => '0',
      din => addsub_s_net_x11,
      en => "1",
      dout => convert5_dout_net_x1
    );

  convert6: entity work.xlconvert
    generic map (
      bool_conversion => 0,
      din_arith => 2,
      din_bin_pt => 12,
      din_width => 38,
      dout_arith => 2,
      dout_bin_pt => 6,
      dout_width => 32,
      latency => 1,
      overflow => xlWrap,
      quantization => xlTruncate
    )
    port map (
      ce => ce_1_sg_x165,
      clk => clk_1_sg_x165,
      clr => '0',
      din => register_q_net_x3,
      en => "1",
      dout => out_x0_net
    );

  halfband1_83ad1d234b: entity work.halfband1_entity_83ad1d234b
    port map (
      ce_1 => ce_1_sg_x165,
      clk_1 => clk_1_sg_x165,
      in_x0 => convert4_dout_net_x1,
      sync => sync_delay_q_net_x4,
      out_x0 => addsub_s_net_x11
    );

  reinterpret2: entity work.reinterpret_41e5bd5e40
    port map (
      ce => '0',
      clk => '0',
      clr => '0',
      input_port => register_q_net_x2,
      output_port => reinterpret2_output_port_net
    );

end structural;

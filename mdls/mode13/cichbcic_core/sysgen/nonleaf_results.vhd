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

library IEEE;
use IEEE.std_logic_1164.all;

entity pfbfirr_core is
  port (
    ce_1: in std_logic; 
    clk_1: in std_logic; 
    pol1_in1: in std_logic_vector(7 downto 0); 
    pol1_in10: in std_logic_vector(7 downto 0); 
    pol1_in11: in std_logic_vector(7 downto 0); 
    pol1_in12: in std_logic_vector(7 downto 0); 
    pol1_in13: in std_logic_vector(7 downto 0); 
    pol1_in14: in std_logic_vector(7 downto 0); 
    pol1_in15: in std_logic_vector(7 downto 0); 
    pol1_in16: in std_logic_vector(7 downto 0); 
    pol1_in2: in std_logic_vector(7 downto 0); 
    pol1_in3: in std_logic_vector(7 downto 0); 
    pol1_in4: in std_logic_vector(7 downto 0); 
    pol1_in5: in std_logic_vector(7 downto 0); 
    pol1_in6: in std_logic_vector(7 downto 0); 
    pol1_in7: in std_logic_vector(7 downto 0); 
    pol1_in8: in std_logic_vector(7 downto 0); 
    pol1_in9: in std_logic_vector(7 downto 0); 
    sync_in: in std_logic; 
    pol1_out1: out std_logic_vector(17 downto 0); 
    pol1_out10: out std_logic_vector(17 downto 0); 
    pol1_out11: out std_logic_vector(17 downto 0); 
    pol1_out12: out std_logic_vector(17 downto 0); 
    pol1_out13: out std_logic_vector(17 downto 0); 
    pol1_out14: out std_logic_vector(17 downto 0); 
    pol1_out15: out std_logic_vector(17 downto 0); 
    pol1_out16: out std_logic_vector(17 downto 0); 
    pol1_out2: out std_logic_vector(17 downto 0); 
    pol1_out3: out std_logic_vector(17 downto 0); 
    pol1_out4: out std_logic_vector(17 downto 0); 
    pol1_out5: out std_logic_vector(17 downto 0); 
    pol1_out6: out std_logic_vector(17 downto 0); 
    pol1_out7: out std_logic_vector(17 downto 0); 
    pol1_out8: out std_logic_vector(17 downto 0); 
    pol1_out9: out std_logic_vector(17 downto 0); 
    sync_out: out std_logic
  );
end pfbfirr_core;

architecture structural of pfbfirr_core is
begin
end structural;


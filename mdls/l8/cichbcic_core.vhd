library IEEE;
use IEEE.std_logic_1164.all;

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
begin
end structural;


----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Design Name: 
-- Module Name: 
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx primitives in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity cichbcic_core_stub is
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
end cichbcic_core_stub;

architecture Behavioral of cichbcic_core_stub is

  component cichbcic_core
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
  end component;
begin

cichbcic_core_i : cichbcic_core
  port map (
    ce_1 => ce_1,
    clk_1 => clk_1,
    in0 => in0,
    in1 => in1,
    in10 => in10,
    in11 => in11,
    in12 => in12,
    in13 => in13,
    in14 => in14,
    in15 => in15,
    in2 => in2,
    in3 => in3,
    in4 => in4,
    in5 => in5,
    in6 => in6,
    in7 => in7,
    in8 => in8,
    in9 => in9,
    sync_in => sync_in,
    out_x0 => out_x0);
end Behavioral;


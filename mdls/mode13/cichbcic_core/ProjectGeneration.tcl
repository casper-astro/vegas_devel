#
# Created by System Generator     Tue Apr  9 09:26:54 2013
#
# Note: This file is produced automatically, and will be overwritten the next
# time you press "Generate" in System Generator.
#

namespace eval ::xilinx::dsptool::iseproject::param {
    set SynthStrategyName {XST Defaults*}
    set ImplStrategyName {ISE Defaults*}
    set Compilation {NGC Netlist}
    set Project {cichbcic_core_cw}
    set Family {Virtex6}
    set Device {xc6vsx475t}
    set Package {ff1759}
    set Speed {-1}
    set HDLLanguage {vhdl}
    set SynthesisTool {XST}
    set Simulator {Modelsim-SE}
    set ReadCores {False}
    set MapEffortLevel {High}
    set ParEffortLevel {High}
    set Frequency {149.99925000375}
    set NewXSTParser {1}
    set ProjectFiles {
        {{cichbcic_core_cw.vhd} -view All}
        {{cichbcic_core.vhd} -view All}
        {{cichbcic_core_cw.ucf}}
        {{cichbcic_core_cw.xdc}}
        {{/tools/casper/projects/hong/my_vegas_devel/mode13/cichbcic_core.mdl}}
    }
    set TopLevelModule {cichbcic_core_cw}
    set SynthesisConstraintsFile {cichbcic_core_cw.xcf}
    set ImplementationStopView {Structural}
    set ProjectGenerator {SysgenDSP}
}
    source SgIseProject.tcl
    ::xilinx::dsptool::iseproject::create

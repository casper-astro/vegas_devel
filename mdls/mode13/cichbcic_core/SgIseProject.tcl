set exename [info nameofexecutable]
set currenttclshell [file tail [ file rootname $exename ] ]
if { ! [string match "xtclsh" $currenttclshell] } {
    error "ERROR: Please run xtclsh."
    return
}

namespace eval ::xilinx::dsptool::iseproject {

    namespace eval ise {}
    namespace export \
        VERBOSITY_QUIET VERBOSITY_ERROR VERBOSITY_WARNING \
        VERBOSITY_INFORMATION VERBOSITY_DEBUG

    set VERBOSITY_QUIET       0
    set VERBOSITY_ERROR       1
    set VERBOSITY_WARNING     2
    set VERBOSITY_INFORMATION 3
    set VERBOSITY_DEBUG       4

    #-------------------------------------------------------------------------
    # Checks for a required parameter.
    #
    # @param  param          Parameter name.
    # @param  postproc       Post processor.
    # @return the parameter value.
    #-------------------------------------------------------------------------
    proc required_parameter {param {postproc ""}} {
        upvar $param p
        if {![info exists p]} {
            error "Required parameter \"[namespace tail $param]\" is not specified."
        }
        if {$postproc != ""} {
            eval $postproc p
        }
        return $p
    }

    #-------------------------------------------------------------------------
    # Checks for an optional parameter.
    #
    # @param  param          Parameter name.
    # @param  defval         Default value of the parameter if unspecified.
    # @param  postproc       Post processor.
    # @return the parameter value.
    #-------------------------------------------------------------------------
    proc optional_parameter {param {defval ""} {postproc ""}} {
        upvar $param p
        if {![info exists p]} {
            set p $defval
        }
        if {$postproc != ""} {
            eval $postproc p
        }
        return $p
    }

    #-------------------------------------------------------------------------
    # Deletes an existing empty parameter.
    #
    # @param  param          Parameter name.
    #-------------------------------------------------------------------------
    proc clear_empty_parameter {param} {
        upvar $param p
        if {[info exists p] && [expr { [string length $p] == 0 }]} {
            unset p
        }
    }

    #-------------------------------------------------------------------------
    # Checks a Boolean flag.
    #
    # @param  param          Parameter name.
    # @param  defval         Default value of the parameter if unspecified.
    # @return 1 if the flag is specified and is true, or 0 othewise.
    #-------------------------------------------------------------------------
    proc check_flag {param {defval ""}} {
        upvar $param p
        return [expr { [info exists p] && $p }]
    }

    #-------------------------------------------------------------------------
    # Tests if the current verbosity level is equal to or
    # greater than the target verbosity level.
    #
    # @param  level          Target verbosity level.
    # @return True if the current verbosity level is equal to or
    #         greater than the target verbosity level.
    #-------------------------------------------------------------------------
    proc meet_verbosity {level} {
        set curr_level [subst $[namespace current]::$level]
        return [expr { $param::_VERBOSITY >= $curr_level }]
    }

    #-------------------------------------------------------------------------
    # Post processor to turn the given parameter to lower case.
    #
    # @param  param          Parameter name.
    # @return the processed parameter value.
    #-------------------------------------------------------------------------
    proc lowercase_pp {param} {
        upvar $param p
        set p [string tolower $p]
        return $p
    }

    #-------------------------------------------------------------------------
    # Post processor for the SynthesisTool parameter.
    #
    # @param  param          Parameter name.
    # @return the processed parameter value.
    #-------------------------------------------------------------------------
    proc synthesis_tool_pp {param} {
        upvar $param p
        switch [string tolower $p] {
            "xst" {
                set p "XST"
            }
            "synplify" {
                set p "Synplify"
            }
            "synplify pro" {
                set p "Synplify Pro"
            }
            default {
                error "Invalid value for parameter \"SynthesisTool\": $p"
            }
        }
    }

    #-------------------------------------------------------------------------
    # Post processor for the HDLLanguage parameter.
    #
    # @param  param          Parameter name.
    # @return the processed parameter value.
    #-------------------------------------------------------------------------
    proc hdl_language_pp {param} {
        upvar $param p
        switch [string tolower $p] {
            "vhdl" {
                set p "VHDL"
            }
            "verilog" {
                set p "Verilog"
            }
            default {
                error "Invalid value for parameter \"HDLLanguage\": $p"
            }
        }
    }

    #-------------------------------------------------------------------------
    # Dumps all variables of a given namespace. The current namespace is used
    # if no namespace is specified.
    #
    # @param  ns             Target namespace.
    #-------------------------------------------------------------------------
    proc dump_variables {{ns ""}} {
        if {$ns eq ""} {
            set ns [namespace current]
        }
        foreach param [lsort [info vars $ns\::*]] {
            upvar $param p
            # TODO : print array, remove upvar
            puts [namespace tail $param]\ =\ $p
        }
    }

    #-------------------------------------------------------------------------
    # Obtains a new unique command name for the given command.
    #
    # @param  cmd            Fully qualified command name.
    # @return fully qualified name of the new command.
    #-------------------------------------------------------------------------
    proc unique_command_name {cmd} {
        upvar _unique_command_id_ id
        if {![info exists id]} {
            set id 0
        }

        set ns [namespace qualifiers $cmd]
        set old_name [namespace tail $cmd]
        set new_name "$old_name\_$id\_"
        set eval_ns [expr { $ns eq "" ? "::" : $ns }]
        while { [lsearch [namespace eval $eval_ns {info proc}] $new_name] >= 0 } {
            incr id
            set new_name "$old_name\_$id\_"
        }

        return "$ns\::$new_name"
    }

    #-------------------------------------------------------------------------
    # Decorates a command with the given decorator. Unless a new command name
    # is specified, the original command is renamed and then replaced by
    # the decorated command.
    #
    # @param  decorator      Fully qualified name of the decorator command.
    # @param  cmd            Fully qualified name of the command to be
    #                        decorated.
    # @param  new_cmd        Fully qualified name of the new command.
    #-------------------------------------------------------------------------
    proc decorate_command {decorator cmd {new_cmd ""}} {
        if {[expr {$new_cmd eq ""}] || [expr {$new_cmd eq $cmd}]} {
            set new_cmd [unique_command_name $cmd]
            set s "rename $cmd $new_cmd; \
                   proc $cmd {args} { \
                       return \[uplevel {$decorator} \[linsert \$args 0 {$cmd} {$new_cmd}\] \] \
                   };"
        } else {
            set s "proc $new_cmd {args} { \
                       return \[uplevel {$decorator} \[linsert \$args 0 {$new_cmd} {$cmd}\] \] \
                   };"
        }
        eval $s
    }

    #-------------------------------------------------------------------------
    # Decorator that logs a given command without execution.
    #
    # @param  invoked_cmd    Invoked command.
    # @param  actual_cmd     Actual command.
    # @param  args           Additional argument list.
    #-------------------------------------------------------------------------
    proc log_command {invoked_cmd actual_cmd args} {
        if [meet_verbosity VERBOSITY_INFORMATION] {
            set cmd "[namespace qualifiers $actual_cmd][namespace tail $actual_cmd]"
            puts "$cmd $args"
        }
    }

    #-------------------------------------------------------------------------
    # Decorator that executes a given command.
    #
    # @param  invoked_cmd    Invoked command.
    # @param  actual_cmd     Actual command.
    # @param  args           Additional argument list.
    # @return the command result.
    #-------------------------------------------------------------------------
    proc run_command {invoked_cmd actual_cmd args} {
        set cmd "[namespace qualifiers $actual_cmd][namespace tail $actual_cmd]"
        if [meet_verbosity VERBOSITY_INFORMATION] {
            puts "$cmd $args"
        }
        if [catch { uplevel $actual_cmd $args } result] {
            error "Failed to execute command \"$cmd $args\".\n$result"
        }
        return $result
    }

    #-------------------------------------------------------------------------
    # Decorates ISE commands with appropriate decorators.
    #-------------------------------------------------------------------------
    proc decorate_ise_commands {} {
        upvar _ise_commands_already_decorated_ decorated
        if [check_flag decorated] {
            return
        } else {
            set decorated True
        }

        set ise_cmd_list {
            ::collection
            ::lib_vhdl
            ::object
            ::partition
            ::process
            ::project
            ::xfile
        }
        if [check_flag param::_DRY_RUN] {
            set decorator [namespace current]::log_command
        } else {
            set decorator [namespace current]::run_command
        }
        foreach cmd $ise_cmd_list {
            set new_cmd "[namespace current]::ise::[namespace tail $cmd]"
            decorate_command $decorator $cmd $new_cmd
        }
    }

    #-------------------------------------------------------------------------
    # Handles an exception when evaluating the given script and displays an
    # appropriate error message.
    #
    # @param  script         Script to evaluate.
    # @param  msg            Message to display upon an exception.
    # @param  append_msg     Specifies whether any returned error message is
    #                        also displayed.
    # @return 1 if the script is evaluated successfully, or 0 othewise.
    #-------------------------------------------------------------------------
    proc handle_exception {script {msg ""} {append_msg True}} {
        if [catch { uplevel $script } result] {
            if {$msg eq ""} {
                set msg "An internal error occurred."
            }
            puts stderr "$msg"
            if {$append_msg} {
                puts stderr "\n$result"
            }
            return 0
        }
        return 1
    }

    #-------------------------------------------------------------------------
    # Processes all project parameters.
    #
    # REQUIRED PARAMETERS
    # ======================================================================
    #   Project
    #     ISE project name.
    #
    #   Family
    #     Device family into which the design is implemented.
    #
    #   Device
    #     Device into which the design is implemented.
    #
    #   Package
    #     Package for the device being targeted.
    #
    #   Speed
    #     Speed grade of the device being targeted.
    #
    #   ProjectFiles
    #     Source files to be added in the project.
    #
    #
    # OPTIONAL PARAMETERS
    # ======================================================================
    # (*) Notes:
    #     "::=" denotes the list of supported values for each parameter.
    #
    # ----------------------------------------------------------------------
    #
    #   CompilationFlow
    #     Compilation flow.
    #
    #   TopLevelModule
    #     Top-level module of the design.
    #
    #   HDLLanguage
    #     Preferred language property controls the default setting for
    #     process properties that generate HDL output.
    #       ::= "VHDL" | "Verilog"
    #
    #   SynthesisTool
    #     Synthesis tool used for the design.
    #       ::= "XST" | "Synplify" | "Synplify Pro"
    #
    #   SynthesisConstraintsFile
    #     Synthesis constraints file. XCF for XST,
    #     SDC for Synplify/Synplify Pro.
    #
    #   SynthesisRegisterBalancing
    #     Register balancing option of the Synthesis process.
    #
    #   SynthesisRegisterDuplication
    #     Register duplication option of the Synthesis process.
    #
    #   SynthesisRetiming
    #     Retiming option of the Synthesis process. Synplify Pro Only.
    #       ::= True | False
    #
    #   WriteTimingConstraints
    #     Specifies whether or not to place timing constraints in the NGC
    #     file.
    #       ::= True | False
    #
    #   WriteVendorConstraints
    #     Specifies whether or not to generate vendor constraints file.
    #       ::= True | False
    #
    #   ReadCores
    #     Specifies whether or not black box cores are read for timing
    #     and area estimation in order to get better optimization of
    #     the rest of the design.
    #       ::= True | False
    #
    #   InsertIOBuffers
    #     Specifies whether or not to infer input/output buffers on all
    #     top-level I/O ports of the design.
    #       ::= True | False
    #
    #   BusDelimiter
    #     Specifies the delimiter type used to define the signal vectors in
    #     the resulting netlist.
    #       ::= "<>" | "[]" | "{}" | "()"
    #
    #   HierarchySeparator
    #     Hierarchy separator character which will be used in name
    #     generation when the design hierarchy is flattened.
    #       ::= "/" | "_"
    #
    #   KeepHierarchy
    #     Specifies whether or not the corresponding design unit should be
    #     preserved and not merged with the rest of the design.
    #       ::= "Yes" | "No" | "Soft"
    #
    #   Frequency
    #     Global clock frequency for timing-driven synthesis.
    #
    #   FanoutLimit
    #     Maximum limit of the fanout of nets.
    #
    #   MapRegisterDuplication
    #     Register duplication option of the Map process.
    #
    #   MapEffortLevel
    #     Effort level of the Map process.
    #
    #   PAREffortLevel
    #     Effort level of the Place & Route process.
    #
    #   BlockMemoryMapFile
    #     Block memory map (.bmm) file for the Data2MEM process.
    #
    #   BlockMemoryContentFile
    #     Block memory content file for the Data2MEM process.
    #
    #   Simulator
    #     Tool used for simulation.
    #
    #   DesignInstance
    #     Design instance name.
    #
    #   TestBenchModule
    #     Test-bench module.
    #
    #   SimulationTime
    #     Simulation time.
    #
    #   BehavioralSimulationCustomDoFile
    #     Custom Do file for the Behavioral Simulation process.
    #
    #   PostTranslateSimulationCustomDoFile
    #     Custom Do file for the Post-Translate Simulation process.
    #
    #   PostMapSimulationCustomDoFile
    #     Custom Do file for the Post-Map Simulation process.
    #
    #   PostPARSimulationCustomDoFile
    #     Custom Do file for the Post-Place & Route Simulation process.
    #
    #   ISimCustomProjectFile
    #     Custom project file for ISE Simulator.
    #
    #   HasVerilogSource
    #     Indicate the project contains a Verilog source file.
    #
    #   ImplementationStopView
    #
    #   ProjectGenerator
    #
    #-------------------------------------------------------------------------
    proc process_parameters {} {
        optional_parameter param::_DRY_RUN False
        optional_parameter param::_VERBOSITY $[namespace current]::VERBOSITY_ERROR

        required_parameter param::Project
        required_parameter param::Family lowercase_pp
        required_parameter param::Device lowercase_pp
        required_parameter param::Package lowercase_pp
        required_parameter param::Speed
        required_parameter param::ProjectFiles

        optional_parameter param::CompilationFlow {general}
        optional_parameter param::HDLLanguage {VHDL} hdl_language_pp
        optional_parameter param::SynthesisTool {XST} synthesis_tool_pp
        optional_parameter param::SynthesisRegisterBalancing {No}
        optional_parameter param::SynthesisRegisterDuplication True
        optional_parameter param::SynthesisRetiming True
        optional_parameter param::WriteTimingConstraints False
        optional_parameter param::WriteVendorConstraints False
        optional_parameter param::ReadCores True
        optional_parameter param::InsertIOBuffers True
        set is_vhdl [expr { $param::HDLLanguage eq "VHDL" }]
        optional_parameter param::BusDelimiter [expr { $is_vhdl ? {()} : {[]} }]
        optional_parameter param::HierarchySeparator {/}
        optional_parameter param::KeepHierarchy {No}
        optional_parameter param::HasVerilogSource False
        optional_parameter param::MapRegisterDuplication True
        optional_parameter param::MapEffortLevel {High}
        optional_parameter param::PAREffortLevel {High}
        optional_parameter param::DesignInstance {sysgen_dut}

        clear_empty_parameter param::TopLevelModule
        clear_empty_parameter param::SynthesisConstraintsFile
        clear_empty_parameter param::Frequency
        clear_empty_parameter param::FanoutLimit
        clear_empty_parameter param::BlockMemoryMapFile
        clear_empty_parameter param::BlockMemoryContentFile
        clear_empty_parameter param::Simulator
        clear_empty_parameter param::TestBenchModule
        clear_empty_parameter param::BehavioralSimulationCustomDoFile
        clear_empty_parameter param::PostTranslateSimulationCustomDoFile
        clear_empty_parameter param::PostMapSimulationCustomDoFile
        clear_empty_parameter param::PostPARSimulationCustomDoFile
        clear_empty_parameter param::ISimCustomProjectFile
        clear_empty_parameter param::ProjectGenerator
        clear_empty_parameter param::ImplementationStopView
    }

    #-------------------------------------------------------------------------
    # Dumps all parameters.
    #-------------------------------------------------------------------------
    proc dump_parameters {} {
        if [meet_verbosity VERBOSITY_DEBUG] {
            dump_variables param
        }
    }

    #-------------------------------------------------------------------------
    # Adds source files to the project.
    #-------------------------------------------------------------------------
    proc add_project_files {} {
        foreach p $param::ProjectFiles {
            set filename [file normalize [lindex $p 0]]
            set opts [lrange $p 1 end]
            set nopts [llength $opts]
            if {$nopts % 2 != 0} {
                error "Parameter \"ProjectFiles\" contains an invalid value \"$p\"."
            }
            # Remember it if the project contains a Verilog source file.
            if [string match -nocase "*.v" $filename] {
                set param::HasVerilogSource True
            }
            set args [list ise::xfile add $filename]
            for {set i 0} {$i < $nopts} {set i [expr {$i + 2}]} {
                set key [lindex $opts $i]
                set val [lindex $opts [expr {$i + 1}]]
                switch -- $key {
                    "-lib" {
                        if {![info exists lib_list($val)]} {
                            set lib_list($val) True
                            ise::lib_vhdl new $val
                        }
                        lappend args "-lib_vhdl" $val
                    }
                    "-view" {
                        lappend args "-view" $val
                    }
                    default {
                        error "Parameter \"ProjectFiles\" contains an invalid value \"$p\". Unknown option \"$key\"."
                    }
                }
            }
            eval $args
        }
        if [info exists param::TopLevelModule] {
            ise::project set top "/$param::TopLevelModule"
        }
    }

    #-------------------------------------------------------------------------
    # Sets the general project settings.
    #-------------------------------------------------------------------------
    proc set_project_settings {} {
        ise::project set family $param::Family
        ise::project set device $param::Device
        ise::project set package $param::Package
        ise::project set speed $param::Speed
    }

    #-------------------------------------------------------------------------
    # Sets the synthesis settings for XST.
    #-------------------------------------------------------------------------
    proc set_xst_synthesis_settings {} {
        # XST specific properties
        ise::project set {Synthesis Tool} {XST (VHDL/Verilog)}
        ise::project set {Optimization Goal} {Speed}
        ise::project set {Optimization Effort} {Normal} -process {Synthesize - XST}
        ise::project set {Keep Hierarchy} $param::KeepHierarchy
        ise::project set {Bus Delimiter} $param::BusDelimiter
        ise::project set {Hierarchy Separator} $param::HierarchySeparator
        set read_cores [project get {Read Cores}]
        # TODO: Remove this check when ISE settles with the read core property value
        if {[string equal -nocase $read_cores "true"] || [string equal -nocase $read_cores "false"]} {
            ise::project set {Read Cores} $param::ReadCores
        } else {
            ise::project set {Read Cores} [ expr { $param::ReadCores ? "Yes" : "No" } ]
        }
        ise::project set {Add I/O Buffers} $param::InsertIOBuffers
        # ise::project set {Optimize Instantiated Primitives} True
        ise::project set {Register Balancing} $param::SynthesisRegisterBalancing
        ise::project set {Register Duplication} $param::SynthesisRegisterDuplication -process {Synthesize - XST}
        ise::project set {Write Timing Constraints} $param::WriteTimingConstraints
        if [info exists param::SynthesisConstraintsFile] {
            ise::project set {Use Synthesis Constraints File} True
            ise::project set {Synthesis Constraints File} $param::SynthesisConstraintsFile
        } else {
            ise::project set {Use Synthesis Constraints File} False
        }
        if [info exists param::FanoutLimit] {
            ise::project set {Max Fanout} $param::FanoutLimit
        }
    }

    #-------------------------------------------------------------------------
    # Sets the synthesis settings for Synplify/Synplify Pro.
    #-------------------------------------------------------------------------
    proc set_synplify_synthesis_settings {} {
        set is_vhdl [expr { $param::HDLLanguage eq "VHDL" }]

        switch $param::SynthesisTool {
            "Synplify" {
                if {$is_vhdl} {
                    ise::project set {Synthesis Tool} {Synplify (VHDL)}
                } else {
                    ise::project set {Synthesis Tool} {Synplify (Verilog)}
                }
            }
            "Synplify Pro" {
                ise::project set {Synthesis Tool} {Synplify Pro (VHDL/Verilog)}
                ise::project set {Retiming} $param::SynthesisRetiming -process {Synthesize - Synplify Pro}
            }
        }

        # Synplify/Synplify Pro specific properties
        ise::project set {Symbolic FSM Compiler} False
        ise::project set {Pipelining} False
        ise::project set {Resource Sharing} False
        ise::project set {Disable I/O insertion} [ expr { $param::InsertIOBuffers ? False : True } ]
        ise::project set {Auto Constrain} False
        if [info exists param::SynthesisConstraintsFile] {
            ise::project set {Constraint File Name} $param::SynthesisConstraintsFile
        }
        ise::project set {Write Vendor Constraint File} $param::WriteVendorConstraints
        if [info exists param::Frequency] {
            ise::project set {Frequency} $param::Frequency
        }
        if [info exists param::FanoutLimit] {
            ise::project set {Fanout Guide} $param::FanoutLimit
        }
    }

    #-------------------------------------------------------------------------
    # Sets the synthesis settings.
    #-------------------------------------------------------------------------
    proc set_synthesis_settings {} {
        ise::project set {Preferred Language} $param::HDLLanguage

        switch -- $param::SynthesisTool {
            "XST" {
                set_xst_synthesis_settings
            }
            "Synplify" - "Synplify Pro" {
                set_synplify_synthesis_settings
            }
        }
    }

    #-------------------------------------------------------------------------
    # Sets the implementation settings.
    #-------------------------------------------------------------------------
    proc set_implementation_settings {} {
        # Translate properties
        ise::project set {Netlist Translation Type} {Timestamp}
        ise::project set {Use LOC Constraints} True
        if [info exists param::BlockMemoryMapFile] {
            ise::project set {Other Ngdbuild Command Line Options} "-bm $param::BlockMemoryMapFile"
        }

        # Determine the type of value the "Map Register Duplication" property accepts
        switch -- $param::Family {
            "virtex" - "virtexe" - "spartan2" - "spartan2e" {
            }
            default {
                set map_reg_dup [project get {Register Duplication} -process {Map}]
                if {[string equal -nocase $map_reg_dup "true"] || [string equal -nocase $map_reg_dup "false"]} {
                    set map_reg_dup $param::MapRegisterDuplication
                } elseif {[string equal -nocase $map_reg_dup "on"] || [string equal -nocase $map_reg_dup "off"]} {
                    set map_reg_dup [ expr { $param::MapRegisterDuplication ? "On" : "Off" } ]
                } else {
                    set map_reg_dup [ expr { $param::MapRegisterDuplication ? "Yes" : "No" } ]
                }
            }
        }

        # Map properties
        switch -glob -- $param::Family {
            "*virtex4*" - "*spartan3*" {
                ise::project set {Map Effort Level} $param::MapEffortLevel
                ise::project set {Perform Timing-Driven Packing and Placement} True
                ise::project set {Register Duplication} $map_reg_dup -process {Map}
            }
            "virtex" - "virtexe" - "spartan2" - "spartan2e" {
                ise::project set {Perform Timing-Driven Packing} True
            }
            default {
                ise::project set {Placer Effort Level} $param::MapEffortLevel
                ise::project set {Register Duplication} $map_reg_dup -process {Map}
            }
        }

        # Place & Route properties
        ise::project set {Place & Route Effort Level (Overall)} $param::PAREffortLevel
    }

    #-------------------------------------------------------------------------
    # Sets the configuration settings
    #-------------------------------------------------------------------------
    proc set_configuration_settings {} {
        switch -- $param::CompilationFlow {
            "hwcosim" {
                ise::project set {FPGA Start-Up Clock} {JTAG Clock}
                ise::project set {Drive Done Pin High} True
                switch -- $param::Family {
                    "virtex2" - "virtex2p" - "virtex4" - "virtex5" {
                        ise::project set {Configuration Pin M0} {Pull Up}
                        ise::project set {Configuration Pin M1} {Pull Down}
                        ise::project set {Configuration Pin M2} {Pull Up}
                    }
                }
            }
        }
        if [info exists param::BlockMemoryContentFile] {
            ise::project set {Other Bitgen Command Line Options} "-bd $param::BlockMemoryContentFile"
        }
    }

    #-------------------------------------------------------------------------
    # Sets the simulation settings
    #-------------------------------------------------------------------------
    proc set_simulation_settings {} {
        set has_testbench [info exists param::TestBenchModule]
        if {!$has_testbench} { return }

        set has_simtime [info exists param::SimulationTime]

        # ISE Simulator settings
        ise::project set {Simulator} "ISim (VHDL/Verilog)"

        set sim_proc_list {
            {Simulate Behavioral Model}
            {Simulate Post-Place & Route Model}
        }

        set top_level_modules [expr { $param::HasVerilogSource ? "$param::TestBenchModule glbl": $param::TestBenchModule }]
        set has_isim_custom_prj_file [info exists param::ISimCustomProjectFile]
        foreach {process} $sim_proc_list {
            ise::project set {Specify Top Level Instance Names} $top_level_modules -process $process
            if {$has_isim_custom_prj_file} {
                ise::project set {Use Custom Project File} True -process $process
                ise::project set {Custom Project Filename} $param::ISimCustomProjectFile -process $process
            } else {
                ise::project set {Use Custom Project File} False -process $process
            }
            ise::project set {Run for Specified Time} $has_simtime -process $process
            if {$has_simtime} {
                ise::project set {Simulation Run Time} $param::SimulationTime -process $process
            }
            if {$param::HasVerilogSource} {
                ise::project set {Other Compiler Options} {-L unisims_ver -L simprims_ver -L xilinxcorelib_ver -L secureip} -process $process
            }
        }

        ise::project set {ISim UUT Instance Name} $param::DesignInstance

        if [info exists param::Simulator] {
            switch [string tolower $param::Simulator] {
                "isim" - "ise simulator" {
                    return
                }
                default {
                    ise::project set {Simulator} "$param::Simulator $param::HDLLanguage"
                }
            }
        }

        # Modelsim settings
        set sim_proc_param_map {
            {Simulate Behavioral Model} param::BehavioralSimulationCustomDoFile
            {Simulate Post-Translate Model} param::PostTranslateSimulationCustomDoFile
            {Simulate Post-Map Model} param::PostMapSimulationCustomDoFile
            {Simulate Post-Place & Route Model} param::PostPARSimulationCustomDoFile
        }

        foreach {process param} $sim_proc_param_map {
                if [info exists $param] {
                    ise::project set {Use Custom Do File} True -process $process
                    ise::project set {Custom Do File} [subst $$param] -process $process
                    ise::project set {Use Automatic Do File} False -process $process
                }
        }

        if {$has_simtime} {
            foreach {process param} $sim_proc_param_map {
                ise::project set {Simulation Run Time} $param::SimulationTime -process $process
            }
        }
    }

    #-------------------------------------------------------------------------
    # Sets the specific settings related to DSP Tools
    #-------------------------------------------------------------------------
    proc set_dsptools_specific_settings {} {
        if [info exists param::ImplementationStopView] {
            ise::project set {Implementation Stop View} $param::ImplementationStopView
        }
        if [info exists param::ProjectGenerator] {
            ise::project set {Project Generator} $param::ProjectGenerator
        }
    }

    #-------------------------------------------------------------------------
    # Starts the project creation.
    #-------------------------------------------------------------------------
    proc start_project_creation {} {
        file delete "$param::Project\.ise"
        file delete "$param::Project\.xise"
        file delete "$param::Project\.gise"
        file delete "$param::Project\.sgp"
        ise::project new $param::Project
    }

    #-------------------------------------------------------------------------
    # Finishes the project creation.
    #-------------------------------------------------------------------------
    proc finish_project_creation {} {
        ise::project close
    }

    #-------------------------------------------------------------------------
    # Creates a new ISE project.
    #-------------------------------------------------------------------------
    proc create_ise_project {} {
        start_project_creation
        set_project_settings
        add_project_files
        set_dsptools_specific_settings
        set_synthesis_settings
        set_implementation_settings
        set_configuration_settings
        set_simulation_settings
        finish_project_creation
    }

    #-------------------------------------------------------------------------
    # Compiles an ISE project into a bitstream.
    #-------------------------------------------------------------------------
    proc compile_ise_project {} {
        ise::project open $param::Project
        ise::process run {Synthesize}
        ise::process run {Translate}
        ise::process run {Map}
        ise::process run {Place & Route}
        ise::process run {Generate Post-Place & Route Static Timing}
        ise::process run {Generate Programming File}
        ise::project close
    }

    #-------------------------------------------------------------------------
    # Entry point for creating a new ISE project.
    #-------------------------------------------------------------------------
    proc create {} {
        set status [handle_exception {
            decorate_ise_commands
        } "ERROR: An error occurred when loading ISE Tcl commands." False]
        if {!$status} { return }

        set status [handle_exception {
            process_parameters
            dump_parameters
        } "ERROR: An error occurred when processing project parameters."]
        if {!$status} { return }

        set status [handle_exception {
            create_ise_project
        } "ERROR: An error occurred when creating the ISE project."]
        if {!$status} { return }
    }

}
# END namespace ::xilinx::dsptool::iseproject

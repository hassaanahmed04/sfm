# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hassaan-ahmed/Desktop/demoo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hassaan-ahmed/Desktop/demoo/build

# Include any dependencies generated for this target.
include CMakeFiles/reconstruction.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/reconstruction.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/reconstruction.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reconstruction.dir/flags.make

CMakeFiles/reconstruction.dir/main.cpp.o: CMakeFiles/reconstruction.dir/flags.make
CMakeFiles/reconstruction.dir/main.cpp.o: /home/hassaan-ahmed/Desktop/demoo/main.cpp
CMakeFiles/reconstruction.dir/main.cpp.o: CMakeFiles/reconstruction.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/hassaan-ahmed/Desktop/demoo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/reconstruction.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reconstruction.dir/main.cpp.o -MF CMakeFiles/reconstruction.dir/main.cpp.o.d -o CMakeFiles/reconstruction.dir/main.cpp.o -c /home/hassaan-ahmed/Desktop/demoo/main.cpp

CMakeFiles/reconstruction.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/reconstruction.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hassaan-ahmed/Desktop/demoo/main.cpp > CMakeFiles/reconstruction.dir/main.cpp.i

CMakeFiles/reconstruction.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/reconstruction.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hassaan-ahmed/Desktop/demoo/main.cpp -o CMakeFiles/reconstruction.dir/main.cpp.s

# Object files for target reconstruction
reconstruction_OBJECTS = \
"CMakeFiles/reconstruction.dir/main.cpp.o"

# External object files for target reconstruction
reconstruction_EXTERNAL_OBJECTS =

reconstruction: CMakeFiles/reconstruction.dir/main.cpp.o
reconstruction: CMakeFiles/reconstruction.dir/build.make
reconstruction: /usr/local/lib/libceres.a
reconstruction: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.83.0
reconstruction: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.83.0
reconstruction: /usr/lib/x86_64-linux-gnu/libssl.so
reconstruction: /usr/lib/x86_64-linux-gnu/libcrypto.so
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_internal_check_op.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_internal_conditions.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_internal_message.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_internal_nullguard.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_examine_stack.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_internal_format.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_internal_proto.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_internal_log_sink_set.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_internal_globals.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_sink.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_entry.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_strerror.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_flags.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_globals.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_flags_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_flags_reflection.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_flags_config.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_flags_program_name.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_flags_private_handle_accessor.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_flags_commandlineflag.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_flags_commandlineflag_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_flags_marshalling.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_vlog_config_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_internal_fnmatch.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_raw_hash_set.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_hash.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_city.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_bad_variant_access.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_low_level_hash.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_hashtablez_sampler.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_cord.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_bad_optional_access.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_cordz_info.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_cord_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_cordz_functions.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_exponential_biased.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_cordz_handle.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_synchronization.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_stacktrace.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_symbolize.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_demangle_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_graphcycles_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_kernel_timeout_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_time.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_civil_time.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_time_zone.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_malloc_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_crc_cord_state.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_crc32c.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_str_format_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_crc_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_crc_cpu_detect.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_strings.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_strings_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_string_view.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_base.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_spinlock_wait.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_int128.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_throw_delegate.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_debugging_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_raw_logging_internal.so.2401.0.0
reconstruction: /home/hassaan-ahmed/miniconda3/lib/libabsl_log_severity.so.2401.0.0
reconstruction: /usr/lib/x86_64-linux-gnu/libspqr.so
reconstruction: /usr/lib/x86_64-linux-gnu/libcholmod.so
reconstruction: /usr/lib/x86_64-linux-gnu/libamd.so
reconstruction: /usr/lib/x86_64-linux-gnu/libcamd.so
reconstruction: /usr/lib/x86_64-linux-gnu/libccolamd.so
reconstruction: /usr/lib/x86_64-linux-gnu/libcolamd.so
reconstruction: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
reconstruction: /usr/lib/x86_64-linux-gnu/libtbb.so.12.11
reconstruction: /usr/lib/x86_64-linux-gnu/liblapack.so
reconstruction: /usr/lib/x86_64-linux-gnu/libblas.so
reconstruction: /usr/lib/x86_64-linux-gnu/libf77blas.so
reconstruction: /usr/lib/x86_64-linux-gnu/libatlas.so
reconstruction: /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.83.0
reconstruction: CMakeFiles/reconstruction.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/hassaan-ahmed/Desktop/demoo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reconstruction"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reconstruction.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reconstruction.dir/build: reconstruction
.PHONY : CMakeFiles/reconstruction.dir/build

CMakeFiles/reconstruction.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reconstruction.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reconstruction.dir/clean

CMakeFiles/reconstruction.dir/depend:
	cd /home/hassaan-ahmed/Desktop/demoo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hassaan-ahmed/Desktop/demoo /home/hassaan-ahmed/Desktop/demoo /home/hassaan-ahmed/Desktop/demoo/build /home/hassaan-ahmed/Desktop/demoo/build /home/hassaan-ahmed/Desktop/demoo/build/CMakeFiles/reconstruction.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/reconstruction.dir/depend


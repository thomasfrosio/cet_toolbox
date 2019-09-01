import os
import math
import time
import struct
import subprocess
import multiprocessing
from threading import Semaphore, Lock
import argparse
import re
from glob import glob
import itertools
from datetime import datetime

import pandas as pd

VERSION = '0.12.1'

"""
Preprocessing of CET raw images.
From raw images, create motion corrected tilt-series and align them using eTomo.
Developed in Peijun Zhang's lab.

Usage:
    python toolbox.py -h


OVERVIEW:
    - Run Motioncor2 one tilt-series at a time. The tilt-series is parallelized across the available GPUs
      (see pp_mc_gpu and pp_mc_jobs_per_gpu). This is the main advantage of this program, as it can correct 
      an entire stack in a few seconds if you have high-end GPUs/multiple GPUs.
    - Once the images from one tilt-series have been motion corrected, the stack will be create and aligned, 
      and the defocus at the tilt axis will be estimated by Ctffind.
      This step is done in parallel of the main process running MotionCor and the number of stack allowed to be
      process at the same time is set by the number of motion corrected stacks or pp_set_max_cpus (default: nb of 
      logical cores).
    - An on-the-fly mode is available (--fly) allowing to process data while it is being acquired by the microscope.

RAW FILE NAMES:
    - The micrograph file names must have at least 3 information:
        -- The stack number.
        -- The tilt angle.
        -- An extension, either .mrc of .tif.
    - You can set the format of your micrograph file names using pp_set_field_nb, pp_set_field_tilt,
      pp_set_prefix2look4 and:
        -- The stack number, at the position pp_set_field_nb, can be padded with zeros (4, 04, 004, etc.)
           and decorated with none-digit characters (4, tilt4, [4], stack4.mrc, etc.). It must be an integer.
        -- On the other end, the tilt angles are less flexible: the extension can be removed (42, 42.00, 42.00.mrc,
           42.tif, etc.) as well as the [] decorator ([42] or [42.00].mrc). Any other digit will not be 
           handle correctly.
    - Usually we use the following format: <prefix>_<stack_nb>_<order>_<tilt>.mrc/tif; i.e. WT_011_037_-54.00.tif

OUTPUTS:
    - pp_path_motioncor/
        -- motioncorred images: <pp_set_prefix2add>_<nb>_<tilt>.mrc
        -- MotionCor2 logs (one per stack)
    - pp_path_stacks/
        -- stack<nb>/
            --- <pp_set_prefix2add>_<nb>.st and eTomo outputs. (<nb> - stack number: 3 digit number padded with 
                zeros (001, 002, etc.).
            --- Ctffind outputs.
    - pp_path_logfile.
    - toolbox_stack_processed.txt

INPUT FILE (-i, --input <input_file_name>):
    - The easiest way to use this program is to use an input file.
      --create_input_file can extract the default parameters into an input file.
      I strongly recommend to read at least once the parameters available for the user, as it will give you
      a better idea of what is and isn't possible to do in the current version.
    - Every input must be in the input file. Empty parameters are authorized.
  
INTERACTIVE MODE:
    - If no argument specified when starting the program, the interactive mode is triggered.
      It will ask you, one by one, to set the parameters.
      For each parameter, it gives you the parameter ID, the expected type and the default value.
      Answering '+' will show the description of this parameter.
  
RESTRICT THE PROCESSING TO SOME STACKS:
    - If you want to restrict the processing to some specific stacks, the toolbox allows two type of restrictions:
      -- positive: specify the stack number(s) that should only be processed using pp_run_nb or --nb.
         This must be an integer or list of integers (padded with zeros or not) separated by comas. Note that 
         this parameter is ignored when the on-the-fly mode is activated.
      -- negative: specify the stack number(s) that should NOT be processed using the toolbox_stack_processed.txt
         file. Every time a stack is processed, the program will register the processed stack into this file. 
         You can also edit this file yourself (see pp_run_overwrite). This file can be ignored (if you want to
         reprocess some stacks), using pp_run_overwrite=1 or --overwrite.
  
RESTRICT THE PROCESSING TO SOME STEPS:
    - In any configuration, you can run the program with or without the --fly flag. You also have to make sure you 
      specify the correct pp_path_raw/pp_path_motioncor, pp_set_field_nb and pp_set_field_tilt.
    - MotionCor2:
      -- If activated, pp_path_raw must correspond to the path with the movies you want to motion correct.
      -- If deactivated, pp_path_raw is NOT used and pp_path_motioncor must correspond to the path with the sums
         you want to use. Moreover, make use that both pp_set_field_nb and pp_set_field_tilt match the motion
         corrected sums (and not the raw).
    - Activate ONLY Batchruntomo:
      -- If you want to use your own stacks for batchruntomo, make sure they have the correct file name format
         and that they are in the correct path (pp_path_stacks/stack<nb>/pp_prefix2add_<nb>.st, with <nb> 
         padded :03). NB: pp_path_motioncor, pp_set_field_nb and pp_set_field_tilt must match the motion
         corrected sums.
      -- For the initial tilt angles, the program will generate the rawtlt using the motion corrected sum file names 
         or you can specify your mdocs file using pp_path_mdocs. You cannot use your own rawtlt files.

ON-THE-FLY:
    - Briefly, you need to set 2 additional parameters:
      -- pp_otf_max_images_per_stack: Tolerated time of inactivity after which the program stops.
      -- pp_otf_max_image2try: Expected number of image per stack (NB: it can handle missing images, no worries).
    - Careful as it will expect images to be written in order (like a microscope does). For instance, if 
      you transfer your images to your directory in *random* order and start the program to process the stacks
      while there are being transferred, it will not work correctly.
    - For a complete description, see OnTheFly.run.
      
SERIAL-EM MDOC FILES:
    - Mdoc files can be used to specify the tilts for initial alignment in eTomo. 
      The files have to follow this format: "<pp_path_mdocfiles>/*_<stack_nb>.mrc.mdoc". 
      pp_path_mdocfiles can be modified by the user and stack_nb must be a number padded with zeros (3 characters)
      (ex: 001, 010, 100, etc.). Otherwise, the program uses the tilts from the raw image filenames.
  
BATCHRUNTOMO:
    - The tilt-series alignment is done by batchruntomo. As this toolbox was made for an emClarity workflow,
      by default it will generate binned SIRT-like filtered tomograms. See pp_brt parameters for more info.
      If you want to use your own adoc file, use pp_brt_adoc or --adoc <file.adoc> 
    - If you want to run only batchruntomo (use your own stacks), make sure you follow what is mentioned in
      "RESTRICT THE PROCESSING TO SOME STEPS".
  
PARALLEL PROCESSING:
    - The toolbox creates a pool of processes to create/align asynchronously stacks (and to run Ctffind). 
      The number of processes is set to be the number of stacks that need to be processed (limited by 
      pp_set_max_cpus) and is effectively the number of tilt-series that can be processed simultaneously.
      By default, pp_set_max_cpus is set to the number of logical cores of your CPU.
    - Each process will additionally start 4 other processes for batchruntomo (only during expensive tasks). 
      These additional processes are entirely managed by IMOD. I set localhost:4, as this is fine for us 
      (eBIC and STRUBI). You can modify this number (see Batchruntomo._get_batchruntomo), but 4 should be enough. 
      Depending on your system and IMOD install, batchruntomo may not be able to use multiple cores...
      
OVERWRITE:
    - By default, the program will not overwrite a tilt-series. It generates a file (toolbox_stack_processed.txt)
      gathering the stack number(s) that were already processed by the toolbox. You can modify it yourself.
      --overwrite or pp_run_overwrite=1 ignore this file and will reprocess everything (processed stacks will
      be then added to the the queue no matter what).
      
MOTIONCOR2 - GPU IDs:
    - By default, the GPUs are set automatically. If one GPU is hosting one or multiple processes, it will
      be discarded. Only works with Nvidia devices (nvidia-smi has to be installed).
      The user can still specify the GPU ID(s), starting from 0.
      
MOTIONCOR2 - PROCESSES per GPUS:
    - At the moment, the user has to define the number of jobs to run simultaneously within the same GPUs using
      pp_mc_job_per_gpu. This is not ideal but to make it automatic I would need to load additional third parties 
      (ex: pyCUDA). Therefore, this step is manual for now. This number mainly depends on the memory of your GPU(s)
      and the size of your images. If too many jobs are spawn in the device, MotionCor2 will fail. The program will 
      let you know and will restart the jobs that failed setting pp_mc_job_per_gpu to 1.
      
DEFAULT PARAMETERS:
    - To change the default parameters, you only need to change the 'descriptor' variable bellow and change
      the corresponding fields. Careful not to mess up the format.

            
TODO:   1)  Denoising (Janni or Topaz? or something simpler like what we are currently doing in MATLAB...)
            The toolbox will generate binned SIRT-like tomogram, so it should be enough for visualization.
        2)  Generate output graphs for alignment quality and Ctffind.
        3)  Unittest.

WARNING:    Will only work on Python3.6 or later. No efforts were made to keep the program compatible with 
            older version.
            I strongly recommend to run the script from the Anaconda Python3.7 base environment,
            otherwise Pandas need to be installed.
            IMOD is also required. The toolbox will check that it has access to IMOD. I strongly recommend IMOD
            >=4.9.9 mainly because older versions do not to set batchruntomo parameters correctly.
            
HTH,
Thomas
"""

# The descriptor must be correctly formatted for the interactive mode and create_input_file to work.
# Format: <param1>//<type1>//<help1>//<default1>//..//<param2>//<type2>//<help2>//<default2>
descriptor = f"""
pp_set_prefix2look4//str//Look for the movies/sums with this prefix. If '*', catch every mrc/tif image. 
NB: This is used even if MotionCor is deactivated//*//
pp_set_prefix2add//str//Prefix to add to every output (motion corrected images, stacks, logs, etc.). 
NB: prefix/suffix from the original images are removed//WT//
pp_set_field_nb//int//Field (sep: '_', counting from 0) containing stack number in the filename of raw images. 
If you do not want to run MotionCor, this must correspond to the motion corrected images. Only numbers will 
be kept, allowing to have tilt<nb>//1//
pp_set_field_tilt//int//Field (sep: '_', counting from 0) containing tilt angle in the filename of raw images. 
If you do not want to run MotionCor, this must correspond to the motion corrected images//3//
pp_set_pixelsize//float|str//Pixel size of the raw images in Angstrom. If 'header', the pixel size is read 
from the header//header//
pp_set_max_cpus//int//Number of processes used in parallel for creating and aligning the tilt-series. Default: 
available logical cores//{multiprocessing.cpu_count()}//

pp_path_raw//str//Path of the raw images directory//../raw//
pp_path_motioncor//str//Where the MotionCor outputs will go. Will be created if doesn't exist//motioncor//
pp_path_stacks//str//Path of stacks and Ctffind outputs. Will be created if doesn't exist//stacks//
pp_path_mdocfiles//str//Path of mdoc files. Used to create the rawtlt file. File names must be 
(path)/*_<stack_nb>.mrc.mdoc with 'stack_nb' being the stack number (zeros padded, 3 characters)//mdocs//
pp_path_logfile//str//Main log file name. Other log files (MotionCor2, Ctftinf, etc.) will be saved independently//
toolbox_{datetime.now():%d%b%Y}.log//

pp_run_motioncor//bool//Run MotionCor2 or not. If not, the motion corrected images must be in path_motioncor and 
the pp_set settings must correspond to these images//1//
pp_run_ctffind//bool//Estimate the defocus of the lower tilt image of each stack using Ctffind//1//
pp_run_stack//bool//Create the stack from motion-corrected sums//1//
pp_run_batchruntomo//bool//Align the tilt-series using IMOD batchruntomo//1//
pp_run_onthefly//bool//Triggers on-the-fly processing//0//
pp_run_overwrite//bool//Will re-process every stack. If 0, will look at a file (toolbox_stack_processed.txt) and 
skip the stacks that are registered inside this file (<nb>:<nb>:). The stack numbers can be padded with 0//0//
pp_run_nb//int|list//Process only these/this stack(s). Must correspond to the stack number at the field 
pp_set_field_nb (+/- 0 padding). This is ignored when on-the-fly is activated. Default: Process everything//all//

pp_otf_max_images_per_stack//int//Expected number of images per stacks. Used to catch the last stack//37//
pp_otf_max_time2try//float//Tolerated time (min) of inactivity//20//

pp_mc_motioncor//str//Path of MotionCor2 program//
/apps/strubi/motioncorr/2-1.1.0-gcc5.4.0-cuda8.0-sm61/MotionCor2//
pp_mc_desired_pixelsize//float|str//Desired pixel size. If lower than current pixel size, Fourier cropping 
will be done by MotionCor2. If 'current': no Ftbin applied, If 'header_x2': Ftbin=2//header_x2//
pp_mc_throw//int//Frame to remove, from the first frame. From 0//0//
pp_mc_trunc//int//Frame to remove, from the last frame. From 0//0//
pp_mc_tolerance//float//Tolerance of alignment accuracy: less than X pixel//0.5//
pp_mc_iter//int//Iterations after which the alignment stops (if tolerance not achieved already)//10//
pp_mc_patch//int,int(,int)//After global alignment, divides the corrected frames into X*X patches on which 
the local motion is measured//5,5,20//
pp_mc_group//int//Equally divide the input stack into non-overlapping sub-groups. Instead of aligning individual 
frames, the sums of these sub-groups are aligned. The shifts of individual frames are then interpolated and 
extrapolated. Recommended for low-signal movie stacks//1//
pp_mc_gpu//int|str//GPU IDs. Can be a list of int separated by comas (ex: 0,1,2,3) or 'auto'. These must correspond 
to the ID displayed using nvidia-smi. If 'auto', the program will select the visible GPUs 
that do not have any process running//auto//
pp_mc_jobs_per_gpu//int//Number of MotionCor jobs per GPU. For K2 super-resolution and 1080Ti, 2-3 jobs max. 
I recommend to try with one stack to see how many memory is allocated//3//
pp_mc_tif//bool//If the raw images are in TIF//1//
pp_mc_gain//str//Gain reference for MotionCor2. Must have the corrected rotation and be a mrc file//nogain//

pp_ctf_ctffind//str//Path of Ctffind///apps/strubi/ctf/4.1.5/ctffind//
pp_ctf_voltage//float//Acceleration voltage (kV)//300//
pp_ctf_cs//float//Spherical aberration (mm)//2.7//
pp_ctf_amp_cont//float//Amplitude contrast (0 to 1)//0.8//
pp_ctf_size2compute//int//Size of amplitude spectrum to compute//512//
pp_ctf_min_res//float//Minimum resolution//30//
pp_ctf_max_res//float//Maximum resolution//5//
pp_ctf_min_def//float//Minimum defocus//5000//
pp_ctf_max_def//float//Maximum defocus//50000//
pp_ctf_step_def//float//Defocus search step//500//
pp_ctf_astig_type//str|float//Do you know what astigmatism is present?//no//
pp_ctf_exhaustive//str//Slower, more exhaustive search//no//
pp_ctf_astig_restraint//str//Use a restraint on astigmatism//no//
pp_ctf_phase_shift//float//Find additional phase shift//no//

pp_brt_adoc//str//Batchruntomo adoc file to use. Overwrites every pp_brt parameters except pp_brt_start 
and pp_brt_end//default//
pp_brt_gold_size//float//Size of gold beads in nm//10//
pp_brt_rotation_angle//float//Initial angle of rotation in the plane of projection. This is the CCW positive 
rotation from the vertical axis to the suspected tilt axis in the unaligned views//86//
pp_brt_bin_coarse//int//Bin used for coarsed alignment. If 'auto', set the binning to have the gold beads diameter 
to ~12.5 pixel//auto//
pp_brt_target_nb_beads//int//(Generous) Target number of beads per projection. Usually 25 is fine//25//
pp_brt_bin_ali//int//Binning used for final stack and tomogram reconstruction//5//
pp_brt_start//int//Starts at this step. See batchruntomo documentation.//0//
pp_brt_end//int//Ends at this step. 12: stop after gold erase. 20: stop after tomogram generation and rotation//20"""

# Used by Batchruntomo as adoc file. <parameters> will be replaced by the actual inputs.
adoc = f"""
# INPUT FILE BATCHRUNTOMO #

# Setup #
setupset.copyarg.gold = <pp_brt_gold_size>
setupset.copyarg.rotation = <pp_brt_rotation_angle>
setupset.copyarg.dual = 0
setupset.copyarg.userawtlt = 1
setupset.scanHeader = 1


# Preprocessing #
runtime.Preprocessing.any.removeXrays = 1
runtime.Preprocessing.any.archiveOriginal = 0
runtime.Preprocessing.any.endExcludeCriterion = 1
runtime.Preprocessing.any.darkExcludeRatio = 0.17
runtime.Preprocessing.any.darkExcludeFraction = 0.33
runtime.Preprocessing.any.removeExcludedViews = 1


# Coarse alignment #
comparam.xcorr.tiltxcorr.ExcludeCentralPeak = 1
comparam.xcorr.tiltxcorr.FilterRadius2 = 0.15
comparam.prenewst.newstack.BinByFactor = <pp_brt_bin_coarse>
comparam.prenewst.newstack.AntialiasFilter = -1
comparam.prenewst.newstack.ModeToOutput =


# Seeding and tracking #
runtime.Fiducials.any.trackingMethod = 0
runtime.Fiducials.any.seedingMethod = 1

comparam.track.beadtrack.LocalAreaTracking = 1
comparam.track.beadtrack.SobelFilterCentering = 1
comparam.track.beadtrack.KernelSigmaForSobel = 1.5
comparam.track.beadtrack.RoundsOfTracking = 4
runtime.BeadTracking.any.numberOfRuns = 2

comparam.autofidseed.autofidseed.TargetNumberOfBeads = <pp_brt_target_nb_beads>
comparam.autofidseed.autofidseed.AdjustSizes = 1
comparam.autofidseed.autofidseed.TwoSurfaces = 0
comparam.autofidseed.autofidseed.MinGuessNumBeads = 3


# Tomogram positionning #
runtime.Positioning.any.sampleType = 2
runtime.Positioning.any.thickness = 2000
runtime.Positioning.any.hasGoldBeads = 1
comparam.cryoposition.cryoposition.BinningToApply = 5


# Alignment #
comparam.align.tiltalign.SurfacesToAnalyze = 1
comparam.align.tiltalign.LocalAlignments = 1
comparam.align.tiltalign.RobustFitting = 1

#comparam.align.tiltalign.MagOption = 0
#comparam.align.tiltalign.TiltOption = 0
#comparam.align.tiltalign.RotOption = -1
#comparam.align.tiltalign.BeamTiltOption = 2

runtime.TiltAlignment.any.enableStretching = 0
runtime.PatchTracking.any.adjustTiltAngles = 0


# Final aligned stack #
runtime.AlignedStack.any.correctCTF = 0
runtime.AlignedStack.any.eraseGold = 2
runtime.AlignedStack.any.filterStack = 0
runtime.AlignedStack.any.binByFactor = <pp_brt_bin_ali>
runtime.AlignedStack.any.linearInterpolation = 1
comparam.newst.newstack.AntialiasFilter = 1

runtime.GoldErasing.any.extraDiameter = 4
runtime.GoldErasing.any.thickness = 3300
comparam.golderaser.ccderaser.ExpandCircleIterations = 3


# Reconstruction #
comparam.tilt.tilt.THICKNESS = 1500
comparam.tilt.tilt.FakeSIRTiterations = 8
runtime.Trimvol.any.reorient = 2

"""

TAB = ' ' * 4


class Colors:
    # Fancy!
    reset = "\033[0m"
    bold = "\033[1m"
    underline = "\033[4m"
    k = "\033[30m"
    r = "\033[91m"
    g = "\033[92m"
    b = "\033[96m"
    b_b = "\033[44m"


class InputParameters:
    """
    Gather and manage all the inputs of the Toolbox.

    NB: Except some exceptions (see self.check_inputs), inputs are a
        string as there are usually used for sub-processing.

    NB: Add an user input:
            - Modify the descriptor. If it isn't in the descriptor,
              the parameters will not be accessible by InputInteractive and
              the collect_input_from_file will raise a ValueError.
            - Add the corresponding attribute(s) in __init__
              and check_inputs if necessary.
    """

    def __init__(self, defaults):
        """
        Parse command line and create an input file if needed.

        :type defaults:     str
        :param defaults:    descriptor.
        """
        self.cmd_line = self._get_command_line()
        self.defaults = defaults.replace('\n', '').split('//')

        # This is a dead end.
        if self.cmd_line.create_input_file:
            self.create_input_file()

        # Set attributes to None for now.
        # self.get_inputs will update them using interactive mode or input file.
        self.pp_set_prefix2look4 = None
        self.pp_set_prefix2add = None
        self.pp_set_field_nb = None
        self.pp_set_field_tilt = None
        self.pp_set_pixelsize = None
        self.pp_set_max_cpus = None

        self.pp_path_raw = None
        self.pp_path_motioncor = None
        self.pp_path_stacks = None
        self.pp_path_mdocfiles = None
        self.pp_path_logfile = None

        self.pp_otf_max_images_per_stack = None
        self.pp_otf_max_time2try = None

        self.pp_mc_motioncor = None
        self.pp_mc_desired_pixelsize = None
        self.pp_mc_throw = None
        self.pp_mc_trunc = None
        self.pp_mc_tolerance = None
        self.pp_mc_iter = None
        self.pp_mc_patch = None
        self.pp_mc_group = None
        self.pp_mc_gpu = None
        self.pp_mc_jobs_per_gpu = None
        self.pp_mc_tif = None
        self.pp_mc_gain = None

        self.pp_ctf_ctffind = None
        self.pp_ctf_voltage = None
        self.pp_ctf_cs = None
        self.pp_ctf_amp_cont = None
        self.pp_ctf_size2compute = None
        self.pp_ctf_min_res = None
        self.pp_ctf_max_res = None
        self.pp_ctf_min_def = None
        self.pp_ctf_max_def = None
        self.pp_ctf_step_def = None
        self.pp_ctf_astig_type = None
        self.pp_ctf_exhaustive = None
        self.pp_ctf_astig_restraint = None
        self.pp_ctf_phase_shift = None

        self.pp_brt_adoc = None
        self.pp_brt_gold_size = None
        self.pp_brt_rotation_angle = None
        self.pp_brt_bin_coarse = None
        self.pp_brt_target_nb_beads = None
        self.pp_brt_bin_ali = None
        self.pp_brt_start = None
        self.pp_brt_end = None

        self.pp_run_motioncor = None
        self.pp_run_ctffind = None
        self.pp_run_stack = None
        self.pp_run_batchruntomo = None
        self.pp_run_onthefly = None
        self.pp_run_overwrite = None
        self.pp_run_nb = None  # See self._set_stack

        self.hidden_oft_gpu = None  # See Metadata._get_gpu_id
        self.hidden_mc_ftbin = None  # See self.set_pixelsize
        self.hidden_queue_filename = 'toolbox_stack_processed.txt'

    def get_inputs(self):
        """Update the attr: using the inputs, either from interactive or from input file."""

        # At this point, the processing is likely to start so check IMOD.
        self.check_dependency('imod')

        # Catch the parameters from an input file.
        # user_inputs is a dict.
        if self.cmd_line.input:
            print(f'{TAB}- Mode: Using inputs from {Colors.bold}{self.cmd_line.input}{Colors.reset}\n')
            user_inputs = self._get_inputs_from_file()
        else:
            print(f'{TAB}- Mode: Interactive.\n')
            interactor = InputInteractive(self.defaults)
            user_inputs = interactor.get_inputs()

        # Update the attributes with collected inputs.
        for key, value in user_inputs.items():
            setattr(self, key, value)
        self.hidden_oft_gpu = self.pp_mc_gpu

        # Check inputs. Change the type of some attributes.
        self._check_inputs()

    def set_bin_coarsed(self):
        """
        Set the binning used for coarsed alignment.
        The idea is to have a gold bead equals to 10-15 pixels.
        """
        if self.pp_brt_bin_coarse == 'auto':
            try:
                self.pp_brt_bin_coarse = int(12.5 // (float(self.pp_brt_gold_size)
                                                      * (self.pp_mc_desired_pixelsize / 10)))
            except TypeError:
                raise

    def set_pixelsize(self, meta=None):
        """
        Modify pixel size if necessary, as well as hidden_mc_ftbin.

        :type meta:    DataFrame or None
        :param meta:   Metadata. Must contain raw, nb and tilt.

        If meta = None, convert current and desired pixel sizes to floats if possible.
        If meta = DataFrame, extract the current pixel size from header, adjust the desired
            pixel size if necessary ('current' or 'header_x2').

        In any case, try to compute hidden_mc_ftbin.
        """
        if meta is not None and self.pp_set_pixelsize == 'header':
            # pp_set_pixelsize can only be a float or 'header'.
            # In the later, extract pixel size from header.
            self.pp_set_pixelsize = self._get_pixel_size_from_meta(meta)
            self._set_desired_pixelsize()

        else:
            # Set current pixel size.
            if self.pp_set_pixelsize != 'header':
                try:
                    self.pp_set_pixelsize = float(self.pp_set_pixelsize)
                except ValueError:
                    raise ValueError(f"Pixel: 'pp_set_pixelsize' should be a float or 'header'.")

            # Set desired pixel size.
            if self.pp_mc_desired_pixelsize not in ('header_x2', 'current'):
                try:
                    self.pp_mc_desired_pixelsize = float(self.pp_mc_desired_pixelsize)
                except ValueError:
                    raise ValueError(
                        f"Pixel: 'pp_mc_desired_pixelsize' should be a float or 'current' or 'header_x2'.")

        # Try to compute hidden_mc_ftbin.
        self._set_ftbin()

    def save2logfile(self):
        """Save the inputs to the log file."""

        inputs = '\n\t'.join(f'{key}: {value}' for key, value in self.__dict__.items() if 'pp_' in key)
        with open(self.pp_path_logfile, 'a') as log_file:
            log_file.write(f'Toobox version {VERSION}.\n'
                           f"Using parameters:\n\t{inputs}\n\n")

    def warnings(self):
        """Warn the user about specific run settings."""

        if not self.pp_run_motioncor:
            logger(f"{Colors.r}WARNING: MotionCor deactivated.\n"
                   f"'pp_set_field_nb', 'pp_set_field_tilt' must match the motion corrected images.{Colors.reset}\n")
        if not self.pp_run_stack and self.pp_run_batchruntomo:
            logger(f"{Colors.r}WARNING: Newstack deactivated.\n"
                   f"Your stacks must be as followed: pp_path_stacks/stack<nb>/pp_prefix2add_<nb>.st, "
                   f"with <nb> being the 3 digit stack number (padded with zeros).{Colors.reset}\n")

    def create_input_file(self):
        """Write an input file using the default parameters."""

        # Format the descriptor.
        formatted_description = ''
        for i in range(0, len(self.defaults), 4):
            left = f'{self.defaults[i]} ({self.defaults[i + 1]}) : '
            if len(self.defaults[i + 2]) > 60:
                right = self._format_input_description(self.defaults[i + 2], left=40, right=60)
            else:
                right = f'{self.defaults[i + 2]}'
            formatted_description += f'{left:>40}{right}\n'

        # Create the header
        date_created = "Created: {0:%d-%b-%Y} | {0:%H:%M}".format(datetime.now())
        header = '-' * 60 + \
                 f'\nToolbox version {VERSION}.' + \
                 f'\n{date_created}\n\n' + \
                 'Description:\n' + formatted_description + \
                 '-' * 60 + '\n\n'

        parameters = "\n".join(f"{self.defaults[i]}={self.defaults[i + 3]}" for i in range(0, len(self.defaults), 4))
        parameters = f"Parameters: (param=value, no whitespace, in-line comments are OK)\n{parameters}\n"

        with open(self.cmd_line.create_input_file, 'w') as f:
            f.write(header + parameters)

        print(f"Input file '{Colors.bold}{self.cmd_line.create_input_file}{Colors.reset}' was created.\n"
              f"Closing.")
        exit()

    @staticmethod
    def check_dependency(program):
        """If program (str) is not in PATH, raise OSError."""

        if not any(os.access(os.path.join(path, program), os.X_OK)
                   for path in os.environ["PATH"].split(os.pathsep)):
            raise OSError(f'Check dependency: {program} needs to be in PATH...\n')

    @staticmethod
    def _format_input_description(description, left, right):
        """Wrap string within left and right padding."""

        final_string = ''
        string = ''
        for i in description.split():
            size = len(string) + len(i)
            if size <= left + right:
                string += ' ' + i if string else i
            else:
                pad = ' ' * left
                final_string += '\n' + pad + string if final_string else string
                string = i

        if string:
            pad = ' ' * left
            final_string += '\n' + pad + string if final_string else string

        return final_string

    @staticmethod
    def _get_command_line():
        """Parse the command line."""

        parser = argparse.ArgumentParser(prog='CET Toolbox',
                                         description='Program helping with CET data pre-processing.')

        # create_input_file OR parse input file.
        parser_group = parser.add_mutually_exclusive_group()
        parser_group.add_argument('-i', '--input',
                                  nargs='?',
                                  type=str,
                                  help='Input file containing the parameters.')

        parser_group.add_argument('-c', '--create_input_file',
                                  nargs='?',
                                  const=f"Toolbox_inputs_{datetime.now():%d%b%Y}.txt",
                                  type=str,
                                  help='Create an input file from the default parameters.')

        # Overwrite inputs from command line.
        parser.add_argument('--fly',
                            nargs='?',
                            const=True,
                            help='Enable on-the-fly processing.')
        parser.add_argument('--logfile',
                            nargs='?',
                            type=str,
                            help='Log file name.')
        parser.add_argument('--nb',
                            type=str,
                            nargs='?',
                            help='Stack number(s) to process. Integer or a list of (optionnaly zero padded) '
                                 'integers separated by comas.')
        parser.add_argument('--overwrite',
                            nargs='?',
                            const=True,
                            help='Ignore previous processing and overwrite everything.')
        parser.add_argument('--adoc',
                            type=str,
                            nargs='?',
                            help='Batchruntomo adoc file.')

        # Version
        parser.add_argument('--version',
                            action='version',
                            version=f'%(prog)s {VERSION}',
                            help="Show program's version.")

        return parser.parse_args()

    def _get_inputs_from_file(self):
        """Extract the inputs from an input file."""

        try:
            with open(self.cmd_line.input, 'r') as f:
                lines = f.readlines()
        except IOError as err:
            raise IOError(f'Collect inputs from file: {err}')

        # Remove the header to be sure it is not going to be parsed.
        head = 0
        for i, line in enumerate(lines):
            if '------' in line:
                head = i
        lines = lines[head:]

        # Parse.
        # Empty parameters are accepted.
        inputs_dict = {}
        r = re.compile(r'^\w.+=(\S+|\s)')
        for line in lines:
            for m in r.finditer(line):
                key, value = m.group().split('=')
                inputs_dict[key] = value

        # Extract the parameters from the defaults and check they are all in the input file.
        if inputs_dict != {}:
            parameters = self.defaults
            parameters_generator = (parameters[item] for item in range(0, len(parameters), 4))
            for parameter in parameters_generator:
                if parameter not in inputs_dict:
                    raise ValueError(f'Collect inputs from file: {parameter} is missing.')
            return inputs_dict
        else:
            raise Exception(f'Collect inputs from file: No parameter detected in {self.cmd_line.input}.')

    def _check_inputs(self):
        """Few sanity checks and format required inputs."""

        head = 'Check inputs:'

        if self.pp_run_motioncor:
            assert os.path.isdir(self.pp_path_raw), f"{head} pp_path_raw ({self.pp_path_raw}) not found."
        if self.pp_set_prefix2add == '':
            raise ValueError(f'{head} {self.pp_set_prefix2add} should not be empty.')

        # Convert to int.
        for _input in ('pp_set_field_nb',
                       'pp_set_field_tilt',
                       'pp_mc_jobs_per_gpu',
                       'pp_set_max_cpus'):
            try:
                setattr(self, _input, int(getattr(self, _input)))
            except ValueError:
                raise ValueError(f"{head} {_input} must be an integer.")

        # Convert to bool.
        for _input in ('pp_run_motioncor',
                       'pp_run_ctffind',
                       'pp_run_stack',
                       'pp_run_batchruntomo',
                       'pp_run_onthefly',
                       'pp_run_overwrite',
                       'pp_mc_tif'):
            try:
                setattr(self, _input, bool(int(getattr(self, _input))))
            except ValueError:
                raise ValueError(f'{head} {_input} must be a boolean.')

        self._check_inputs_priority()
        self._set_stack()
        self.set_pixelsize()

    def _check_inputs_priority(self):
        """
        When the command line gives access to an input,
        make sure it overwrites the inputs.
        """
        if self.cmd_line.fly:
            self.pp_run_onthefly = self.cmd_line.fly

        if self.cmd_line.logfile:
            self.pp_path_logfile = self.cmd_line.logfile

        if self.cmd_line.overwrite:
            self.pp_run_overwrite = self.cmd_line.overwrite

        if self.cmd_line.nb:
            self.pp_run_nb = self.cmd_line.nb

    def _set_stack(self):
        """
        pp_run_nb can be modified by the command line (--nb) or by the inputs (file or interactive).
        It will be read by Metadata, which expects a list of int or empty list.

        At this point, pp_run_nb is a string.
        If 'all', all the available stacks should be selected: Set it to [].
        If string of integers (separated by comas), restricts to specific stacks:
            Convert str of int -> list of int.

        NB: In addition to these possible restrictions, Metadata._exclude_queue can add a negative (process
            everything except these ones) priority restriction using the tool_processed.queue file.

        NB: When on-the-fly, self.pp_run_nb will be set to the queue.
        """
        tmp = []
        if isinstance(self.pp_run_nb, str) and self.pp_run_nb != 'all' and not self.pp_run_onthefly:
            try:
                for nb in self.pp_run_nb.split(','):
                    tmp.append(int(nb))
            except ValueError:
                raise ValueError("Restrict stack: pp_run_nb must be 'all' "
                                 "or list of integers separated by comas.")
        self.pp_run_nb = tmp

    @staticmethod
    def _get_pixelsize_header(mrc_filename):
        """Parse the header and compute the pixel size."""

        size = 4
        offsets = [0, 4, 8, 40, 44, 48]
        structs = '3i3f'
        header = b''

        with open(mrc_filename, 'rb') as f:
            for offset in offsets:
                f.seek(offset)
                header += f.read(size)

        [nx, ny, nz,
         cellax, cellay, cellaz] = struct.unpack(structs, header)

        # Make sure the pixel size is the same for each axis.
        px, py, pz = cellax / nx, cellay / ny, cellaz / nz
        if math.isclose(px, py, rel_tol=1e-4) and math.isclose(px, pz, rel_tol=1e-4):
            return px
        else:
            raise Exception(f'Extract pixel size: {mrc_filename} has different pixel sizes in x, y and z.')

    def _get_pixel_size_from_meta(self, meta):
        """
        Catch the pixel from header.
        Check only for lowest tilt of every stack and make sure to
        have the same pixel size for every stacks.
        """
        pixelsizes = []
        for stack in meta['nb'].unique():
            meta_stack = meta[meta['nb'] == stack]
            image = meta_stack['raw'].loc[meta_stack['tilt'].abs().idxmin(axis=0)]

            pixelsizes.append(self._get_pixelsize_header(image))

        if len(set(pixelsizes)) == 1:
            return pixelsizes[0]
        else:
            raise Exception('Set pixel size: more than one pixel size was detected. '
                            'It is not supported at the moment, so stop here...')

    def _set_desired_pixelsize(self):
        """If the desired pixel size rely on the current pixel size, update it."""

        if not self.pp_run_motioncor:
            self.pp_mc_desired_pixelsize = self.pp_set_pixelsize
        else:
            if self.pp_mc_desired_pixelsize == 'header_x2':
                self.pp_mc_desired_pixelsize = self.pp_set_pixelsize * 2
            elif self.pp_mc_desired_pixelsize == 'current':
                self.pp_mc_desired_pixelsize = self.pp_set_pixelsize

    def _set_ftbin(self):
        """Compute Ftbin for MotionCor2."""
        if isinstance(self.pp_set_pixelsize, float) and isinstance(self.pp_mc_desired_pixelsize, float):
            self.hidden_mc_ftbin = self.pp_mc_desired_pixelsize / self.pp_set_pixelsize


class InputInteractive:
    """Ask the user to set the required parameters."""

    def __init__(self, defaults):
        self.defaults = defaults
        self.padding_allowed = 55
        self.trimming_help = 5

    def get_inputs(self):
        """
        Ask the user to enter the inputs one by one.
        The default parameters will be suggested and used if nothing else is given.
        Entering '+' will display the description for this parameter.

        The defaults format is described in the global scope.

        TODO: Live input checks?
        """
        # Extract the parameters from the defaults.
        parameters = self.defaults
        parameters_organized = [parameters[item:item + 4] for item in range(0, len(parameters), 4)]

        all_inputs = dict()

        print('--- Project ---')
        all_inputs.update(self._get_inputs_collector(
            filter(lambda x: x[0].split('_')[1] == 'set', parameters_organized)))

        print('--- Paths ---')
        all_inputs.update(self._get_inputs_collector(
            filter(lambda x: x[0].split('_')[1] == 'path', parameters_organized)))

        print('--- Steps ---')
        all_inputs.update(self._get_inputs_collector(
            filter(lambda x: x[0].split('_')[1] == 'run', parameters_organized)))

        # Run only the steps that were selected by the user.
        functions = [['--- MotionCor ---', 'pp_run_motioncor', 'mc'],
                     ['--- Ctffind ---', 'pp_run_ctffind', 'ctf'],
                     ['--- Batchruntomo ---', 'pp_run_batchruntomo', 'brt'],
                     ['--- On-the-fly ---', 'pp_run_onthefly', 'otf']]
        for func in functions:
            if all_inputs[func[1]] == '1' or all_inputs[func[1]] == 'True':
                print(func[0])
                all_inputs.update(self._get_inputs_collector(
                    filter(lambda x: x[0].split('_')[1] == func[2], parameters_organized)))
            else:
                all_inputs.update(self._get_inputs_collector(
                    filter(lambda x: x[0].split('_')[1] == func[2], parameters_organized),
                    use_default=True))

        return all_inputs

    def _get_inputs_collector(self, list_of_parameters, use_default=False):
        """
        Format a list of parameters to input.

        :param list_of_parameters:    Catch user inputs for these parameters.
        :param use_default:         If True, shortcut, use default.
                                    No need to ask the user as these parameters will not be used.
        :return:                    Dictionary gathering the answer for each parameter.
        """
        # For some reason pycharm doesn't like to feed a list of pairs to dict(). Tsss.
        # So create dict directly.
        answers = {}
        if use_default:
            answers.update({param[0]: param[3] for param in list_of_parameters})
        else:
            for parameter in list_of_parameters:

                answer = input(self._format_input_header(parameter))
                if not answer:
                    answer = parameter[3]
                elif answer == '+':
                    answer = input(self._format_input_help(parameter[2]))
                    if not answer:
                        answer = parameter[3]

                answers[parameter[0]] = answer
        return answers

    def _format_input_header(self, parameter):
        """
        For a given parameter, format the string to print.
        Format:
        <param_name> (<param_type>) [<param_default>]:
        """
        space = len(parameter[0] + parameter[1] + parameter[3]) + 6
        padding_left = self.padding_allowed - space
        if padding_left > 0:
            newline_or_space = ' '
        else:
            newline_or_space = '\n'
            padding_left = self.padding_allowed - len(parameter[3]) - 2

        def_value = f'{Colors.g}{parameter[3]}]{Colors.reset}'
        param = f'{Colors.b}{parameter[0]}{Colors.reset}'

        return f"{param} ({parameter[1]}){newline_or_space}{def_value}{' ' * padding_left}: "

    def _format_input_help(self, param_help):
        """Wrap string within a self.padding_allowed - self.trimming_help."""
        final_string = ''
        string = ''
        for i in param_help.split():
            size = len(string) + len(i)
            if size < self.padding_allowed - self.trimming_help:
                string += ' ' + i if string else i
            else:
                pad = ' ' * (self.padding_allowed - size)
                final_string += '\n' + string + pad if final_string else string + pad
                string = i

        if string:
            pad = ' ' * (self.padding_allowed - len(string))
            final_string += '\n' + string + pad if final_string else string + pad

        return final_string + ': '


class OnTheFly:
    """When the microscope is done with a stack, send it to pre-processing."""

    def __init__(self, inputs):
        self.path = inputs.pp_path_raw if inputs.pp_run_motioncor else inputs.pp_path_motioncor
        self.prefix = inputs.pp_set_prefix2look4
        self.extension = 'tif' if inputs.pp_mc_tif and inputs.pp_run_motioncor else 'mrc'
        self.field_nb = inputs.pp_set_field_nb

        # Refresh
        self.time_between_checks = 5

        # Queue of stacks. Processed stacks are saved and not reprocessed.
        self.queue_filename = inputs.hidden_queue_filename
        self.processed = self._exclude_queue() if not inputs.pp_run_overwrite else []
        self.queue = None

        # Tolerate some inactivity.
        self.buffer = 0
        try:
            self.buffer_tolerance_sec = int(inputs.pp_otf_max_time2try) * 60
        except ValueError as err:
            raise ValueError(f'On-the-fly: {err}')
        self.buffer_tolerance = self.buffer_tolerance_sec // self.time_between_checks

        # Catch the available raw files.
        self.data = None
        self.data_stacks_available = None
        self.data_current = None

        # Compute the current stack.
        self.stack_current = None
        self.len_current_stack_previous_check = 0
        self.len_current_stack = 0
        try:
            self.len_expected = int(inputs.pp_otf_max_images_per_stack)
        except ValueError as err:
            raise ValueError(f'On-the-fly: {err}')

    def run(self, inputs):
        """
        On-the-fly: run preprocessing while data is being written...

        How does it works:
            (loop every n seconds).
            1) Catch mrc|tif files in path (raw or motioncor).
            2) Group the files in tilt-series.
            3) Split stack in two: old stack and current stack. If only one stack, old stack is not defined.
                - old stack: added to the queue if not already processed.
            4) Decide if current stack is finished or not. If so, add to the queue if not already processed.
            5) Send the queue to pre-processing and clear the queue.

        Current stack:
            - The program use a buffer to "remember" how long it's been since the last change in the raw files.
            - When a tilt-series is being written, the buffer will be reset every time a new image (from the
              same stack) is detected. Therefore, the program will tolerate having nothing to send to
              processing for a long time (a tilt-series can be acquired in more than 40min).
            - The user has to specify the expected number of images per stack. The program will send the stack
              to processing if this number is reached.
              NB:   When a tilt-series is send to processing, the program tag this stack as processed and
                    will no longer touch it.
                    If a stack has more images than expected, it becomes ambiguous so the program will stop
                    by raising an AssertationError.

            - Stack with less images than expected:
                - If new images of this stack are detected (the microscope is doing the acquisition of this stack),
                  then the program will wait for it to finish.
                - The tolerated time between images is set by the user. If nothing is written after this
                  tolerated time of inactivity, the tilt-series is send to processing (no matter the number
                  of images) and the program stops. It is the only way I found to process the last tilt-series
                  of an acquisition with missing images...

        NB: If an old stack has less images than expected, the program should handle this without any difficulty.
        NB: Stack that are already processed (toolbox_stack_processed.txt) are already in self.processed (__init__),
            so this function will not send them to pre-processing.
        NB: --stack is ignored: positive selection cannot be used with --fly
        """
        running = True
        while running:
            print(f"\rFly: Buffer = "
                  f"{round(self.buffer * self.time_between_checks)} /{self.buffer_tolerance_sec}sec",
                  end='')

            time.sleep(self.time_between_checks)
            self._get_raw_files()

            # The goal is to identify the tilt-series that are finished and register them in this list.
            self.queue = []

            # Split the files into the current stack and old stacks if any.
            len_avail = len(self.data_stacks_available)
            if len_avail == 1:
                self.stack_current = self.data_stacks_available[0]
            elif len_avail > 1:
                self._get_old_stacks()
            else:
                continue

            # It is more tricky to know what to do with the last stack.
            self._analyse_last_stack()

            # If the buffer reaches the limit, it means nothing is happening for too long, so stop.
            if self.buffer == self.buffer_tolerance:
                if self.stack_current not in self.processed:
                    self.queue.append(self.stack_current)
                    self.processed.append(self.stack_current)
                running = False

            # Send to preprocessing.
            if self.queue:
                print('\n')
                inputs.pp_run_nb = self.queue
                preprocessing(inputs)

            # Reset the length if necessary.
            self.len_current_stack_previous_check = self.len_current_stack

    def _exclude_queue(self):
        """Extract the stacks already processed from inputs.pp_hidden_queue_filename."""

        try:
            with open(self.queue_filename, 'r') as f:
                remove_stack = f.readlines()

                list2remove = []
                for line in remove_stack:
                    line = [int(i) for i in line.strip('\n').strip(' ').strip(':').split(':') if i != '']
                    list2remove += line
                return list2remove

        except IOError:
            # First time running.
            return []

    @staticmethod
    def _set_ordered(iterable2clean):
        """Remove redundant values while preserving the order."""

        cleaned = []
        for item in iterable2clean:
            if item not in cleaned:
                cleaned.append(item)

        return cleaned

    def _get_raw_files_number(self, file):
        filename_split = file.split('/')[-1].split('_')
        return int(''.join(i for i in filename_split[self.field_nb] if i.isdigit()))

    def _get_raw_files(self):
        """
        Catch the raw files in path, order them by time of writing and set the
        number of the stack.
        """
        files = sorted(glob(f'{self.path}/{self.prefix}*.{self.extension}'), key=os.path.getmtime)
        self.data = pd.DataFrame(dict(raw=files))
        self.data['nb'] = self.data['raw'].map(self._get_raw_files_number)
        self.data_stacks_available = self._set_ordered(self.data['nb'])

    def _get_old_stacks(self):
        """
        At this point, we know there is more than one stack.
        Therefore, old stack are finished and can be processed.
        """
        stack_current = self.data_stacks_available[-1]
        if stack_current != self.stack_current:
            self.len_current_stack_previous_check = 0
        self.stack_current = stack_current

        for old_stack in self.data_stacks_available[:-1]:
            if old_stack not in self.processed:
                self.queue.append(old_stack)
                self.processed.append(old_stack)

    def _analyse_last_stack(self):
        self.data_current = self.data[self.data['nb'] == self.stack_current]
        self.len_current_stack = len(self.data_current)
        if self.len_current_stack == self.len_expected:
            if self.stack_current not in self.processed:
                # Stack has not been processed yet and has the expected size: go to pp.
                self.queue.append(self.stack_current)
                self.processed.append(self.stack_current)
                self.len_current_stack = 0
            else:
                # Stack has been processed and has the expected length. It is either:
                # - the last stack of the acquisition.
                # - it is not the last one, we are just waiting for the first image of the next stack.
                # In both cases, we fill the buffer and wait.
                self.buffer += 1

        elif self.len_current_stack < self.len_expected:
            # In most cases, the stack is not finished, but it can be the actual last stack which for some
            # reason has missing images. To differentiate between the two, check the number of images
            # that were there before:
            # - if no new images, increase the self.buffer.
            # - if new images, the stack is just not finished so reset the self.buffer.
            if self.len_current_stack == self.len_current_stack_previous_check:
                self.buffer += 1
            elif self.len_current_stack > self.len_current_stack_previous_check:
                self.buffer = 0
            else:
                raise AssertionError(f'On-the-file: the tilt-series {self.stack_current} has '
                                     f'less images than the previous check.'
                                     f'It is ambiguous, so stop here.')
        else:
            raise AssertionError(f'On-the-file: the tilt-series {self.stack_current} has more images than expected.'
                                 f'It is ambiguous, so stop here.')


class WorkerManager:
    """Manages processes"""

    def __init__(self, stack2process, inputs):
        """
        Start the pool.
        The size of the pool is set by the number of stacks that need to be processed and is limited by the number
        of logical cores or pp_set_max_cpus. For on-the-fly processing, most of the time only one stack is send to
        pp, therefore Stack waits for Ctffind (wait a few seconds). As such, if there is one stack, the pool will
        be set to 2 processes to run everything in parallel.

        :param stack2process:   stack numbers that will be processed in this session of pre-processing.
        :param inputs:          InputParameters.
        """
        processes = len(stack2process)
        if processes > inputs.pp_set_max_cpus:
            processes = inputs.pp_set_max_cpus
        elif processes == 1:
            processes = 2

        self.semaphore = Semaphore(processes)
        self.pool = multiprocessing.Pool(processes=processes)
        self.filename_queue = inputs.hidden_queue_filename

    def new_async(self, run, task):
        """Start a new task, wait if all worker are busy."""
        self.semaphore.acquire()
        self.pool.apply_async(run, args=(task,),
                              callback=self.task_done,
                              error_callback=self.task_failed)

    def task_done(self, _):
        """
        Called once task is done, releases the caller if blocked.
        Any output is lost: the workers should sent there log to the logger themselves.
        """
        self.semaphore.release()

    def task_failed(self, error):
        """
        When an exception is raised from a child process, terminate the pool and
        release the semaphore to prevent the main process to be stuck in self.new_async.
        This is handled by the result thread, thus raising an exception will not
        stop the program, just this thread. The pool is closed, so the main process
        will fail by itself if a new job is submitted (ValueError: Pool not running).
        """
        self.pool.close()
        self.pool.terminate()
        self.semaphore.release()
        raise error

    def close(self):
        """Wait for the processes to finish and synchronize to parent."""
        self.pool.close()
        self.pool.join()


class Metadata:
    """Manage the metadata"""

    def __init__(self, inputs):
        self.inputs = inputs
        self.stacks_nb = None
        self.stacks_len = None
        self.stacks_images_per_stack = None
        self.path2catch = inputs.pp_path_raw if inputs.pp_run_motioncor else inputs.pp_path_motioncor
        self.extension = 'tif' if inputs.pp_mc_tif and inputs.pp_run_motioncor else 'mrc'

    def get_metadata(self):
        """Gather metadata and return it in one DataFrame."""

        meta = self._get_raw_files()

        self.stacks_len = len(meta)
        self.stacks_nb = meta['nb'].unique()
        self.stacks_images_per_stack = ', '.join((str(i)
                                                  for i in set((len(meta[meta['nb'] == stack])
                                                                for stack in self.stacks_nb))))

        # Set output file name and assign every image to one GPU.
        if self.inputs.pp_run_motioncor:
            # Only used by MotionCor.
            meta = meta.sort_values(by='nb', axis=0, ascending=True)
            meta['gpu'] = self._get_gpu_id()
            meta['output'] = meta.apply(
                lambda row: f"{self.inputs.pp_path_motioncor}/"
                            f"{self.inputs.pp_set_prefix2add}_{row['nb']:03}_{row['tilt']}.mrc",
                axis=1)
        else:
            meta['output'] = meta['raw']

        return meta

    def save_processed_stack(self):
        """Save in a text file the stacks that were processed."""

        with open(self.inputs.hidden_queue_filename, 'a') as f:
            f.write('\n:' + ':'.join((str(i) for i in self.stacks_nb)) + ':')

    def _get_raw_files(self):
        """
        Gather raw file names into a DataFrame.
        For each .mrc|.tif file, catch: raw file name, nb and tilt.

        Some stacks can be removed (see self._clean_raw_files)
        """
        raw_files = glob(f'{self.path2catch}/{self.inputs.pp_set_prefix2look4}*.{self.extension}')
        assert raw_files, f'Get Raw files: No {self.extension} files detected in path: {self.path2catch}.'

        # Apply the restrictions on the stacks that should be processed
        # and extract tilt-series nb and tilt angle for each image.
        raw_files, raw_files_nb, raw_files_tilt = self._clean_raw_files(raw_files)
        raw_files = sorted(raw_files, key=os.path.getmtime)
        if not raw_files:
            logger('Nothing to process (maybe it is already processed?).')
            exit()

        return pd.DataFrame(dict(raw=raw_files, nb=raw_files_nb, tilt=raw_files_tilt))

    def _clean_raw_files(self, raw_files):
        """
        First apply a restriction on which stack to keep for processing:
            There are two possible restrictions:
                - pp_run_nb: positive restriction; process only these specified stacks.
                - toolbox.queue: negative restriction; do not process these stacks.

            The negative restriction is the priority restriction and can be deactivated
            using pp_run_overwrite. In any case, it should only be ran when --fly is False
            because OnTheFly already check for the toolbox_stack_processed.txt.

            Both restrictions are a list of integers (stack nb) or an empty list if no restriction found.

        Then, structure the raw files to feed to pd.DataFrame.

        :param raw_files:               List of every raw files.
        :return raw_files_cleaned:      Selected raw files.
        :return raw_files_cleaned_nb:   Stack numbers of the corresponding raw_files_cleaned.
        :return raw_files_cleaned_tilt: Tilt angles of the corresponding raw_files_cleaned.
        """
        # Every stack in stack2remove will be remove from raw_files.
        # If stack2remove is empty, negative restriction is not applied.
        # OnTheFly already manages the negative restriction.
        if self.inputs.pp_run_overwrite or self.inputs.pp_run_onthefly:
            stack2remove = []
        else:
            stack2remove = self._exclude_stacks2remove()

        # Start cleaning.
        raw_files_cleaned, raw_files_cleaned_nb, raw_files_cleaned_tilt = [], [], []
        for file in raw_files:
            filename_split = file.split('/')[-1].split('_')
            stack_nb = int(''.join(i for i in filename_split[self.inputs.pp_set_field_nb] if i.isdigit()))

            # First check that the stack was not processed already.
            if stack_nb in stack2remove:
                continue

            # Then check that the stack is one of the stack that should be processed.
            # If self.inputs.pp_run_nb is empty, then positive restriction is not applied.
            if self.inputs.pp_run_nb and stack_nb not in self.inputs.pp_run_nb:
                continue

            raw_files_cleaned.append(file)
            raw_files_cleaned_nb.append(stack_nb)
            try:
                file_tilt = filename_split[self.inputs.pp_set_field_tilt].replace(
                    f'.{self.extension}', '').replace('[', '').replace(']', '')
                raw_files_cleaned_tilt.append(float(file_tilt))
            except IndexError as err:
                raise IndexError(f'Clean raw files: {err}')

        return raw_files_cleaned, raw_files_cleaned_nb, raw_files_cleaned_tilt

    def _exclude_stacks2remove(self):
        """Catch every stack that was already processed (saved in inputs.hidden_queue_filename)."""

        stack2remove = []
        try:
            with open(self.inputs.hidden_queue_filename, 'r') as f:
                remove_stack = f.readlines()

            if remove_stack:
                for line in remove_stack:
                    line = [int(i) for i in line.strip('\n').strip(' ').strip(':').split(':') if i != '']
                    stack2remove += line

            return stack2remove

        # First time running.
        except IOError:
            return stack2remove

    def _get_gpu_id(self):
        """
        Set a GPU to an image.
        For each tilt-series, the images are dispatched across the visible GPUs.

        :return mc_gpu:     list of GPU to map with an image.
        """
        if self.inputs.pp_run_onthefly:
            # Used for otf: every time pp is called, it must recompute the available GPUs.
            # Thus it needs to remember the original input.
            self.inputs.pp_mc_gpu = self.inputs.hidden_oft_gpu

        # Catch GPU
        if self.inputs.pp_mc_gpu == 'auto':
            self.inputs.pp_mc_gpu = self._get_gpu_from_nvidia_smi()
        else:
            try:
                self.inputs.pp_mc_gpu = [int(gpu) for gpu in self.inputs.pp_mc_gpu.split(',')]
            except ValueError:
                raise ValueError("Get GPU: pp_mc_gpu must be an (list of) integers or 'auto'")

        # Map to DataFrame
        mc_gpu = [str(i) for i in self.inputs.pp_mc_gpu] * math.ceil(self.stacks_len / len(self.inputs.pp_mc_gpu))
        len_diff = len(mc_gpu) - self.stacks_len
        if len_diff:
            mc_gpu = mc_gpu[:-len_diff]

        return mc_gpu

    @staticmethod
    def _get_gpu_from_nvidia_smi():
        """
        Catch available GPUs using nvidia-smi.
        It could be much faster using pyCUDA or something similar, but I want to limit
        the number of library to install for the user.
        """
        nv_uuid = subprocess.run(['nvidia-smi', '--list-gpus'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='ascii')
        nv_processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv'],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='ascii')

        # Catch the visible GPUs.
        if nv_uuid.returncode is not 0 or nv_processes.returncode is not 0:
            raise AssertionError(f'Get GPU: nvidia-smi returned an error: {nv_uuid.stderr}')
        else:
            nv_uuid = nv_uuid.stdout.strip('\n').split('\n')
            visible_gpu = list()
            for gpu in nv_uuid:
                id_idx = gpu.find('GPU ')
                uuid_idx = gpu.find('UUID')

                gpu_id = gpu[id_idx + 4:id_idx + 6].strip(' ').strip(':')
                gpu_uuid = gpu[uuid_idx + 5:-1].strip(' ')

                # Discard the GPU hosting a process.
                if gpu_uuid not in nv_processes.stdout.split('\n'):
                    visible_gpu.append(gpu_id)

        if visible_gpu:
            return visible_gpu
        else:
            raise ValueError(f'Get GPU: {len(nv_uuid)} GPU detected, but none of them is free.')


class MotionCor:
    def __init__(self, inputs, meta_tilt, stack):
        """
        Run Motioncor2 on a given stack.

        :param inputs:      InputParameters

        :type meta_tilt:    DataFrame
        :param meta_tilt:   Subset of the metadata. Describe one tilt-series.

        :type stack:        int
        :param stack:       Tilt-series number.
        """
        self.log = []
        self.stack_padded = f'{stack:03}'
        self.log_filename = f"{inputs.pp_path_motioncor}/{inputs.pp_set_prefix2add}_{self.stack_padded}.log"
        self.first_run = True

        self._run_motioncor(inputs, meta_tilt)

        # If pp_mc_jobs_per_gpu is too high and the device has no memory available, MotionCor
        # gives a Cufft2D error. So rerun missing images if any, but this time one per GPU.
        meta_tilt = self._check_motioncor_output(meta_tilt)
        if len(meta_tilt) > 0:
            logger(f"\n{Colors.r}MotionCor WARNING:\n"
                   f"{TAB}{len(meta_tilt)} images failed. It may be because no memory was available "
                   f"on the device. You may stop the program and decrease pp_mc_jobs_per_gpu.{Colors.reset}\n"
                   f"{TAB}Reprocessing the missing images on at a time... ", nl=True)
            self._run_motioncor(meta_tilt, inputs)

        # Save output.
        self._save2logfile()
        logger(f'MotionCor2: stack{self.stack_padded} processed.', stdout=False)

    def _run_motioncor(self, inputs, meta_tilt):

        job_per_gpu = int(inputs.pp_mc_jobs_per_gpu) if self.first_run else 1
        self.first_run = False

        # Prepare generator for each image. Multiply by GPUs to allow iteration by chunks of GPUs.
        mc_commands = [self._get_command((_in, _out, _gpu), inputs)
                       for _in, _out, _gpu in zip(meta_tilt.raw, meta_tilt.output, meta_tilt.gpu)]

        jobs = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for cmd in mc_commands)

        # Run subprocess by chunks of GPU.
        runs, run = len(mc_commands), 0
        header = f'{Colors.b_b}{Colors.k}stack{self.stack_padded}{Colors.reset}'
        self._update_progress(runs, run, head=header)
        for job in self._yield_chunks(jobs, len(inputs.pp_mc_gpu) * job_per_gpu):
            # From the moment the next line is read, every process in job are spawned.
            for process in [i for i in job]:
                self.log.append(process.communicate()[0].decode('UTF-8'))
                self._update_progress(runs, run, head=header, done='Motion corrected.')
                run += 1

    @staticmethod
    def _get_command(image, inputs):
        if inputs.pp_mc_tif:
            input_motioncor = 'InTiff'
        else:
            input_motioncor = 'InMrc'

        return [inputs.pp_mc_motioncor,
                f'-{input_motioncor}', image[0],
                '-OutMrc', image[1],
                '-Gpu', image[2],
                '-Gain', inputs.pp_mc_gain,
                '-Tol', inputs.pp_mc_tolerance,
                '-Patch', inputs.pp_mc_patch,
                '-Iter', inputs.pp_mc_iter,
                '-Group', inputs.pp_mc_group,
                '-FtBin', str(inputs.hidden_mc_ftbin),
                '-PixSize', str(inputs.pp_set_pixelsize),
                '-Throw', inputs.pp_mc_throw,
                '-Trunc', inputs.pp_mc_trunc]

    @staticmethod
    def _check_motioncor_output(meta_tilt):
        """
        Check that all the motion corrected images are where they need to be.
        Return DataFrame of raw images that need to be re-run.
        """
        return meta_tilt.loc[~meta_tilt['output'].apply(lambda x: os.path.isfile(x))]

    def _save2logfile(self):
        """Gather stdout of every image in one single log file"""

        with open(self.log_filename, 'a') as log:
            log.write('\n'.join(self.log))

    @staticmethod
    def _yield_chunks(iterable, size):
        iterator = iter(iterable)
        for first in iterator:
            yield itertools.chain([first], itertools.islice(iterator, size - 1))

    @staticmethod
    def _update_progress(runs, run, head, done=None):
        """
        Simple progress bar.

        :param runs:    Total number of iterations.
        :param run:     Current iteration.
        :param head:    String to print before the bar.
        :param done:    String to print at 100%.
        """
        bar_length = 15
        progress = (run + 1) / runs if run else 0
        if progress >= 1:
            progress = 1
            status = "{}\r\n".format(done)
        else:
            status = "Corrected..."
        block = int(round(bar_length * progress, 0))
        bar = "#" * block + "-" * (bar_length - block)
        text = "\r{}: [{}] {}% {}".format(head, bar, round(progress * 100), status)
        print(text, end='')


class Stack:
    """
    Called by pre-processing. First create the stack and then call Batchruntomo to align it.

    The structure of this class is quite limited (__init__ needs to do
    everything) because of the way I originally organized the multiprocessing.
    I could also use __call__.
    """

    def __init__(self, task_from_manager):
        """
        Once the stack is created, call Batchruntomo to start the alignment.

        :param task_from_manager:   [stack, meta_tilt, inputs]
        """
        stack, meta_tilt, inputs = task_from_manager
        self.stack_padded = f'{stack:03}'
        self.path = f'{inputs.pp_path_stacks}/stack{self.stack_padded}'
        self.filename_stack = f"{self.path}/{inputs.pp_set_prefix2add}_{self.stack_padded}.st"
        self.filename_fileinlist = f"{self.path}/{inputs.pp_set_prefix2add}_{self.stack_padded}.txt"
        self.filename_rawtlt = f"{self.path}/{inputs.pp_set_prefix2add}_{self.stack_padded}.rawtlt"
        self.log = f'Alignment - {Colors.b_b}{Colors.k}stack{self.stack_padded}{Colors.reset}:\n'

        os.makedirs(self.path, exist_ok=True)

        # To create the template for newstack and the rawtlt,
        # the images needs to be ordered by tilt angles.
        meta_tilt = meta_tilt.sort_values(by='tilt', axis=0, ascending=True)
        self._create_rawtlt(meta_tilt, inputs.pp_path_mdocfiles)

        # Run newstack.
        if inputs.pp_run_stack:
            self._create_template_newstack(meta_tilt)

            t1 = time.time()
            subprocess.run(self._get_newstack(),
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
            t2 = time.time()
            self.log += f'{TAB}Newstack took {t2 - t1:.2f}s.\n'

        elif not os.path.isfile(self.filename_stack):
            raise FileNotFoundError(f'Stack: {self.filename_stack} is not found and pp_run_stack=0.')

        # Run batchruntomo and send the logs to logger.
        if inputs.pp_run_batchruntomo:
            batchruntomo = Batchruntomo(inputs,
                                        self.filename_stack,
                                        self.path)
            self.log += batchruntomo.log

        logger(self.log, nl=True)

    def _create_template_newstack(self, meta_tilt):
        """
        Create a template used by newstack fileinlist.

        template: "
        {nb_images}
        {path/of/image.mrc}
        0
        {path/of/image.mrc}
        0
        ..."
        """
        template = f"{len(meta_tilt)}\n" + '\n0\n'.join(meta_tilt['output']) + '\n0\n'
        with open(self.filename_fileinlist, 'w') as f:
            f.write(template)

    def _create_rawtlt(self, meta_tilt, mdoc_path):
        """
        Create .rawtlt file.
        If mdoc file is not found or is not correct,
        tilts from image filenames will be used.
        """
        mdocfile = glob(f"{mdoc_path}/*_{self.stack_padded}.mrc.mdoc")
        try:
            if len(mdocfile) == 0 or len(mdocfile) > 1:
                raise ValueError

            with open(mdocfile[0], 'r') as f:
                rawtlt = [float(line.replace('TiltAngle = ', '').strip('\n'))
                          for line in f if 'TiltAngle' in line]
            if len(meta_tilt) != len(rawtlt):
                raise ValueError
            rawtlt.sort()
            rawtlt = '\n'.join((str(i) for i in rawtlt)) + '\n'
            self.log += f'{TAB}Mdoc file: True.\n'

        # No mdoc, more than one mdoc or mdoc with missing images.
        except ValueError:
            self.log += f'{TAB}Mdoc file: False.\n'
            rawtlt = '\n'.join(meta_tilt['tilt'].astype(str)) + '\n'

        with open(self.filename_rawtlt, 'w') as f:
            f.write(rawtlt)

    def _get_newstack(self):
        cmd = ['newstack',
               '-fileinlist', self.filename_fileinlist,
               '-output', self.filename_stack,
               '-quiet']
        return cmd


class Batchruntomo:
    """
    The structure of this class is quite limited (__init__ needs to do
    everything) because of the way I originally organized the multiprocessing.
    """

    def __init__(self, inputs, stack_filename, stack_path):

        self.rootname = stack_filename.split('/')[-1].replace('.st', '')
        self.path = stack_path
        self.filename_adoc = f'{self.path}/tool_preprocessing.adoc'
        self.first_run = True
        self.log = f'{TAB}Batchruntomo:\n'

        self._create_adoc(inputs)

        # First run.
        batchruntomo = subprocess.run(self._get_batchruntomo(inputs),
                                      stdout=subprocess.PIPE,
                                      encoding='ascii')
        self.stdout = batchruntomo.stdout

        # There is a small bug in the IMOD install. David said it will be fixed in the next release.
        # Basically, one cannot change the residual report threshold. So run batchruntomo first,
        # up to tiltalign in order to have align.com and then modify align.com and restart from there.
        if 'ABORT SET:' in self.stdout:
            self._get_batchruntomo_log(abort=True)

        # Change the threshold in align.com
        with open(f'{self.path}/align.com', 'r') as align:
            f = align.read().replace('ResidualReportCriterion\t3', 'ResidualReportCriterion\t1')
        with open(f'{self.path}/align.com', 'w') as corrected_align:
            corrected_align.write(f)

        # Second run with correct report threshold...
        batchruntomo = subprocess.run(self._get_batchruntomo(inputs),
                                      stdout=subprocess.PIPE,
                                      encoding='ascii')
        self.stdout += batchruntomo.stdout
        self._get_batchruntomo_log()

    def _create_adoc(self, inputs):
        """Create an adoc file from default adoc or using specified adoc file directly."""

        if inputs.pp_brt_adoc == 'default':
            # Compute bin coarsed using desired pixel size.
            inputs.set_bin_coarsed()

            adoc_file = adoc
            for param in ('pp_brt_gold_size',
                          'pp_brt_rotation_angle',
                          'pp_brt_bin_coarse',
                          'pp_brt_target_nb_beads',
                          'pp_brt_bin_ali'):
                adoc_file = adoc_file.replace(f'<{param}>', f'{getattr(inputs, param)}')

            with open(self.filename_adoc, 'w') as file:
                file.write(adoc_file)
        else:
            if os.path.isfile(inputs.pp_brt_adoc):
                self.filename_adoc = inputs.pp_brt_adoc
            else:
                inputs.pp_brt_adoc = 'default'
                self._create_adoc(inputs)

    def _get_batchruntomo(self, inputs):
        cmd = ['batchruntomo',
               '-DirectiveFile', self.filename_adoc,
               '-RootName', self.rootname,
               '-CurrentLocation', self.path,
               '-CPUMachineList', 'localhost:4',
               '-StartingStep', inputs.pp_brt_start,
               '-EndingStep', f'{6 if int(inputs.pp_brt_end) >= 6 else inputs.pp_brt_end}']

        if self.first_run:
            self.first_run = False
        else:
            cmd[-1] = inputs.pp_brt_end
            cmd[-3] = '6'
        return cmd

    def _get_batchruntomo_log(self, abort=False):
        self.stdout = self.stdout.split('\n')

        if abort:
            for line in self.stdout:
                if 'ABORT SET:' in line:
                    self.log += f'{TAB * 2}{line}.\n'
            return

        # erase.log
        self.log += self._get_batchruntomo_log_erase(f'{self.path}/eraser.log')

        # stats.log, cliphist.log and track.log
        # Easier to catch the info directly in main log.
        for line in self.stdout:
            if 'Views with locally extreme values:' in line:
                self.log += f'{TAB * 2}- Stats: {line}.\n'
            elif 'low SDs or dark regions' in line:
                self.log += f'{TAB * 2}- Cliphist: {line}.\n'
            elif 'total points accepted' in line:
                self.log += f"{TAB * 2}- Autofidseed: {line.split('=')[-1]} beads accepted as fiducials.\n"

        # track.log
        self.log += self._get_batchruntomo_log_track(f'{self.path}/track.log')

        # restricalign.log
        for line in self.stdout:
            if 'restrictalign: Changed align.com' in line:
                self.log += f'{TAB * 2}- Restrictalign: Restriction were applied to statisfy measured/unknown ratio.\n'
                break
            elif 'restrictalign: No restriction of parameters needed' in line:
                self.log += f'{TAB * 2}- Restrictalign: No restriction of parameters needed.\n'
                break

        self.log += self._get_batchruntomo_log_align(self.stdout)

    @staticmethod
    def _get_batchruntomo_log_erase(log):
        # Catch number of pixel replaced and if program succeeded.
        exit_status = 'Not found'
        try:
            with open(log, 'r') as file:
                count_pixel = 0
                for line in file:
                    if 'replaced in' in line:
                        count_pixel += int(line.split()[3])
                    elif 'SUCCESSFULLY COMPLETED' in line:
                        exit_status = 'Succeded'
            log = f'{TAB * 2}- Erase: {count_pixel} pixels were replaced. Exit status: {exit_status}.\n'
        except IOError:
            log = f'{TAB * 2}erase.log is missing... Alignment may have failed.\n'
        return log

    @staticmethod
    def _get_batchruntomo_log_track(log):
        # track.log
        exit_status = 'Not found'
        try:
            with open(log, 'r') as file:
                missing_points = 0
                for line in file:
                    if 'Total points missing =' in line:
                        missing_points = line.split(' ')[-1].strip('\n')
                    elif 'SUCCESSFULLY COMPLETED' in line:
                        exit_status = 'Succeded'
            log = f'{TAB * 2}- Track beads: {missing_points} missing points. Exit status: {exit_status}.\n'
        except IOError:
            log = f'{TAB * 2}track.log is missing... Alignment may have failed.\n'
        return log

    @staticmethod
    def _get_batchruntomo_log_align(log):
        residual_error_mean_and_sd = 'Residual error mean and sd: None.'
        residual_error_weighted_mean = 'Residual error weighted mean: None'
        residual_error_local_mean = 'Residual error local mean: None'
        weighted_error_local_mean = 'Weighted error local mean: None'
        for line in log:
            if 'Residual error mean and sd:' in line:
                residual_error_mean_and_sd = line.strip()
            if 'Residual error weighted mean:' in line:
                residual_error_weighted_mean = line.strip()
            if 'Residual error local mean:' in line:
                residual_error_local_mean = line.strip()
            if 'Weighted error local mean:' in line:
                weighted_error_local_mean = line.strip()

        return (f'{TAB * 2}{residual_error_mean_and_sd}\n'
                f'{TAB * 2}{residual_error_weighted_mean}\n'
                f'{TAB * 2}{residual_error_local_mean}\n'
                f'{TAB * 2}{weighted_error_local_mean}\n')


class Ctffind:
    """
    The structure of this class is quite limited (__init__ or __call__ needs to do
    everything) because of the way I originally organized the multiprocessing.
    """

    def __init__(self, task_from_manager):
        stack, meta_tilt, inputs = task_from_manager
        stack_padded = f'{stack:03}'
        stack_display_nb = f'{Colors.b_b}{Colors.k}stack{stack_padded}{Colors.reset}'
        path_stack = f'{inputs.pp_path_stacks}/stack{stack_padded}'
        filename_log = f"{path_stack}/{inputs.pp_set_prefix2add}_{stack_padded}_ctffind.log"

        self.filename_output = f"{path_stack}/{inputs.pp_set_prefix2add}_{stack_padded}_ctffind.mrc"
        self.log = f'Ctffind - {stack_display_nb}:\n'
        self.stdout = None

        # Get the image closest to 0. Only this one will be used.
        image = meta_tilt.loc[meta_tilt['tilt'].abs().idxmin(axis=0)]

        # Get ctffind command and run.
        os.makedirs(path_stack, exist_ok=True)

        ctf_command, ctf_input_string = self._get_ctffind(image, inputs)
        ctffind_run = subprocess.run(ctf_command,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     input=ctf_input_string,
                                     encoding='ascii')

        if ctffind_run.stderr:
            raise ValueError(f'Ctffind: An error has occurred ({ctffind_run.returncode}) '
                             f'on stack{stack_padded}.')
        else:
            self.stdout = ctffind_run.stdout

        # Save stdout.
        with open(filename_log, 'w') as f:
            f.write(self.stdout)

        # Send the logs back to main.
        self._get_ctffind_log()
        logger(self.log, nl=True)

    def _get_ctffind(self, image, inputs):
        """The inputs will go through stdin. Last input: expert options=no."""
        cmd = [inputs.pp_ctf_ctffind]
        input_dict = [image['output'],
                      self.filename_output,
                      str(inputs.pp_mc_desired_pixelsize),
                      inputs.pp_ctf_voltage,
                      inputs.pp_ctf_cs,
                      inputs.pp_ctf_amp_cont,
                      inputs.pp_ctf_size2compute,
                      inputs.pp_ctf_min_res,
                      inputs.pp_ctf_max_res,
                      inputs.pp_ctf_min_def,
                      inputs.pp_ctf_max_def,
                      inputs.pp_ctf_step_def,
                      inputs.pp_ctf_astig_type,
                      inputs.pp_ctf_exhaustive,
                      inputs.pp_ctf_astig_restraint,
                      inputs.pp_ctf_phase_shift,
                      'no']

        input_string = '\n'.join(input_dict)
        return cmd, input_string

    def _get_ctffind_log(self):
        """Format the stdout of Ctffind and save it as a str."""
        look4 = ['MRC data mode',
                 'Bit depth',
                 'Estimated defocus values',
                 'Estimated azimuth of astigmatism',
                 'Score',
                 'Thon rings with good fit up to',
                 'CTF aliasing apparent from']
        possible_lines = filter(lambda i: i != '', self.stdout.split('\n'))

        for line in possible_lines:
            if any(item in line for item in look4):
                self.log += f'{TAB}{line}\n'
            else:
                continue


class Logger:
    """Save something to log file and/or sent something to stdout."""

    def __init__(self, inputs):
        self.lock = Lock()
        self.filename_log = inputs.pp_path_logfile

    def __call__(self, log, stdout=True, nl=False):
        """
        Send a string to stdout and log file one process at a time.

        :param log:     String to print and save to logfile.
        :param stdout:  print or not.
        :param nl:      Add a newline at the beginning of the message.
        """
        self.lock.acquire()

        nl = '\n' if nl else ''
        message = f'{nl}{Colors.underline}{datetime.now():%d%b%Y-%H:%M:%S}{Colors.reset} - {log}'

        if stdout:
            print(message)
        with open(self.filename_log, 'a') as f:
            f.write(message + '\n')

        self.lock.release()


def preprocessing(inputs):
    """
    Each tilt-series is treated sequentially, but most steps are asynchronous.
    See individual function fore more detail.

    :param inputs:      InputParameters
    """
    # Set output directories.
    os.makedirs(inputs.pp_path_motioncor, exist_ok=True)
    os.makedirs(inputs.pp_path_stacks, exist_ok=True)

    logger('Start preprocessing.')

    metadata_object = Metadata(inputs)
    meta = metadata_object.get_metadata()

    # Once the metadata is loaded, set its pixel size and calculate Ftbin.
    inputs.set_pixelsize(meta)

    # Some stdout.
    logger(f"Collecting data:\n"
           f"{TAB}Raw images: {metadata_object.stacks_len}\n"
           f"{TAB}Tilt-series: {metadata_object.stacks_nb}\n"
           f"{TAB}Possible nb of images per stack: {metadata_object.stacks_images_per_stack}")
    if inputs.pp_run_motioncor:
        logger(f"Starting MotionCor on GPU {', '.join([str(gpu) for gpu in inputs.pp_mc_gpu])}:")

    worker = WorkerManager(metadata_object.stacks_nb, inputs)

    # Compute sequentially each tilt-series.
    # First run MotionCor, then stack and ctffind, both in parallel.
    for tilt_number in metadata_object.stacks_nb:
        meta_tilt = meta[meta['nb'] == tilt_number]

        if inputs.pp_run_motioncor:
            MotionCor(inputs, meta_tilt, tilt_number)

        # Communicate the job to workers.
        job2do = [tilt_number, meta_tilt, inputs]
        if inputs.pp_run_ctffind:
            worker.new_async(Ctffind, job2do)
        if inputs.pp_run_stack or inputs.pp_run_batchruntomo:
            worker.new_async(Stack, job2do)

    # Wrap everything up.
    worker.close()
    metadata_object.save_processed_stack()


if __name__ == '__main__':

    print(f'\n{Colors.bold}CET Toolbox{Colors.reset}\n'
          f'{TAB}- From raw images to aligned stacks.\n'
          f'{TAB}- Version: {Colors.bold}{VERSION}{Colors.reset}')

    inputs_object = InputParameters(descriptor)
    inputs_object.get_inputs()
    inputs_object.save2logfile()
    logger = Logger(inputs_object)
    inputs_object.warnings()

    if inputs_object.pp_run_onthefly:
        print(f'On-the-fly processing: (Tolerated inactivity: {inputs_object.pp_otf_max_time2try}min).')
        otf = OnTheFly(inputs_object)
        otf.run(inputs_object)
    else:
        preprocessing(inputs_object)

    print('Closing.')

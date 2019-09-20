# CET Toolbox
Preprocessing of CET raw images.
From raw images, create motion corrected tilt-series and align them using eTomo programs.

### Create an input file
```
python toolbox.py --create_input_file <input_filename>
# OR
python toolbox.py -c <input_filename>
# OR
python toolbox.py -c
```

### To start the processing
```
python toolbox.py -input <input_filename>
# OR
python toolbox.py -i <input_filename>
# OR
python toolbox.py
```

### Options accessible from the command line
```
python toolbox.py -h
```

## Overview
Run Motioncor2 one tilt-series at a time. The tilt-series are parallelized across the available GPUs (see **ba_mc_gpu** and **ba_mc_jobs_per_gpu**). This is the main advantage of this program, as it can correct an entire stack in a few seconds if one has high-end/multiple GPUs.

Once the images from one tilt-series have been motion corrected, the stacks will be created and aligned, and the defocus at the tilt axis will be estimated by Ctffind. This step is done in parallel of the main process running MotionCor2 and the number of stack allowed to be process at the same time is set by the number of motion corrected stacks or **ad_set_max_cpus** (default: nb of logical cores).
- An on-the-fly mode is available (**--fly** or **ba_run_onthefly**) allowing to process data while it is being acquired by the microscope.
- Some parameters are hidden by default: to show the advanced parameters in the input file or the interactive mode, run the program with **--advanced**

## Requirements
Will only work with Python3.6 or later. No efforts were made to keep the program compatible with older versions. I strongly recommend to run the script from the Anaconda Python3.7 base environment, otherwise Pandas need to be installed.

## Raw filenames
The micrograph file names must have at least 3 information:
  - The stack number.
  - The tilt angle.
  - An extension, either .mrc of .tif.

You can set the format of your micrograph file names using with:
  - **ba_set_field_nb**
  - **ba_set_field_tilt**
  - **ba_set_prefix2look4**
  
The stack number, at the position **ba_set_field_nb**, can be padded with zeros (4, 04, 004, etc.) and decorated with none-digit characters (4, tilt4, [4], stack4.mrc, etc.). It must be an integer.

On the other end, the tilt angles, at the position **ba_set_field_tilt**,  are less flexible: extensions are allowed (42, 42.00, 42.00.mrc, 42.tif, etc.) as well as the '[ ]' decorator ([42] or [42.00].mrc) but any other digit will not be handled correctly.
- Usually we use the following format:
  > **{prefix}**_**{stack_nb}**_*{order}*_**{tilt}.{mrc|tif}** or i.e. **WT**_**011**\_037_**-54.00.tif**
  - *{order}* is not required.

Subfolders: **ba_path_raw** can end with '\*', meaning that the movies are grouped into sub-folders (i.e. raw/stack*). This does not work for **ad_path_motioncor**!

## Outputs
- **ad_path_motioncor**/
  - motioncorred images: {**ba_set_prefix2add**}_{**nb**}_{**tilt**}.mrc
  - MotionCor2 logs (one per stack)
- **ad_path_stacks**/
  - stack{nb}/
    - {**ba_set_prefix2add**}_{**nb**}.st and eTomo outputs.\
    ({nb} - stack number: 3 digit number padded with zeros (001, 002, etc.).
    - Ctffind outputs.
    - **ba_path_logfile**.
    - toolbox_stack_processed.txt

## Input file
- The easiest way to use this program is to use an input file.
  - **--create_input_file** or equivalently **-c** can extract the default parameters into an input file. I strongly recommend to read at least once the available parameters, as it will give you a better idea of what is and isn't possible to do in the current version.
 - "basic" parameters must be in the input file but "advanced" parameters are optional. Empty parameters are allowed.
 - Add **--advanced** to create an advanced input file.

## Interactive mode
- If no argument specified when starting the program, the interactive mode is triggered. It will ask you, one by one, to set the parameters. For each parameter, it gives you the parameter ID, the expected type and the default value. Answering '+' will show the description of this parameter.
- Add **--advanced** to show the advanced parameters.

## Restrict the processing to some stacks
If you want to restrict the processing to some specific stacks, the toolbox allows two type of restrictions:
  - positive: specify the stack number(s) that should only be processed using **ad_run_nb** or **--nb**. This must be an integer or list of integers (padded with zeros or not) separated by comas. Note that this parameter is ignored when the on-the-fly mode is activated.
  - negative: specify the stack number(s) that should *NOT* be processed using the toolbox_stack_processed.txt file. Every time a stack is processed, the program will register the processed stack into this file. You can also edit this file yourself (see **ad_run_overwrite**). This file can be ignored (if you want to reprocess some stacks), using **ad_run_overwrite**=1 or **--overwrite**.

## Restrict the processing to some steps
In any configuration, you can run the program with or without the **--fly** flag. You also have to make sure you specified the correct **ba_path_raw**/**ad_path_motioncor**, **ba_set_field_nb** and **ba_set_field_tilt**.
  - MotionCor2:
    - If activated, **ba_path_raw** must correspond to the path with the movies you want to motion correct.
    - If deactivated, **ba_path_raw** is *NOT* used and **ad_path_motioncor** must correspond to the path with the sums you want to use. Moreover, make sure that both **ba_set_field_nb** and **ba_set_field_tilt** match the motion corrected sums (and not the raw).
  - Activate ONLY Batchruntomo:
    - If you want to use your own stacks for batchruntomo, make sure they have the correct file name format and that they are in the correct path (**ad_path_stacks**/**stack**{**nb**}/**ba_prefix2add**_{**nb**}.st, with {nb} padded :03). NB: **ad_path_motioncor**, **ba_set_field_nb** and **ba_set_field_tilt** must match the motion corrected sums.
    - For the initial tilt angles, the program will generate the rawtlt using the motion corrected sum file names or your mdoc files using **ad_path_mdocs**. You cannot use your own rawtlt files.

## On-the-fly
Briefly, you need to set 2 additional parameters:
  - **ba_otf_max_images_per_stack**: Tolerated time of inactivity after which the program stops.
  - **ba_otf_max_image2try**: Expected number of images per stack (NB: it can handle missing images, no worries).

Careful as it will expect images to be written in order (like a microscope does). For instance, if you transfer your images to your directory in *random* order and start the program to process the stacks while there are being transferred, it will not work correctly.

For a complete description, see OnTheFly.run.

## Mdoc files (TiltAngle)
Mdoc files can be used to specify the tilts for initial alignment in eTomo. The files have to follow this format: "**ad_path_mdocfiles**/*_{**stack_nb**}.mrc.mdoc". **ad_path_mdocfiles** can be modified by the user and {stack_nb} must be a number padded with zeros (3 characters), (ex: 001, 010, 100, etc.). Otherwise, the program uses the tilts from the raw image filenames.

## Batchruntomo
The tilt-series alignment is done by batchruntomo. As this toolbox was made for an emClarity workflow, by default it will generate binned SIRT-like filtered tomograms. See **_brt** parameters for more info. If you want to use your own adoc file, use **ad_brt_adoc** or **--adoc** {file.adoc} 

If you want to run only batchruntomo (use your own stacks), make sure you follow what is mentioned in "RESTRICT THE PROCESSING TO SOME STEPS".

## Parallel processing
The toolbox creates a pool of processes to create/align asynchronously stacks (and to run Ctffind). The number of processes is set to be the number of stacks that need to be processed (limited by **ad_set_max_cpus**) and is effectively the number of tilt-series that can be processed simultaneously. By default, **ad_set_max_cpus** is set to the number of logical cores of your CPU.

Each process will additionally start 4 other processes for batchruntomo (only during expensive tasks). These additional processes are entirely managed by IMOD. I set localhost:4, as this is fine for us (eBIC and STRUBI). You can modify this number (see Batchruntomo._get_batchruntomo), but 4 should be enough. Depending on your system and IMOD install, batchruntomo may not be able to use multiple cores...

## Overwrite
By default, the program will not overwrite a tilt-series. It generates a file (toolbox_stack_processed.txt) gathering the stack number(s) that were already processed by the toolbox. You can modify it yourself. **--overwrite** or **ad_run_overwrite**=1 ignore this file and will reprocess everything (processed stacks will be then added to the the queue no matter what).

## MotionCor2 - GPU IDs
By default, the GPUs are set automatically. If one GPU is hosting one or multiple processes, it will be discarded. Only works with Nvidia devices (nvidia-smi has to be installed). The user can still specify the GPU ID(s), starting from 0.

## MotionCor2 - number of processes per GPUS
At the moment, the user has to define the number of jobs to run simultaneously within the same GPUs using **ba_mc_job_per_gpu**. This is not ideal but to make it automatic I would need to load additional third parties (ex: pyCUDA). Therefore, this step is manual for now. This number mainly depends on the memory of your GPU(s) and the size of your images. If too many jobs are spawn in the device, MotionCor2 will fail. The program will let you know and will restart the jobs that failed setting **ba_mc_job_per_gpu** to 1. One can also play with **ad_mc_gpu_mem_usage**.

## Default parameters
To change the default parameters, you only need to change the 'descriptor' variable bellow and change the corresponding fields. Careful not to mess up the format.

## Acknowledgements
This program was originally created for Peijun Zhang's team (Oxford Uni, STRUBI and eBIC). Zhengyi Yang was the first user and helps for debugging.

### Built With
  - [IMOD](https://bio3d.colorado.edu/imod/): I strongly recommend IMOD >= 4.9.9 mainly because older versions do not to set batchruntomo parameters correctly.
  - [Ctffind4](http://grigoriefflab.janelia.org/ctf): path in **ba_ctf_ctffind**
  - [MotionCor2](https://hpc.nih.gov/apps/RELION/MotionCor2-UserManual-05-03-2018.pdf): path in **ba_mc_motioncor**. Apparently new versions (>1.1.0) may have some problems with multiple MotionCor2 jobs running side-by-side withing the same GPU. I am curently looking at this...



# Partial preprocessing with fMRIPrep

## Basic steps

### Convert PAR/REC to nifti

```bash
subID=002
master -m 02a -s ${subID} -n 2 --lines --sge
```

### Reconstruct lines

```bash
subID=002
master -m 03 -s ${subID} - n2 -e 5 --sge
```

### Preprocess lines
```bash
subID=002
call_lsprep -s sub-${subID} -n 2 --verbose --filter_pca 0.18 --no_button --ica
```

### Run NORDIC

```bash
subID=002
master -m 10 -s ${subID} -n 2 --sge
```

---
## Exotic preprocessing
### Run motion correction with ANTs

First, create a reference image with ANTs

```bash
subID=002

for runID in `seq 1 2`; do
    orig_file=${DIR_DATA_HOME}/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_run-${runID}_acq-3DEPI_bold.nii.gz
    ref_file=$(dirname ${orig_file})/$(basename ${orig_file} .nii.gz)ref.nii.gz

    call_antsreference ${orig_file} ${ref_file}
done
```

use this reference image to warp the brainmask in T1w-space to func-space. Also project the WM-segmentation to func-space because the SDC-workflow report requires one. Now also create a registration file between the two sessions:

```bash
# create transformation mapping ses-2 to ses-1
call_ses1_to_ses --inv sub-${subID} ${sesID}
call_ses1_to_ses sub-${subID} ${sesID}
tfm_fwd=${DIR_DATA_DERIV}/pycortex/sub-${subID}/transforms/sub-${subID}_from-ses${sesID}_to-ses1_desc-genaff.mat
tfm_inv=${DIR_DATA_DERIV}/pycortex/sub-${subID}/transforms/sub-${subID}_from-ses1_to-ses${sesID}_desc-genaff.mat

```

and apply this to the brainmask and white-matter segmentation
```bash
for runID in `seq 1 2`; do

    # set orig file and reference file previously created
    orig_file=${DIR_DATA_HOME}/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_run-${runID}_acq-3DEPI_bold.nii.gz
    ref_file=$(dirname ${orig_file})/$(basename ${orig_file} .nii.gz)ref.nii.gz

    # warp brainmask to func-space
    inv=1
    mov=${DIR_DATA_DERIV}/manual_masks/sub-${subID}/ses-1/sub-${subID}_ses-1_acq-MP2RAGE_desc-spm_mask.nii.gz
    mask=${DIR_DATA_HOME}/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_run-${runID}_acq-3DEPI_desc-brain_mask.nii.gz
    call_antsapplytransforms --gen ${ref_file} ${mov} ${mask} ${tfm_inv}

    # warp white matter segmentation to func-space
    mov=${DIR_DATA_DERIV}/manual_masks/sub-${subID}/ses-1/sub-${subID}_ses-1_acq-MP2RAGE_label-WM_probseg.nii.gz
    wm=${DIR_DATA_HOME}/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_run-${runID}_acq-3DEPI_label-WM_probseg.nii.gz
    call_antsapplytransforms --gen ${ref_file} ${mov} ${wm} ${tfm_inv}

    # enforce 3D images
    for img in ${mask} ${wm}; do
        dm=`fslval ${img} dim0`
        if [[ ${dm} -gt 3 ]]; then
            fslroi ${img} ${img} 0 1
            fslorient -copyqform2sform ${img}
        fi
    done
done
```

run motion correction; before doing so, we'll back up the original files with an extra ``rec``-tag and name the motion corrected files exactly like the original files. That way, the FMAP still has the correct ``IntendedFor``-field.

```bash
for runID in `seq 1 2`; do

    # set orig file
    orig_file=${DIR_DATA_HOME}/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_run-${runID}_acq-3DEPI_bold.nii.gz
    
    # set reference
    ref_file=$(dirname ${orig_file})/$(basename ${orig_file} .nii.gz)ref.nii.gz

    # set output
    out_base=$(dirname ${orig_file})/$(basename ${orig_file} _bold.nii.gz)

    # rename orig file
    new_orig=${out_base}_desc-bold_nomoco.nii.gz
    mv ${orig_file} ${new_orig}

    # get mask
    mask=${DIR_DATA_HOME}/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_run-${runID}_acq-3DEPI_desc-brain_mask.nii.gz
    
    # run; the output will now be named exactly like the original file
    call_antsmotioncorr --in ${new_orig} --mask ${mask} --out ${out_base} --ref ${ref_file} --verbose
done
```

`call_topup` takes these files as input, but it can also look for them in the ``workdir`` as if McFlirt module was run. For this, we need to create additional directories and rename the files.

```bash
wf_folder=${DIR_DATA_SOURCE}/sub-${subID}/ses-2
for runID in `seq 1 2`; do

    # set orig file
    orig_file=${DIR_DATA_HOME}/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_run-${runID}_acq-3DEPI_bold.nii.gz

    # set output
    out_base=$(dirname ${orig_file})/$(basename ${orig_file} _bold.nii.gz)

    # set full working directory
    full_wf=${wf_folder}/single_subject_${subID}_wf/func_preproc_ses_2_task_SRFi_run_${runID}_acq_3DEPI_wf

    # make bold_hmc_wf folder
    mkdir -p ${full_wf}/bold_hmc_wf

    # make mcflirt for RMS-file
    mkdir -p ${full_wf}/bold_hmc_wf/mcflirt
    cp ${out_base}*.rms ${full_wf}/bold_hmc_wf/mcflirt

    # copy motion parameters
    mkdir -p ${full_wf}/bold_hmc_wf/normalize_motion
    cp ${out_base}_desc-motionpars.txt ${full_wf}/bold_hmc_wf/normalize_motion/motion_params.txt
done
```

### Distortion correction (topup)
```bash
call_topup --sub ${subID} --ses 2 --acq 3DEPI --mask ${mask} --wm ${wm}
```

### Confounds
```bash
subID=002
sesID=2
for runID in `seq 1 2`; do
    in_file=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_acq-3DEPI_run-${runID}_desc-preproc_bold.nii.gz

    # tfm_inv describes ses1-to-ses2
    call_confounds -s sub-${subID} -n ${sesID} --in ${in_file} --tfm ${tfm_inv}
done
```

### refine registration to T1w with bbregister
```bash
subID=002
sesID=2

# create transformation mapping ses-2 to ses-1
matrix1=${DIR_DATA_DERIV}/pycortex/sub-${subID}/transforms/sub-${subID}_from-ses${sesID}_to-ses1_desc-genaff.mat

# register
for runID in `seq 1 2`; do

    # define BOLD timeseries
    ref_file=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_acq-3DEPI_run-${runID}_boldref.nii.gz

    # t1w-space as reference
    ref_anat=${DIR_DATA_DERIV}/fmriprep/sub-${subID}/ses-1/anat/sub-${subID}_ses-1_acq-MP2RAGE_desc-preproc_T1w.nii.gz

    # run bbregister
    call_bbregwf --in ${ref_file} --tfm ${matrix1} --ref ${ref_anat} --verbose

done
```

### Denoising with pybest

From here, we can use the ``master`` command again to run pybest. Make sure to specify the ``--func`` flag, which will output nifti-files that we can use for Feat

```bash
master -m 16 -s ${subID} -n ${sesID} --func -t SRFi

```

---
## Feat

### Run level 1 analysis

```bash
# first convert all *tsv-files to 3-column format files
call_onsets2fsl --in ${DIR_DATA_HOME}/${subID}/ses-${sesID}
```

```bash
cd ${DIR_DATA_HOME}/code
./call_feat2 -s sub-${subID} -n ${sesID} -j 10 --pybest #--debug
```
### Run level 2 analysis to average runs

```bash
# inject identity matrix into Feat
call_injectmatrices -s ${subID} -n ${sesID} -l level1 -r "1,2"
```

```bash
# run Feat
cd $DIR_DATA_HOME/code
./call_feat2 -s sub-${subID} -g level2 -l level1 -j 10 --ow
```

### Project stats to line
```bash
subID=002
sesID=2
runID=1
beam_ref=${DIR_DATA_HOME}/sub-${subID}/ses-${sesID}/anat/sub-${subID}_ses-${sesID}_task-SRFa_run-${runID}_acq-1slice_T1w.nii.gz

# project tstat1
img1=${DIR_DATA_DERIV}/feat/level2/sub-${subID}_desc-level1.gfeat/cope1.feat/stats/tstat1.nii.gz
out=$(dirname ${img1})/tstat1_space-line.nii.gz
call_antsapplytransforms --verbose ${beam_ref} ${img1} ${out} identity

# project brain mask
img2=${DIR_DATA_HOME}/sub-${subID}/ses-2/func/sub-${subID}_ses-2_task-SRFi_run-${runID}_acq-3DEPI_desc-brain_mask.nii.gz
out=$(dirname ${img1})/mask_space-line.nii.gz
call_antsapplytransforms --gen --verbose ${beam_ref} ${img2} ${out} identity
```

---
### project stats to surface through transformation files

```bash
subject="sub-008"
gft_dir="${DIR_DATA_DERIV}/feat/level2/${subject}_desc-level1_confs.gfeat"
for z in `seq 1 2`; do
    cpe=${gft_dir}/cope${z}.feat/stats/zstat1.nii.gz
    t1=${DIR_DATA_DERIV}/pycortex/${subject}/transforms/${subject}_from-ses1_to-ses2_rec-motion1_desc-genaff.mat
    t2=${DIR_DATA_DERIV}/fmriprep/${subject}/ses-1/anat/${subject}_ses-1_acq-MP2RAGE_from-T1w_to-fsnative_mode-image_xfm.txt
    call_vol2fsaverage --verbose -o ${gft_dir} -t ${t2},${t1} -i 0,1 -p ${subject}_ses-2_task-SRF ${subject} ${cpe} desc-cope${z}
done

cpe="${DIR_DATA_HOME}/${subject}/ses-2/func/${subject}_ses-2_task-SRFa_run-1_bold.nii.gz"
t1=${DIR_DATA_DERIV}/pycortex/${subject}/transforms/${subject}_from-ses1_to-ses2_rec-motion1_desc-genaff.mat
t2=${DIR_DATA_DERIV}/fmriprep/${subject}/ses-1/anat/${subject}_ses-1_acq-MP2RAGE_from-T1w_to-fsnative_mode-image_xfm.txt
call_vol2fsaverage --gen --verbose -o ${gft_dir} -t ${t2},${t1} -i 0,1 -p ${subject}_ses-2_task-SRF ${subject} ${cpe} desc-beam
```

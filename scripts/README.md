# Scripts

## Partial FOV

As extra confirmation of our planning, we acquire a 3D-EPI partial FOV run with the same experiment as the line experiment (https://github.com/gjheij/LineExps/lineprf). Because `fMRIprep` does not work well with this level of partial FOV, I've had to adapt the pipeline slighty. I now do the following steps (see https://github.com/spinoza-centre/pRFline/tree/main/data for execution):
- Motion/distortion correction (`call_topup`)
- Apply inverse of `from-ses1_to-ses2` matrix to create a `temporary`-space `call_antsapplytransforms` close to `T1w`-space
- Refine this `bold-to-T1w` registration with `fMRIPrep`'s implementation of `bbregister` (`call_bbregwf`)
- Project to volumetric `FSNative`-space via refined registration to `T1w` (`call_antsapplytransforms`)

We can then use `partial_fit.py` to fit a pRF-model to this data (you can check with FreeView that we have reasonable overlap with the structural image despite the limited FOV). Internally, this calls on `pRFmodelFitting` in https://github.com/gjheij/linescanning/blob/main/linescanning/prf.py. The call is as follows:

```bash
# set subject/session
subID=002
sesID=2

# set path
bids_dir=${DIR_DATA_DERIV}/fmriprep
log_dir=${DIR_DATA_SOURCE}/sub-${subID}/ses-${sesID}/sub-${subID}_ses-${sesID}_task-pRF_run-imgs

# submit to cluster or run locally with python
# job="python"
job="qsub -N pfov_${subID} -pe smp 5 -wd /data1/projects/MicroFunc/Jurjen/programs/project_repos/pRFline/logs"

${job} partial_fit.py -s ${subID} -n ${sesID} -b ${bids_dir} -l ${log_dir} -v --fsnative # fit with fsnative
```

## Line

For the line-experiments, we can run `line_fit.py`, which internally preprocessed the functional runs (high/low-pass filtering), and averages over runs and design iterations (2x runs and 3x iterations per run). It also strips the run from it's baseline, because of wonky averaging and selects the ribbon-voxels from the dataframe to limit the demand on resources. We create a separate design matrix from the same screenshots because of different repetition times (`1.111s` vs `0.105`s). First, we need some registration matrices for:

- Mapping `ses-1` to the image closest to the line-scanning acquisition (generally, this will be the `rec-motion1` image). Based on the subject & session IDs, it selects `ses-1` data from `DIR_DATA_DERIV/pymp2rage` and the `ses-${sesID}` data (low-res anatomical image) from `$DIR_DATA_HOME`.

  ```bash
  # subject/session information
  subID=009
  sesID=2

  # shortcuts
  base_path=sub-${subID}/ses-${sesID}
  base=sub-${subID}_ses-${sesID}

  # cmd
  call_ses1_to_motion1 sub-${subID} ${sesID}
  ```

- Registrations mapping each indivual anatomical slice to the slice closest to the partial FOV anatomical data (again, usually this is the `rec-motion1` image), so we can project all the segmentations to each run as accurately as possible. For this, we need to run ITK-Snap a bunch of times and save the files as `from-run1_to-runX.txt`, in `$DIR_DATA_HOME/sub-${subID}/ses-${sesID}/anat`:

  ```bash
  n_runs=3
  subj_dir=$DIR_DATA_HOME/${base_path}/anat
  mov=${subj_dir}/${base}_acq-1slice_run-1_T1w.nii.gz
  for run in `seq 1 ${n_runs}`; do

    if [[ ${run} -eq 1 ]]; then
      call_createident ${subj_dir}/from-run1_to-run${run}.txt
    else
      ref=${subj_dir}/${base}_acq-1slice_run-${run}_T1w.nii.gz

      # open ITK-Snap > ctrl+R  (or 'cmd+R' on mac) > 'manual' > align
      # press the floppy button on the bottom right, save as:
      echo "save image as ${subj_dir}/from-run1_to-run${run}.txt"
      itksnap -g ${ref} -o ${mov}
    fi
  done
  ```

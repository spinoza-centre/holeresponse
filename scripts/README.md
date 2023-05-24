# Scripts

## Partial FOV

As extra confirmation of our planning, we acquire a 3D-EPI partial FOV run with the same experiment as the line experiment (https://github.com/gjheij/LineExps/ActNorm). Because `fMRIprep` does not work well with this level of partial FOV, I've had to adapt the pipeline slighty. I now do the following steps (see https://github.com/spinoza-centre/holeresponse/tree/main/data for execution):
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

from linescanning import (
    utils,
    plotting,
    dataset,
    preproc,
    fitting
)
import nibabel as nb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import transform
from scipy import signal
from .utils import SubjectsDict
from joblib import Parallel, delayed
import holeresponse as hr

opj = os.path.join

class H5Parser(preproc.DataFilter):

    def __init__(
        self,
        h5_files,
        filter_evs=True,
        n_jobs=None,
        subj_dict=None,
        *args,
        **kwargs
        ):

        self.h5_files = h5_files
        self.filter_evs = filter_evs
        self.n_jobs = n_jobs
        
        if subj_dict == None:
            self.subj_dict = SubjectsDict()
        else:
            self.subj_dict = subj_dict

        if isinstance(self.h5_files, str):
            self.h5_files = [self.h5_files]

        if not isinstance(self.n_jobs, int):
            self.n_jobs = len(self.h5_files)

        if len(self.h5_files)>0:
            self.df_func = []
            self.df_avg = []
            self.df_avg_full = []
            self.dict_ribbon = {}
            self.dict_ribbon_full = {}
            self.dict_wm = {}
            self.df_onsets = []
            self.df_onsets_full = []
            self.df_comps = []
            self.df_wm = []
            self.df_filt = []
            self.df_orig = []
            self.df_raw = []
            self.h5_objs = {}
            self.dict_list = []

            self.dict_list = Parallel(n_jobs=self.n_jobs,verbose=True)(
                delayed(self.format_data)(
                    h5,
                    *args,
                    **kwargs
                )
                for h5 in self.h5_files
            )

            # read in subject specific data
            for data_dict in self.dict_list:
                sub_id = list(data_dict.keys())[0]

                self.dict_ribbon[sub_id] = data_dict[sub_id]["ribbon"]
                self.dict_ribbon_full[sub_id] = data_dict[sub_id]["ribbon_full"]
                self.dict_wm[sub_id] = data_dict[sub_id]["wm_ribbon"]
                self.h5_objs[sub_id] = data_dict[sub_id]["obj"]
                self.df_func.append(data_dict[sub_id]["func"])
                self.df_onsets.append(data_dict[sub_id]["onsets"])
                self.df_onsets_full.append(data_dict[sub_id]["onsets_full"])
                self.df_orig.append(data_dict[sub_id]["orig_func"])
                self.df_comps.append(data_dict[sub_id]["comps"])
                self.df_wm.append(data_dict[sub_id]["wm_comps"])
                self.df_avg.append(data_dict[sub_id]["avg"])
                self.df_raw.append(data_dict[sub_id]["raw"])
                self.df_avg_full.append(data_dict[sub_id]["avg_full"])

            # concatenate
            self.df_func = pd.concat(self.df_func)
            self.df_onsets = pd.concat(self.df_onsets)
            self.df_avg = pd.concat(self.df_avg)
            self.df_avg_full = pd.concat(self.df_avg_full)
            self.df_comps = pd.concat(self.df_comps)
            self.df_wm = pd.concat(self.df_wm)
            self.df_raw = pd.concat(self.df_raw)
            self.df_orig = pd.concat(self.df_orig)
            self.df_onsets_full = pd.concat(self.df_onsets_full)
            
            if self.filter_evs:
                expr = ("event_type != response","&","event_type != blink")
                self.df_onsets = utils.select_from_df(self.df_onsets, expression=expr)
                self.df_onsets_full = utils.select_from_df(self.df_onsets_full, expression=expr)
    
    def format_data(
        self, 
        h5, 
        resample_kws={},
        verbose=False,
        compartments=3,
        wm_comps=None,
        excl_runs=True,
        ribbon=None,
        wm=None,
        lp=False,
        lp_kw={},
        unique_ribbon=False,
        use_raw=False,
        task_id=None,
        ribbon_correction=False,        
        *args, 
        **kwargs
        ):

        # initiate output
        data_dict = {}

        # read bids properties
        comps = utils.split_bids_components(h5)
        sub_id = f"sub-{comps['sub']}"
        data_dict[sub_id] = {}

        # parse h5-file
        kwargs= utils.update_kwargs(
            kwargs,
            "verbose",
            verbose
        )
        ddict = self.read_single_h5(h5, *args, **kwargs)

        # basic format
        if use_raw:
            func = ddict["obj"].df_func_raw.copy()
        else:
            func = ddict["func"]

        onsets = ddict["onsets"]
        if isinstance(task_id, str):
            for tmp_df in [func,onsets]:
                
                idx = list(tmp_df.index.names)
                add_task = tmp_df.reset_index()
                add_task["task"] = task_id
                idx.insert(1, "task")
                tmp_df = add_task.set_index(idx)

        # apply filtering
        if len(lp_kw)>0:
            add_txt = f" with options: {lp_kw}"
        else:
            add_txt = ""

        utils.verbose(f"Low-pass filtering data{add_txt}", verbose)
        super().__init__(
            func,
            filter_strategy="lp",
            lp_kw=lp_kw
        )
        df_filt = self.get_result()

        # use low-passed data or not
        if lp:
            active_func = df_filt.copy()
        else:
            active_func = func.copy()

        # excl runs (defined in utils.py)
        if excl_runs:
            filt_ddict = {}
            for tag,df in zip(["func","onsets"],[active_func, onsets]):
                filt_ddict[tag] = self.exclude_runs(
                    df, 
                    subject=sub_id,
                    subj_dict=self.subj_dict
                )
        else:
            filt_ddict = {
                "func": active_func.copy(),
                "onsets": onsets
            }            

        # ribbon format - nr of voxels differs per subject, so store in dictionary (Marco would be proud)
        has_dict = False
        if not isinstance(ribbon, (list,tuple)):
            has_dict = self.subj_dict.has_ribdict(sub_id)
            # check if we have run-specific ribon values   
            if not has_dict:
                ribbon = self.subj_dict.get_ribbon(sub_id)

                if ribbon_correction:
                    ribbon = self.correct_ribbon(
                        sub_id, 
                        ribbon,
                        verbose=verbose,
                        subj_dict=self.subj_dict
                    )
        
        # print(has_dict)
        ddict_rib = {}
        for tag,df in zip(["ribbon", "ribbon_full"], [filt_ddict["func"], active_func]):
            if not has_dict or not unique_ribbon:
                ddict_rib[tag] = utils.select_from_df(
                    df, 
                    expression="ribbon", 
                    indices=ribbon
                )
            else:
                ddict_rib[tag] = self.fetch_unique_ribbon(
                    df,
                    subject=sub_id,
                    correct=ribbon_correction,
                    verbose=verbose,
                    subj_dict=self.subj_dict
                )

        # get white matter voxels
        if not isinstance(wm, (list,tuple)):
            wm = self.subj_dict.get_wm(sub_id)

        wm_tmp = utils.select_from_df(
            filt_ddict["func"], 
            expression="ribbon", 
            indices=wm
        )

        # invert if necessary
        inv_rib = self.subj_dict.get_invert(sub_id)
        if inv_rib:
            utils.verbose(f"Inverting ribbon order", verbose)
            for key,val in ddict_rib.items():
                ddict_rib[key] = val[val.columns[::-1]]

        df_tmp = []
        for df,tag in zip([ddict_rib["ribbon"], wm_tmp],["gm","wm"]):
            df_avg = pd.DataFrame(df.mean(axis=1))
            df_avg.rename(columns={0: tag}, inplace=True)
            df_tmp.append(df_avg)
        
        df_tmp = pd.concat(df_tmp, axis=1)
        df_avg_full = pd.DataFrame(ddict_rib["ribbon_full"].mean(axis=1))
        df_avg_full.rename(columns={0: "avg"}, inplace=True)

        # make 3-compartment model so we can average depths across subjects
        utils.verbose(f"Making {compartments} GM-compartments", verbose)
        df_comp = self.make_compartment_model(
            ddict_rib["ribbon"], 
            compartments=compartments,
            **resample_kws
        )

        # take 25% of GM for WM
        if not isinstance(wm_comps, int):
            wm_comps = int(round((compartments*0.25),0)) 
        
        utils.verbose(f"Making {wm_comps} WM-compartments", verbose)
        wm_comp = self.make_compartment_model(
            wm_tmp, 
            compartments=wm_comps,
            **resample_kws
        )

        data_dict[sub_id]["ribbon"] = ddict_rib["ribbon"].copy()
        data_dict[sub_id]["ribbon_full"] = ddict_rib["ribbon_full"].copy()
        data_dict[sub_id]["func"] = filt_ddict["func"]
        data_dict[sub_id]["onsets_full"] = onsets
        data_dict[sub_id]["onsets"] = filt_ddict["onsets"]
        data_dict[sub_id]["orig_func"] = active_func
        data_dict[sub_id]["comps"] = df_comp
        data_dict[sub_id]["wm_comps"] = wm_comp
        data_dict[sub_id]["obj"] = ddict["obj"]
        data_dict[sub_id]["raw"] = ddict["obj"].df_func_raw
        data_dict[sub_id]["avg"] = df_tmp
        data_dict[sub_id]["avg_full"] = df_avg_full
        data_dict[sub_id]["wm_ribbon"] = wm_tmp.copy()
        utils.verbose(f"Done with '{sub_id}'\n", verbose)

        return data_dict

    @classmethod
    def correct_ribbon(
        self, 
        subject, 
        ribbon, 
        verbose=False,
        subj_dict=None):

        if subj_dict == None:
            subj_dict = SubjectsDict()

        corr = subj_dict.get(subject, "ribbon_correction")
        utils.verbose(f"Correcting voxel selection with {corr} voxels (positive = right | negative = left)", verbose)
        ribbon = tuple([i+corr for i in ribbon])

        return ribbon

    @classmethod
    def fetch_unique_ribbon(
        self, 
        func,
        rib_dict=None,
        subject=None,
        correct=False,
        subj_dict=None,
        **kwargs
        ):

        if subj_dict == None:
            subj_dict = SubjectsDict()
                    
        # get ribbon dict | organized in dictionaries collecting task and runs as keys and values, respectively
        if not isinstance(rib_dict, dict):
            rib_dict = subj_dict.get(subject, "rib_dict")

        # loop over tasks in self.df_func
        df_ribbon = []
        task_ids = utils.get_unique_ids(func, id="task")
        for task in task_ids:
            task_df = utils.select_from_df(func, expression=f"task = {task}")
            
            # get run IDs
            rib_runs = []
            run_ids = utils.get_unique_ids(task_df, id="run")
            for run in run_ids:
                run_df = utils.select_from_df(task_df, expression=f"run = {run}")
                ribbon = rib_dict[task][f"run-{run}"]

                    # correct ribbon location based on functional outcomes
                if correct:
                    ribbon = self.correct_ribbon(
                        subject, 
                        ribbon, 
                        subj_dict=subj_dict,
                        **kwargs
                    )

                # print(f" Specific ribbon for task-{task} & run-{run}: {ribbon}")
                rib_df = utils.select_from_df(
                    run_df,
                    expression="ribbon",
                    indices=ribbon
                )
                
                # synchronize columns
                diff = np.diff(ribbon)
                rib_df.columns = np.arange(0,diff)
                rib_runs.append(rib_df)

            rib_runs = pd.concat(rib_runs)
            df_ribbon.append(rib_runs)

        if len(df_ribbon)>0:
            df_ribbon = pd.concat(df_ribbon)

        return df_ribbon
    
    @classmethod
    def exclude_per_task(self, df, subject=None):

        try:
            task_ids = utils.get_unique_ids(df, id="task")
        except:
            task_ids = None

        if isinstance(task_ids, list):

            new = []
            task_ids = utils.get_unique_ids(df, id="task")
            for task in task_ids:
                
                # get task df
                task_df = utils.select_from_df(df, expression=f"task = {task}")
                new.append(
                    self.exclude_runs(
                        task_df,
                        subject=subject,
                        task=task,
                        subj_dict=self.subj_dict
                    )
                    
                )

            new = pd.concat(new)
        else:
            new = self.exclude_runs(
                df,
                subject=subject,
                subj_dict=self.subj_dict
            )

        return new

    @classmethod
    def exclude_runs(
        self, 
        df, 
        subject=None, 
        task=None,
        subj_dict=None
        ):

        if subj_dict == None:
            subj_dict = SubjectsDict()

        # get task-specific runs to exclude
        all_runs = utils.get_unique_ids(df, id="run", as_int=True)

        if isinstance(task, str):
            excl_runs = subj_dict.get_excl_runs(subject, task=task)
        else:
            excl_runs = subj_dict.get_excl_runs(subject)

        keep_runs = [i for i in all_runs if i not in excl_runs]
        
        new = []
        for j in keep_runs:
            new.append(utils.select_from_df(df, expression=f"run = {j}"))

        if len(new)>0:
            return pd.concat(new)
        else:
            raise ValueError(f"No dataframes to concatenate. Did you specify the correct runs to exclude?")
    
    @staticmethod
    def upsample2d(
        df, 
        n, 
        axis=1, 
        **kwargs
        ):

        add_index = False
        is_df = True
        if isinstance(df, pd.DataFrame):
            is_df = True
            if len(df.index.names)>0:
                add_index = True
            values = df.values
        else:
            values = df.copy()

        if axis == 1:
            tfm = transform.resize(values, (values.shape[0],n), **kwargs)
        else:
            tfm = transform.resize(values, (n,values.shape[1]), **kwargs)
        
        if is_df:
            if add_index:
                ret_data = pd.DataFrame(tfm, index=df.index)
            else:
                ret_data = pd.DataFrame(tfm)
        else:
            ret_data = tfm.copy()
        
        return ret_data   

    @classmethod
    def make_compartment_model(
        self, 
        df, 
        compartments=3, 
        **kwargs
        ):

        # we need to interpolate if the number of compartments is greater than data shape
        if compartments>df.shape[-1]:
            return self.upsample2d(df, compartments, **kwargs)
        else:
            vox = np.arange(0, df.shape[-1])
            vox3comp = np.array_split(vox, compartments)
            
            comps_ = []
            for c in vox3comp:
                data = utils.select_from_df(df, expression="ribbon", indices=list(c)).mean(axis=1)
                comps_.append(data)

            return pd.concat(comps_, axis=1)

    @staticmethod
    def read_single_h5(
        h5_file, 
        return_type="psc",
        **kwargs):

        dd = dataset.Dataset(
            h5_file,
            **kwargs
        )

        func = dd.fetch_fmri(dtype=return_type)
        onsets = dd.fetch_onsets()

        return {
            "func": func,
            "onsets": onsets,
            "obj": dd
        }
    
    @classmethod
    def merge_pars_stim_values(
        self,
        df1,
        df2,
        desc="stim_size",
        ):

        # get old index
        old_index = list(df1.index.names)

        sorter = [i for i in old_index if i != "vox"]
        # list with new dfs
        t_dfs = []

        # loop through subjects
        sub_ids = utils.get_unique_ids(df1, id="subject")
        for sub in sub_ids:
            
            expr = f"subject = {sub}"
            s_data = utils.select_from_df(df1, expression=expr)
            
            # loop through bins
            ev_ids = utils.get_unique_ids(df1, id="event_type")
            for b in ev_ids:
                
                # select bin-specific parameters
                ev_df = utils.select_from_df(s_data, expression=f"event_type = {b}")

                # select EPI values
                vals = utils.multiselect_from_df(df2, expression=[f"subject = {sub}", f"event_type = {b}"]).values[0][0]
                ev_df[desc] = vals
                t_dfs.append(ev_df)
                
        new_df = pd.concat(t_dfs).sort_values(sorter)

        try:
            new_df.set_index(old_index, inplace=True)  
        except:
            pass

        return new_df
    
    @classmethod
    def merge_dataframes(self, dfs):
        ref_index = list(dfs[0].index.names)
        new_df = []
        for ix,df in enumerate(dfs):
            df["task"] = f"task{ix+1}"
            new_df.append(df)

        new_df = pd.concat(new_df)

        return new_df.groupby(ref_index).mean()
    
class CorrelationPlotter():

    def __init__(
        self, 
        slice_file: str=None,
        stats_file: str=None,
        beam_file: str=None,
        mask_file: str=None,
        line_stats: np.ndarray=None,
        t_thr: float=2.3,
        vmax=7,
        **kwargs):

        self.slice_file = slice_file
        self.stats_file = stats_file
        self.beam_file = beam_file
        self.mask_file = mask_file
        self.t_thr = t_thr
        self.line_stats = line_stats
        self.vmax = vmax

        # read files into numpy arrays
        for tag,ff in zip(
            ["slice", "stats", "beam", "mask"],
            [self.slice_file, self.stats_file, self.beam_file, self.mask_file]):

            if isinstance(ff, str):
                if os.path.exists(os.path.exists(ff)):
                    img = np.rot90(nb.load(ff).get_fdata())
                    setattr(self, f"{tag}_img", img)

        # get arrays for positive and negative t-stats
        self.t_pos, self.t_neg = self._split_stat_array(thr=self.t_thr)

        # prepare bunch of colormaps
        for i,tag,func in zip(
            ["cm_beam", "cm_mask", "cm_pos", "cm_neg"], 
            ["red", "white", "pos", "neg"],
            [utils.make_binary_cm,utils.make_binary_cm,utils.make_stats_cm,utils.make_stats_cm]):
            
            # make colormap
            cm = func(tag)
            setattr(self,i,cm)

    def plot_images(
        self, 
        axs=None, 
        figsize=(8,8), 
        extent=[0,720,0,720], # left-right, up-down 
        inset_axis=None,
        inset_extent=[310,410,310,410],
        add_cross=False,
        skip_y=False,
        y_label=None,
        fontsize=22,
        **kwargs):
        
        if not axs:
            fig,self.img_axs = plt.subplots(figsize=figsize)
        else:
            self.img_axs = axs

        if not isinstance(extent, list):
            plot_slice = self.slice_img
            plot_beam = self.beam_img
            plot_pos = self.t_pos
            plot_neg = self.t_neg
        else:
            plot_slice = self.slice_img[extent[0]:extent[1],extent[2]:extent[3],:]
            plot_beam = self.beam_img[extent[0]:extent[1],extent[2]:extent[3],:]
            plot_pos = self.t_pos[extent[0]:extent[1],extent[2]:extent[3]]
            plot_neg = self.t_neg[extent[0]:extent[1],extent[2]:extent[3]]
            
        self.img_axs.imshow(plot_slice, cmap="gray")
        self.img_axs.imshow(plot_beam, cmap=self.cm_beam, alpha=0.4)
        self.img_axs.imshow(plot_pos, cmap=self.cm_pos, vmin=self.t_thr, vmax=self.vmax)
        self.img_axs.imshow(plot_neg, cmap=self.cm_neg, vmin=self.t_thr, vmax=self.vmax)

        plotting.conform_ax_to_obj(ax=self.img_axs, font_size=fontsize, **kwargs)

        if add_cross:
            self.img_axs.axvline(plot_neg.shape[0]//2, lw=0.5, color='white')
            self.img_axs.axhline(plot_neg.shape[1]//2, lw=0.5, color='white')
        
        if isinstance(y_label, str):
            self.img_axs.set_ylabel(y_label, fontsize=fontsize)
            self.img_axs.set_yticklabels([])
            self.img_axs.set_xticklabels([])
            self.img_axs.set_xticks([])
            self.img_axs.set_yticks([])
        else:
            self.img_axs.axis("off")

        if isinstance(inset_axis, (list,int,float)):
            if isinstance(inset_axis, (int,float)):
                inset_axis = [(ii+1)*inset_axis for ii in range(4)]
            
            plot_slice = self.slice_img[inset_extent[0]:inset_extent[1],inset_extent[2]:inset_extent[3],:]
            plot_beam = self.beam_img[inset_extent[0]:inset_extent[1],inset_extent[2]:extent[3],:]
            plot_pos = self.t_pos[inset_extent[0]:inset_extent[1],inset_extent[2]:inset_extent[3]]
            plot_neg = self.t_neg[inset_extent[0]:inset_extent[1],inset_extent[2]:inset_extent[3]]

            axs2 = self.img_axs.inset_axes(inset_axis)
            axs2.imshow(plot_slice, cmap="gray")
            axs2.imshow(plot_beam, cmap=self.cm_beam, alpha=0.4)
            axs2.imshow(plot_pos, cmap=self.cm_pos, vmin=self.t_thr, vmax=self.vmax)
            axs2.imshow(plot_neg, cmap=self.cm_neg, vmin=self.t_thr, vmax=self.vmax)

            plotting.conform_ax_to_obj(ax=axs2)

            if add_cross:
                axs2.axvline(plot_neg.shape[0]//2, lw=0.5, color='white')
                axs2.axhline(plot_neg.shape[1]//2, lw=0.5, color='white')

            axs2.axis("off")

    def _split_stat_array(self, thr=2.3):

        # positive
        stats_pos = np.full_like(self.stats_img, np.nan)
        stats_pos[self.stats_img>thr] = self.stats_img[self.stats_img>thr]

        # negative, but make absolute so plotting is easier
        stats_neg = np.full_like(self.stats_img, np.nan)
        stats_neg[self.stats_img<-thr] = abs(self.stats_img[self.stats_img<-thr])

        return stats_pos,stats_neg

    def plot_tstat_correlation(
        self,
        axs=None,
        figsize=(6,6),
        **kwargs):

        if not axs:
            self.fig,self.corr_axs = plt.subplots(figsize=figsize)
        else:
            self.corr_axs = axs

        # brain mask slice/line to get relevant voxels
        self.mask_beam = self.mask_img[np.where(self.beam_img.squeeze()>0)].reshape((16,720)).mean(axis=0)
        self.line_idc = np.where(self.mask_beam>0.95)[0]

        # for each relevant voxel, get the distance in mm to the target; red is further to the right, blue is further to the left
        self.dist_to_targ = [(i-self.beam_img.shape[0]//2)*0.25 for i in self.line_idc]

        # fetch stats in beam space
        self.stats_in_line = self.stats_img[np.where(self.beam_img.squeeze()>0)].reshape((16,720)).mean(axis=0)

        # plot t-stats from line vs 3D-EPI
        self.corr = plotting.LazyCorr(
            x=self.line_stats[self.line_idc],
            y=self.stats_in_line[self.line_idc],
            axs=self.corr_axs,
            x_label="t-stats line",
            y_label="t-stats 3D-EPI",
            color_by=self.dist_to_targ,
            **kwargs)      

class T2StarSlices(SubjectsDict):

    def __init__(
        self, 
        subject, 
        transpose=True, 
        ses=None, 
        set_kw={},
        **kwargs
        ):
        super().__init__(**set_kw)
        self.subject = subject

        if not isinstance(ses, int):
            self.session = self.get_session(self.subject)
        else:
            self.session = ses

        self.ribbon = self.get_ribbon(self.subject)
        self.invert = self.get_invert(self.subject)
        self.transpose = transpose

        self.df,self.f_line = self.load_data(**kwargs)
    
    def plot_intensity(
        self, 
        axs=None, 
        figsize=(5,5), 
        avg=False,
        normalize=False,
        run_ix=0,
        plot_kw={},
        **kwargs
        ):

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)

        if not hasattr(self, "df"):
            df,f_line = self.load_data(**kwargs)
        else:
            df = self.df.copy()
            f_line = self.f_line

        # plot average or single run | default = run-1
        lbl = "T$_2$*"
        if avg:

            # normalize data
            if normalize:    
                lbl = "norm(T$_2$*)"
                df = (df-df.min())/(df.max()-df.min())

            tc_sem = df.sem().values
            tc = df.mean().values
        else:
            tc_sem = None
            tc = df.iloc[run_ix,:].values

            if normalize:    
                lbl = "norm(T$_2$*)"
                tc /= tc.max()

        # set default title
        if f_line:
            txt = "line"
            x_lbl = "voxels"
            x_ticks = np.linspace(0, 720, num=5, endpoint=True, dtype=int)
        else:
            txt = "depth"
            x_lbl = "% away from pial"
            x_ticks = None

        for kw,el in zip(
            ["title","x_ticks"],
            [f"intensity across {txt}",x_ticks]):

            plot_kw = utils.update_kwargs(
                plot_kw,
                kw,
                el
            )

        plotting.LazyPlot(
            tc,
            xx=list(df.columns),
            axs=axs,
            error=tc_sem,
            color="r",
            line_width=3,
            x_label=x_lbl,
            y_label=lbl,
            **plot_kw
        )

    def load_data(
        self, 
        ribbon=None, 
        invert=None
        ):

        full_line = True
        if isinstance(ribbon, tuple):
            full_line = False
            if len(ribbon)==0:
                ribbon = self.ribbon

            if not isinstance(invert, bool):
                invert = self.invert
        else:
            invert = False
            
        slices = self.get_slices()

        ddata = []
        for slc in slices:
            dd = nb.load(slc).get_fdata().squeeze()

            if isinstance(ribbon, (tuple,list)):
                rib_vals = dd[ribbon[0]:ribbon[1],352:368].mean(axis=1).T
            else:
                # take full line
                rib_vals = dd[:,352:368].mean(axis=1).T

            ddata.append(rib_vals)

        ddata = pd.DataFrame(ddata)

        if invert:
            ddata = ddata.iloc[:,::-1]

        if isinstance(ribbon, (tuple,list)):
            depth = np.linspace(0, 100, ddata.shape[1])
            ddata.columns = depth

        return ddata,full_line

    def get_slices(self):
        all_files = utils.FindFiles(opj(self.proj_dir, self.subject, f"ses-{self.session}", "anat"), extension="nii.gz").files
        slice_files = utils.get_file_from_substring(["acq-1slice"], all_files)

        return slice_files
    
def average_tasks(df):

    dfs = []
    sub_ids = utils.get_unique_ids(df, id="subject")
    for sub in sub_ids:
        sub_df = utils.select_from_df(df, expression=f"subject = {sub}")
        task_ids =  utils.get_unique_ids(sub_df, id="task")
        task_df = []
        for task in task_ids:
            tmp_df = utils.select_from_df(sub_df, expression=f"task = {task}")
            run_ids = utils.get_unique_ids(tmp_df, id="run")

            run_list = []
            for run in run_ids:
                run_df = utils.select_from_df(tmp_df, expression=f"run = {run}")
                run_vals = run_df.values
                run_list.append(run_vals[...,np.newaxis])

            run_values = np.concatenate(run_list, axis=1).mean(axis=1)
            tmp_run = pd.DataFrame(run_values, columns=["onset"])
            tmp_run["subject"] = utils.get_unique_ids(tmp_df, id="subject")[0]
            tmp_run["task"],tmp_run["event_type"] = task,run_df.reset_index()["event_type"].values

            tmp_run.set_index(["subject","task","event_type"], inplace=True)
            task_df.append(tmp_run)
        
        task_df = pd.concat(task_df)
        dfs.append(task_df)

    return pd.concat(dfs)

def melt_epochs(df):
    sub_ids = utils.get_unique_ids(df, id="subject")
    ev_ids = utils.get_unique_ids(df, id="event_type")

    df_ev = []

    for ev in ev_ids:
        ev_df = utils.select_from_df(df, expression=f"event_type = {ev}")
        for sub in sub_ids:
            sub_df = utils.select_from_df(ev_df, expression=f"subject = {sub}")
            task_ids = utils.get_unique_ids(sub_df, id="task")

            ep_df = []
            start = 0
            for task in task_ids:
                task_df = utils.select_from_df(sub_df, expression=f"task = {task}")
                run_ids = utils.get_unique_ids(task_df, id="run")

                for run in run_ids:
                    run_df = utils.select_from_df(task_df, expression=f"run = {run}")

                    tmp = run_df.reset_index()
                    tmp.drop(["task","run"], axis=1, inplace=True)

                    epoch_ids = utils.get_unique_ids(run_df, id="epoch")
                    epoch_col = tmp["epoch"].values

                    new_ids = []
                    new_epochs = np.arange(start,start+len(epoch_ids))
                    # print(f"task-{task} | run-{run} | {new_epochs}")
                    for i in epoch_col:
                        for ix,ii in enumerate(epoch_ids):
                            if i == ii:
                                new_ids.append(new_epochs[ix])
                    
                    tmp["epoch"] = np.array(new_ids)

                    tmp["subject"],tmp["event_type"] = sub, ev
                    tmp.set_index(["subject","event_type","epoch","t"], inplace=True)
                    ep_df.append(tmp)

                    start += len(epoch_ids)

            df_ev.append(pd.concat(ep_df))

    return pd.concat(df_ev)
                
def make_single_df(func, idx=["subject","run","t"]):

    new_func = []

    subj_ids = utils.get_unique_ids(func, id="subject")
    for sub in subj_ids:
        
        sub_df = utils.select_from_df(func, expression=f"subject = {sub}")
        task_ids = utils.get_unique_ids(sub_df, id="task")
        rr = 1
        for task in task_ids:

            expr = f"task = {task}"
            t_func = utils.select_from_df(sub_df, expression=expr)

            run_ids = utils.get_unique_ids(t_func, id="run")
            for run in run_ids:

                expr = f"run = {run}"
                r_func = utils.select_from_df(t_func, expression=expr).reset_index().drop(["task"], axis=1)
                r_func["run"] = rr
                new_func.append(r_func)

                rr += 1

    df_func = pd.concat(new_func).set_index(idx)
    
    return df_func

def single_weights(df, ref):

    if isinstance(df, pd.DataFrame):
        df = df.values

    if ref.ndim>1:
        ref = ref.squeeze()

    weights = []
    for d in range(df.shape[1]):
        depth_hrf = df[:,d]
        # corr_coeff = np.corrcoef(avg_hrf,d_vals)[0,1]
        w_depth = sum(depth_hrf*ref)/sum(ref)
        weights.append(w_depth)

    weights = np.array(weights)
    return weights


def weighted_hrf_depth(
    avg,
    subjs,
    hrf=None,
    ev="act",
    bsl=20,
    zscore=False,
    detrend=False,
    norm=False,
    ):

    # get average HRF profile
    if not isinstance(hrf, (np.ndarray,pd.DataFrame)):
        avg_hrf = avg.mean(axis=1).values
    else:
        if isinstance(hrf, np.ndarray):
            avg_hrf = hrf.copy()
        elif isinstance(hrf, pd.DataFrame):
            avg_hrf = hrf.values

    # get subject-specific profiles over depth & correct baseline because we did the same in the plots above
    ev1 = utils.select_from_df(subjs, expression=f"event_type = {ev}")

    # loop through subjects
    sub_ids = utils.get_unique_ids(ev1, id="subject")
    sub_weights = []
    sub_zscore = []
    for sub in sub_ids:
        sub_vals = utils.select_from_df(ev1, expression=f"subject = {sub}")
        sub_shift = fitting.Epoch.correct_baseline(sub_vals, bsl=bsl).values
        sub_w = single_weights(sub_shift, avg_hrf)

        if detrend:
            sub_w = signal.detrend(sub_w)

        sub_weights.append(sub_w)

        # zscore?
        if zscore:
            m_ = sub_w.mean()
            s_ = sub_w.std()
            
            sub_zscore.append((sub_w-m_)/s_)
        elif norm:
            m_ = sub_w.mean()
            sub_zscore.append((sub_w-m_))

    gr_ = np.concatenate([i[...,np.newaxis] for i in sub_weights], axis=1)

    avg_corr = gr_.mean(axis=1)
    avg_sem = pd.DataFrame(gr_).sem(axis=1).values
    ddict = {
        "subjs": sub_weights,
        "avg": avg_corr,
        "sem": avg_sem,
        "hrf": avg_hrf.copy(),
        "zscore": zscore,
        "norm": norm
    }

    if len(sub_zscore)>0:
        gr_z = np.concatenate([i[...,np.newaxis] for i in sub_zscore], axis=1)
        avg_z = gr_z.mean(axis=1)
        sem_z = pd.DataFrame(gr_z).sem(axis=1).values

        for key,val in zip(["subjs_z","avg_z","sem_z"],[sub_zscore,avg_z,sem_z]):
            ddict[key] = val
        
    return ddict

class MediumAnnulusGLM(SubjectsDict):

    def __init__(
        self, 
        hr_kws={},
        *args,
        **kwargs
        ):

        # init class
        super().__init__(**hr_kws)

        # run
        self.glms,self.ratios = self.run_subjects_glms(*args, **kwargs)

    @classmethod
    def run_subjects_glms(self, df, **kwargs):
        
        sub_ids = utils.get_unique_ids(df, id="subject")
        sub_glms = {}
        for sub in sub_ids:
            sub_df = utils.select_from_df(df, expression=f"subject = {sub}")
            sub_glms[sub] = self.run_single_glm(sub_df, **kwargs)

        sub_ratios = []
        for key,val in sub_glms.items():
            sub_ratios.append(val["ratio"])

        sub_ratios = pd.DataFrame(sub_ratios, columns=["ratio"])
        sub_ratios["subject"] = sub_ids
        sub_ratios.set_index(["subject"], inplace=True)

        return sub_glms,sub_ratios
    
    @classmethod
    def get_predictions(self, glm_obj):
        dm,betas,data = glm_obj["dm"],glm_obj["betas"],glm_obj["data"]
        pred1 = dm[:,:2]@betas[:2]
        pred2 = dm[:,::2]@betas[::2]

        return [data,pred1,pred2],glm_obj["t"]
    
    def plot_predictions(
        self, 
        subject, 
        axs=None,
        figsize=(5,5),
        **kwargs
        ):

        if not hasattr(self, "glms"):
            raise AttributeError(f"{self} does not have 'glms' attribute.. Make sure to run 'run_subjects_glms()'")
        
        # initialize axes
        if not isinstance(axs, (mpl.axes._axes.Axes,mpl.figure.SubFigure)):
            _,ax = plt.subplots(figsize=figsize)
        else:
            if isinstance(axs, mpl.figure.SubFigure):
                ax = axs.subplots()
            else:
                ax = axs

        data_list,t_ = self.get_predictions(self.glms[subject])
        def_dict = {
            "line_width": [1,3,3],
            "color": ["#cccccc"]+self.ev_colors[::2],
            "markers": [".",None,None],
            "add_hline": 0,
            "labels": ["medium","y_center","y_large"],
            "y_label": "magnitude (%)",
            "x_label": "time (s)"
        }

        for key,val in def_dict.items():
            kwargs = utils.update_kwargs(
                kwargs,
                key,
                val
            )

        pl = plotting.LazyPlot(
            data_list,
            xx=t_,
            axs=ax,
            **kwargs
        )

        return pl
        
    @staticmethod
    def run_single_glm(
        df,
        order=[1,2],
        time_col="t",
        **kwargs
        ):

        # use plotepochprofiles class to format profiles exactly the same as the plots
        ev_epoch = hr.viz.PlotEpochProfiles(
            utils.select_from_df(df, expression="ribbon", indices=[0]).groupby(["subject","event_type", "epoch","t"]).mean(),
            skip_plot=True,
            **kwargs         
        )

        # select center/large annulus timecourses; make 2d array (rows = time, columns = events)
        func_data = ev_epoch.final_data["func"]
        dm = np.concatenate([i[...,np.newaxis] for i in func_data], axis=1)[:,::2]

        # get mean & std to zscore
        m_ = dm.mean(axis=0)
        s_ = dm.std(axis=0)

        dm = (dm-m_)/s_

        # add intercept
        icpt = np.ones((dm.shape[0],1))

        # concatenate design matrix into (time,3)
        dm = np.concatenate([icpt, dm], axis=1)

        # select medium annulus profile
        data = func_data[1]

        # run glm
        betas, sse, rank, s = np.linalg.lstsq(dm, data)
        
        # get ratios
        return {
            "betas": betas,
            "sse": sse,
            "rank": rank,
            "s": s,
            "dm": dm.copy(),
            "data": data.squeeze(),
            "ratio": abs(betas[order[0]])/abs(betas[order[1]]),
            "t": utils.get_unique_ids(df, id=time_col)
        }

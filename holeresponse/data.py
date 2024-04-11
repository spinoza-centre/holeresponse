from linescanning import (
    utils,
    plotting,
    dataset,
    preproc
)
import nibabel as nb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import transform
from .utils import SubjectsDict
opj = os.path.join
subj_dict = SubjectsDict()

class H5Parser(preproc.DataFilter):

    def __init__(
        self,
        h5_files,
        filter_evs=True,
        compartments=3,
        wm_comps=None,
        excl_runs=True,
        resample_kws={},
        ribbon=None,
        wm=None,
        lp=False,
        lp_kw={},
        unique_ribbon=False,
        use_raw=False,
        *args,
        **kwargs
        ):

        self.h5_files = h5_files
        self.filter_evs = filter_evs
        self.excl_runs = excl_runs
        self.compartments = compartments
        self.resample_kws = resample_kws
        self.wm_comps = wm_comps
        self.lp = lp
        self.lp_kw = lp_kw
        self.unique_ribbon = unique_ribbon
        self.ribbon = ribbon
        self.use_raw = use_raw

        self.verbose = False
        if "verbose" in list(kwargs.keys()):
            self.verbose = kwargs["verbose"]

        if isinstance(self.h5_files, str):
            self.h5_files = [self.h5_files]

        if len(self.h5_files)>0:
            self.df_func = []
            self.df_avg = []
            self.dict_ribbon = {}
            self.dict_wm = {}
            self.df_onsets = []
            self.df_comps = []
            self.df_wm = []
            self.df_filt = []
            self.df_orig = []
            self.df_raw = []
            self.h5_objs = {}
            for h5 in self.h5_files:

                # read bids properties
                comps = utils.split_bids_components(h5)
                sub_id = f"sub-{comps['sub']}"

                # parse h5-file
                ddict = self.read_single_h5(h5, *args, **kwargs)
                self.h5_objs[sub_id] = ddict["obj"]
                self.df_raw.append(ddict["obj"].df_func_raw)
                
                # excl runs (defined in utils.py)
                if self.excl_runs:
                    filt_ddict = {}
                    for tag in ["func","onsets"]:
                        filt_ddict[tag] = self.exclude_runs(ddict[tag], subject=sub_id)
                else:
                    filt_ddict = ddict

                # basic format
                if self.use_raw:
                    self.func = ddict["obj"].df_func_raw.copy()
                else:
                    self.func = filt_ddict["func"]

                # apply filtering
                if len(self.lp_kw)>0:
                    add_txt = f" with options: {self.lp_kw}"
                else:
                    add_txt = ""

                utils.verbose(f"Low-pass filtering data{add_txt}", self.verbose)
                super().__init__(
                    self.func,
                    filter_strategy="lp",
                    lp_kw=self.lp_kw
                )
                self.df_filt = self.get_result()

                # use low-passed data or not
                if self.lp:
                    self.active_func = self.df_filt.copy()
                else:
                    self.active_func = self.func.copy()

                self.df_func.append(self.active_func)
                self.df_onsets.append(filt_ddict["onsets"])
                self.df_orig.append(self.func)

                # ribbon format - nr of voxels differs per subject, so store in dictionary (Marco would be proud)
                has_dict = False
                if not isinstance(self.ribbon, (list,tuple)):
                    has_dict = subj_dict.has_ribdict(sub_id)
                    # check if we have run-specific ribon values   
                    if not has_dict:
                        ribbon = subj_dict.get_ribbon(sub_id)

                # print(has_dict)
                if not has_dict or not self.unique_ribbon:
                    df_ribbon = utils.select_from_df(
                        self.active_func, 
                        expression="ribbon", 
                        indices=ribbon
                    )
                else:
                    df_ribbon = self.fetch_unique_ribbon(
                        self.active_func,
                        subject=sub_id
                    )

                # get white matter voxels
                if not isinstance(wm, (list,tuple)):
                    wm = subj_dict.get_wm(sub_id)

                wm_tmp = utils.select_from_df(
                    self.active_func, 
                    expression="ribbon", 
                    indices=wm
                )

                self.dict_wm[sub_id] = wm_tmp.copy()

                # invert if necessary
                inv_rib = subj_dict.get_invert(sub_id)
                if inv_rib:
                    utils.verbose(f"Inverting ribbon order", self.verbose)
                    df_ribbon = df_ribbon[df_ribbon.columns[::-1]]

                self.dict_ribbon[sub_id] = df_ribbon.copy()

                # average over ribbon/wm
                df_tmp = []
                for df,tag in zip([df_ribbon, wm_tmp],["gm","wm"]):
                    df_avg = pd.DataFrame(df.mean(axis=1))
                    df_avg.rename(columns={0: tag}, inplace=True)
                    df_tmp.append(df_avg)
                
                df_tmp = pd.concat(df_tmp, axis=1)
                self.df_avg.append(df_tmp)

                # make 3-compartment model so we can average depths across subjects
                utils.verbose(f"Making {self.compartments} GM-compartments", self.verbose)
                df_comp = self.make_compartment_model(
                    df_ribbon, 
                    compartments=self.compartments,
                    **self.resample_kws
                )

                # take 25% of GM for WM
                if not isinstance(self.wm_comps, int):
                    self.wm_comps = int(round((self.compartments*0.25),0)) 
                
                utils.verbose(f"Making {self.wm_comps} WM-compartments", self.verbose)
                wm_comp = self.make_compartment_model(
                    wm_tmp, 
                    compartments=self.wm_comps,
                    **self.resample_kws
                )

                self.df_comps.append(df_comp)
                self.df_wm.append(wm_comp)

                utils.verbose(f"Done with '{sub_id}'\n", self.verbose)

            self.df_func = pd.concat(self.df_func)
            self.df_onsets = pd.concat(self.df_onsets)
            self.df_avg = pd.concat(self.df_avg)
            self.df_comps = pd.concat(self.df_comps)
            self.df_wm = pd.concat(self.df_wm)
            self.df_raw = pd.concat(self.df_raw)
            self.df_orig = pd.concat(self.df_orig)
            
            if self.filter_evs:
                self.df_onsets = utils.select_from_df(self.df_onsets, expression=("event_type != response","&","event_type != blink"))
    
    @classmethod
    def fetch_unique_ribbon(
        self, 
        func,
        rib_dict=None,
        subject=None,
        ):
                    
        # get ribbon dict | organized in dictionaries collecting task and runs as keys and values, respectively
        if not isinstance(rib_dict, dict):
            subj_dict = SubjectsDict()
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
    def exclude_runs(self, df, subject=None):

        new = []
        task_ids = utils.get_unique_ids(df, id="task")
        for task in task_ids:
            
            # get task df
            task_df = utils.select_from_df(df, expression=f"task = {task}")

            # get task-specific runs to exclude
            all_runs = utils.get_unique_ids(task_df, id="run", as_int=True)
            excl_runs = subj_dict.get_excl_runs(subject, task=task)
            keep_runs = [i for i in all_runs if i not in excl_runs]
            
            for j in keep_runs:
                new.append(utils.select_from_df(task_df, expression=f"run = {j}"))

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

def make_single_df(func, idx=["subject","run","t"]):

    task_ids = utils.get_unique_ids(func, id="task")

    new_func = []
    rr = 1
    for task in task_ids:

        expr = f"task = {task}"
        t_func = utils.select_from_df(func, expression=expr)

        run_ids = utils.get_unique_ids(t_func, id="run")
        for run in run_ids:

            expr = f"run = {run}"
            r_func = utils.select_from_df(t_func, expression=expr).reset_index().drop(["task"], axis=1)
            r_func["run"] = rr
            new_func.append(r_func)

            rr += 1

    df_func = pd.concat(new_func).set_index(idx)
    
    return df_func

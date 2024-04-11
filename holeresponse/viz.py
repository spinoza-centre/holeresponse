import numpy as np
from linescanning import (
    utils,
    plotting,
    fitting,
    prf
)
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import seaborn as sns
from scipy import stats,ndimage
import pandas as pd
import os 
from .utils import *
from holeresponse import data

opj = os.path.join
opd = os.path.dirname

class PlotStimPars(plotting.Defaults):
    def __init__(
        self,
        df=None,
        gridspec_kw={},
        axs=None,
        cm="inferno",
        save_as=None,
        o_pars=["magnitude","fwhm","time_to_peak","half_rise_time"],
        plot_kwargs={},
        **kwargs):

        super().__init__(**plot_kwargs)
        self.df = df
        self.gridspec_kw = gridspec_kw
        self.axs = axs
        self.cm = cm
        self.o_pars = o_pars
        self.save_as = save_as

        # get parameters to include
        if not isinstance(self.o_pars, (list,np.ndarray)):
            self.include_pars = [i for i in list(self.df.columns) if i not in ["sizes","depth","magnitude_ix"]]
        else:
            self.include_pars = self.o_pars

        try:
            self.stim_sizes = self._get_unique_ids(self.df, id="sizes")
        except:
            pass

        try:
            self.depth = self._get_unique_ids(self.df, id="depth")
        except:
            pass        

    @staticmethod
    def get_units(par):

        allowed_options = [
            "magnitude",
            "mag",
            "time_to_peak",
            "time to peak",
            "ttp",
            "fwhm",
            "ecc",
            "eccentricity"
        ]
        
        if par in ["magnitude","mag"]:
            unit = "(%)"
        elif par in ["fwhm","time_to_peak","ttp","half_rise_time"]:
            unit = "(s)"
        elif par in ["ecc","eccentricity"]:
            unit = "(dva)"
        else:
            raise ValueError(f"Unknown parameter '{par}' specified. Must be one of {allowed_options}")

        return unit

    def plot_metrics_across_stims(
        self, 
        axs=None,
        cm="inferno",
        include_pars=None,
        add_title=True,
        fig_kwargs={},
        save_as=None,
        **kwargs):

        if isinstance(include_pars, list):
            self.include_pars = include_pars

        # sort out axes
        if not isinstance(axs, (list,np.ndarray)):
            figsize = (len(self.include_pars)*4,4)
            fig,axs = plt.subplots(
                ncols=len(self.include_pars),
                figsize=figsize,
                constrained_layout=True,
                **fig_kwargs)
            
            add_subxlabel = True
        else:
            add_subxlabel = False

        colors = sns.color_palette(cm, len(self.depth))
        for p_ix,par in enumerate(self.include_pars):

            if add_title:
                unit = self.get_units(par)
                lbl = f"{par.replace('_',' ')} {unit}"
            else:
                lbl = None

            for ix,dep in enumerate(self.depth):
                if p_ix+1 == len(self.include_pars):
                    lbls = [f"{round(self.depth[ix],2)}%"]
                else:
                    lbls = None

                pars = utils.select_from_df(self.df, expression=f"depth = {dep}")
                pl = plotting.LazyPlot(
                    pars[par].values,
                    xx=pars["sizes"].values.astype(float),
                    x_ticks=pars["sizes"],
                    axs=axs[p_ix],
                    color=colors[ix],
                    # label=tag,
                    x_dec=2,
                    line_width=2,
                    labels=lbls,
                    markers="o",
                    x_lim=[
                        0.90*pars["sizes"].values.astype(float)[0],
                        1.1*pars["sizes"].values.astype(float)[-1]],
                    title=lbl,
                    **kwargs
                )

            if add_subxlabel:
                fig.supxlabel("stimulus size (dva)", fontsize=pl.font_size)

        if isinstance(save_as, str):
            fig.savefig(
                save_as,
                bbox_inches="tight",
                dpi=300,
                facecolor="white"
            )

    def imshow_metrics_across_stims(
        self, 
        axs=None,
        cm="viridis",
        include_pars=None,
        add_title=True,
        fig_kwargs={},
        save_as=None,
        **kwargs):

        if isinstance(include_pars, list):
            self.include_pars = include_pars

        # sort out axes
        if not isinstance(axs, (list,np.ndarray)):
            figsize = (len(self.include_pars)*3.5,4)
            fig,axs = plt.subplots(
                ncols=len(self.include_pars),
                figsize=figsize,
                constrained_layout=True,
                **fig_kwargs)
            
            add_subxlabel = True
            single_y_lbl = True
        else:
            add_subxlabel = False
            single_y_lbl = False

        for p_ix,par in enumerate(self.include_pars):

            if add_title:
                unit = self.get_units(par)
                lbl = f"{par.replace('_',' ')} {unit}"
            else:
                lbl = None

            y_lbl = None
            if single_y_lbl:
                if p_ix == 0:
                    y_lbl = "depth"

            im_data = []
            for ev in self.stim_sizes:
                df = utils.select_from_df(self.df, expression=f"sizes = {ev}")
                im_data.append(df[par].values[...,np.newaxis])

            im_data = np.concatenate(im_data, axis=1)
            axs[p_ix].imshow(im_data, cmap=cm, **kwargs)
            _,pl = plotting.conform_ax_to_obj(
                axs[p_ix], 
                y_label=y_lbl,
                y_ticks=list(np.arange(0,df.shape[0])),
                x_ticks=list(np.arange(0,len(self.stim_sizes))),
                title=lbl)


            if add_subxlabel:
                fig.supxlabel("stimulus number", fontsize=pl.font_size)
                

        if isinstance(save_as, str):
            fig.savefig(
                save_as,
                bbox_inches="tight",
                dpi=300,
                facecolor="white"
            )
            
    def _get_unique_ids(self, df, id="event_type"):
        try:
            df = df.reset_index()
        except:
            pass

        return list(np.unique(df[id].values))            

class PlotEpochProfiles():

    def __init__(
        self,
        df,
        figsize=(5,5),
        cm="inferno",
        title=None,
        axs=None,
        ev_names=None,
        bsl=20,
        time_col="t",
        correct=True,
        force_title=False,
        *args,
        **kwargs
        ):

        self.df = df
        self.figsize = figsize
        self.cm = cm
        self.axs = axs
        self.title = title
        self.ev_names = ev_names
        self.bsl = bsl
        self.correct = correct

        # read EVs
        self.evs = utils.get_unique_ids(self.df, id="event_type")

        # initialize axes
        can_I_add_title = False
        if not isinstance(axs, mpl.axes._axes.Axes):
            self.fig,self.axs = plt.subplots(figsize=self.figsize)
            can_I_add_title = True

        # loop over evs and average over epochs
        self.m_data = []
        self.e_data = []
        for ix,ev in enumerate(self.evs):
            
            ddf = utils.select_from_df(self.df, expression=f"event_type = {ev}")
            avg_,sem_ = self.get_avg_and_sem(
                ddf, 
                bsl=self.bsl,
                correct=self.correct
            )
            
            self.m_data.append(avg_)
            self.e_data.append(sem_)

            t_ = utils.get_unique_ids(ddf, id=time_col)

        if not isinstance(self.ev_names, list):
            self.ev_names = self.evs

        # plot data in list format
        defs = {
            "x_label": "time (s)",
            "y_label": "magnitude (%)",
            "line_width": 2,
        }

        for key,val in defs.items():
            kwargs = utils.update_kwargs(
                kwargs,
                key,
                val
            )
        
        kwargs = utils.update_kwargs(
            kwargs,
            "labels",
            self.ev_names
        )

        self.pl = plotting.LazyPlot(
            self.m_data,
            xx=t_,
            error=self.e_data,
            axs=self.axs,
            cmap=self.cm,
            add_hline=0,
            **kwargs
        )

        add_axvspan(self.axs, ymax=0.1, alpha=0.5)
        if isinstance(title, (str,dict)):
            if can_I_add_title:
                self.fig.suptitle(
                    self.title, 
                    fontsize=self.pl.title_size, 
                    fontweight="bold"
                )

            if force_title:
                if isinstance(self.title, str):
                    self.title = {
                        "title": self.title,
                        "fontsize": self.pl.title_size,
                        "fontweight": "bold"
                    }

                self.pl._set_title(
                    self.axs,
                    self.title,
                )

    @staticmethod
    def get_avg_and_sem(df, correct=True, **kwargs):

        # get list of epoch IDs
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be dataframe with 'epoch' in index. Not {df} of type '{type(df)}'")
        
        try:
            eps = utils.get_unique_ids(df, id="epoch")
            searcher = "epoch"
        except:
            eps = utils.get_unique_ids(df, id="subject")
            searcher = "subject"

        # shift baseline 
        avg_ = []
        for i in eps:
            d_ = utils.select_from_df(df, expression=f"{searcher} = {i}").values
            if correct:
                d_shift = fitting.Epoch.correct_baseline(d_, **kwargs)
                avg_.append(d_shift)
            else:
                avg_.append(d_)

        shifted = np.concatenate(avg_, axis=1)
        avg_ = shifted.mean(axis=1)
        sem_ = stats.sem(shifted, axis=1)
        
        return avg_,sem_
    
class XinYuPlot(plotting.Defaults):

    def __init__(
        self,
        df,
        axs=None,
        cm="seismic",
        figsize=(16,4),
        plot_kwargs={},
        title=None,
        ev_names=None,
        ev_colors=None,
        relative_to=None,
        sharey=True,
        y_lbl="% away from pial",
        annot_ribbon=False,
        *args,
        **kwargs
        ):

        super().__init__(**plot_kwargs)

        self.df = df
        self.cm = cm
        self.figsize = figsize
        self.axs = axs
        self.ev_names = ev_names
        self.ev_colors = ev_colors
        
        evs = utils.get_unique_ids(df, id="event_type", sort=False)
        ncols = len(evs)
        wratios = [1/len(evs) for _ in evs]
        single_cb = False
        if sharey:
            ncols += 1
            wratios = [1,1,1,0.05]
            single_cb = True
            vmin,vmax = np.amin(self.df.values),np.amax(self.df.values)

            for kw in ["sns_kws", "cb_kws"]:
                if kw in list(kwargs.keys()):
                    sn_kw = kwargs[kw]
                else:
                    sn_kw = kwargs[kw] = {}

                for tag,val in zip(["vmin","vmax"],[vmin,vmax]):
                    
                    # get min max from sns_kws if specified
                    if kw == "cb_kws":
                        if tag in list(kwargs["sns_kws"]):
                            sn_kw[tag] = kwargs["sns_kws"][tag]
                            
                    if not tag in list(sn_kw.keys()):
                        sn_kw = utils.update_kwargs(
                            sn_kw,
                            tag,
                            val,
                            force=True
                        )

                    kwargs[kw] = sn_kw


        can_I_add_title = False
        if not isinstance(self.axs, list):
            can_I_add_title = True
            if isinstance(self.axs, mpl.figure.SubFigure):
                self.fig = self.axs
                axs = self.axs.subplots(
                    ncols=ncols,
                    gridspec_kw={
                        "width_ratios": wratios,
                        # "wspace": 0
                    }
                )
            else:
                self.fig,axs = plt.subplots(
                    ncols=ncols, 
                    figsize=self.figsize,
                    gridspec_kw={
                        "width_ratios": wratios,
                        "wspace": 0.15
                    }
                )

        # pass on axis for colorbar
        if sharey:
            kwargs["cb_kws"]["axs"] = axs[-1]

        self.avg_imshow = {}
        self.objs = {}
        for ix,ev in enumerate(evs):

            annot = False
            ddf = utils.select_from_df(self.df, expression=f"event_type = {ev}")
            if ix == 0:
                y_ = y_lbl

                if annot_ribbon:
                    annot = True
            else:
                y_ = None
            
            # allow custom ev names
            if isinstance(self.ev_names, list):
                if len(self.ev_names) != len(evs):
                    raise ValueError(f"Length of custom EVs ({len(self.ev_names)}) does not match original EVs ({len(evs)})")
                
                ev = self.ev_names[ix]

            if not isinstance(self.ev_colors, list):
                self.ev_colors = ["k" for _ in self.ev_names]

            # draw contours based on existing object
            if isinstance(relative_to, str):
                if ev != relative_to:
                    if relative_to in list(self.objs.keys()):
                        kwargs = utils.update_kwargs(
                            kwargs,
                            "contours_relative",
                            self.objs[relative_to]
                        )
                    else:
                        utils.verbose(f"Cannot use '{relative_to}' as reference for contours; not yet available..", True)

            # if y-axis is shared we should have only 1 cbar at the last column
            add_cb = True
            if single_cb:
                if (ix+1)<len(evs):
                    add_cb = False 
            
            kwargs = utils.update_kwargs(
                kwargs,
                "cbar",
                add_cb,
                force=True
            )

            kw = "sns_kws"
            if kw in list(kwargs.keys()):
                sn_kw = kwargs[kw]
            else:
                sn_kw = kwargs[kw] = {}

            sn_kw = utils.update_kwargs(
                sn_kw,
                "cbar",
                add_cb,
                force=True
            )

            kwargs[kw] = sn_kw

            # only add ticks at the first column
            add_yticks = []
            if ix == 0:
                add_yticks = None 
            
            kwargs = utils.update_kwargs(
                kwargs,
                "y_ticks",
                add_yticks,
                force=True
            )        

            obj = self.single_xinyu_plot(
                ddf,
                ev=ev,
                ev_col=self.ev_colors[ix],
                axs=axs[ix],
                cm=self.cm,
                y_lbl=y_,
                annot=annot,
                *args,
                **kwargs
            )

            self.objs[ev] = obj
            self.avg_imshow[ev] = obj["data"].copy()

        if isinstance(title, (str,dict)):
            if isinstance(title, str):
                title = {
                    "t": title,
                    "fontsize": self.title_size,
                    "fontweight": "bold",
                    "y": 1.1
                }

            if can_I_add_title:
                self.fig.suptitle(**title)
                # plt.tight_layout()

    @staticmethod
    def single_xinyu_plot(
        data,
        ev=None,
        ev_col="k",
        axs=None,
        time_col="t",
        time_dec=2,
        time_ticks=None,
        show_time=None,
        as_depth=True,
        depth_ticks=None,
        cm="seismic",
        transpose=True,
        y_lbl="% away from pial",
        x_lbl="time (s)",
        annot=True,
        annot_color="#ffffff",
        depth_dec=0,
        sns_kws={},
        cb_kws={},
        cont_kws={},
        contours=True,
        contours_relative=None,
        cont_lines=[0.25,0.5,0.75],
        bsl=None,
        annot_onset=False,
        **kwargs
        ):

        if not isinstance(axs, mpl.axes._axes.Axes):
            _,axs = plt.subplots(figsize=(5,5))

        # correct baseline
        if isinstance(bsl, int):
            data = fitting.Epoch.correct_baseline(
                data,
                bsl=bsl
            )
        
        # select specific time window
        orig_data = data.copy()
        if isinstance(show_time, list):
            data = utils.multiselect_from_df(
                data,
                expression=[
                    f"{time_col} gt {show_time[0]}",
                    f"{time_col} lt {show_time[-1]}"
                ]
            )

        # define columns as percentages
        if as_depth:
            data.columns = np.linspace(0,100,data.shape[-1], dtype=int)
            
        # massage df; reset index and round time-axis
        data.reset_index(inplace=True)
        for elem in ["event_type", "covariate"]:
            if elem in list(data.columns):
                data.drop([elem], axis=1, inplace=True)
                
        data.set_index([time_col], inplace=True)

        # transpose to ensure time is on x-axis
        if transpose:
            data = data.T
        
        use_data = data.values
        pop_levels = False
        lvls = None
        cntr = None
        if contours:
        
            # add the levels based on percentages
            if not "levels" in list(cont_kws.keys()):
                
                # get min-max range of data
                mmin,mmax = use_data.min().min(), use_data.max().max()

                # force into list
                if isinstance(cont_lines, (float, int)):
                    cont_lines = [cont_lines]

                # multiple data range with levels
                lvls = tuple([i*(mmax+abs(mmin)) for i in cont_lines])
                
                # update kwargs
                cont_kws = utils.update_kwargs(
                    cont_kws, 
                    "levels", 
                    lvls
                )
                
                # later remove from kwargs
                pop_levels = True

            # draw contours based on existing single_xinyu_plot-object
            if isinstance(contours_relative, dict):
                lvls = fetch_level_values(
                    contours_relative,
                    data
                )

                # update kwargs
                cont_kws = utils.update_kwargs(
                    cont_kws, 
                    "levels", 
                    lvls,
                    force=True
                )
                
                # later remove from kwargs
                pop_levels = True

            # set default colors
            cont_kws = utils.update_kwargs(
                cont_kws, 
                "colors", 
                "black"
            )

            # remove colors from kwargs if cmap is specified
            if "cmap" in list(cont_kws.keys()):
                cont_kws.pop("colors")

            # add the contours
            cntr = axs.contour(
                np.linspace(0,use_data.shape[1], num=use_data.shape[1]),
                np.linspace(0,use_data.shape[0], num=use_data.shape[0]),
                use_data, 
                **cont_kws
            )
            
        else:
            use_data = data.copy()


        # decide what to do with colorbar (stems from sharey-arg in XinYuPlot-class)
        fix_cbar = True
        if "cbar" in list(sns_kws):
            if not sns_kws["cbar"]:
                fix_cbar = False

            sns_kws = utils.update_kwargs(
                sns_kws,
                "cbar",
                False,
                force=True
            )

        # make heat map
        sns.heatmap(
            use_data,
            cmap=cm,
            ax=axs,
            **sns_kws
        )

        # set time ticks
        if isinstance(time_ticks, int):
            new_ticks = np.linspace(0.5,data.shape[-1], num=time_ticks)
            t_min,t_max = list(data.columns)[0],list(data.columns)[-1]
            new_lbls = np.linspace(t_min,t_max, num=time_ticks).round(time_dec)

            if time_dec == 0:
                new_lbls = [int(i) for i in new_lbls]
                
            axs.set_xticks(new_ticks, rotation=0, labels=new_lbls)

        # set y-ticks
        axs.set_yticks(axs.get_yticks(), rotation=0, labels=axs.get_yticklabels())
        if isinstance(depth_ticks, int):

            new_ticks = np.linspace(0.5,data.shape[0], num=depth_ticks)
            if as_depth:
                max_val = 100
            else:
                max_val = data.shape[0]

            new_lbls = np.linspace(0,max_val, num=depth_ticks)
            
            if isinstance(depth_dec, int):
                if depth_dec>0:
                    new_lbls = new_lbls.round(depth_dec)
                else:
                    new_lbls = new_lbls.astype(int)
            
            axs.set_yticks(new_ticks, rotation=0, labels=new_lbls)            

        if annot_onset:
            # find at which index t=0
            t_onset = utils.find_nearest(utils.get_unique_ids(orig_data, id="t"),0)[0]

            kwargs = utils.update_kwargs(
                kwargs,
                "add_vline",
                {
                    "pos": int(t_onset)
                },
            )
            
        pl,obj = plotting.conform_ax_to_obj(
            ax=axs,
            title={
                "title": ev,
                "fontweight": "bold",
                "color": ev_col
            },
            x_label=x_lbl,
            y_label=y_lbl,
            **kwargs
        )

        if fix_cbar:
            for tag,ffunc in zip(["vmin","vmax"],[np.amin,np.amax]):
                if tag in list(sns_kws.keys()):
                    if not tag in list(cb_kws.keys()):
                        cb_kws = utils.update_kwargs(
                            cb_kws,
                            tag,
                            sns_kws[tag]
                        )
                else:
                    if not tag in list(cb_kws.keys()):
                        cb_kws = utils.update_kwargs(
                            cb_kws,
                            tag,
                            ffunc(use_data),
                            force=True
                        )

            if not "ax" in list(cb_kws.keys()):
                if axs.collections[-1].colorbar != None:
                    if isinstance(axs.collections[-1].colorbar.ax, mpl.axes._axes.Axes):
                        cb_ax = axs.collections[-1].colorbar.ax
                        cb_kws = utils.update_kwargs(
                            cb_kws,
                            "axs",
                            cb_ax,
                            force=True
                        )

            plotting.LazyColorbar(
                cmap=cm,
                # ori="horizontal",
                **cb_kws
            )

        if annot:
            annotate_cortical_ribbon(
                axs,
                fontsize=obj.font_size,
                fontweight="bold",
                xycoords="axes fraction",
                color=annot_color
            )   

        # pop levels so contour values are updated
        if pop_levels:
            cont_kws.pop("levels")

        return {
            "axs": axs,
            "obj": obj,
            "data": use_data,
            "orig": orig_data,
            "levels": lvls,
            "contours": cntr
        }

class PlotDeconvProfiles():

    def __init__(
        self,
        obj,
        axs=None,
        title=None,
        inset_par="time_to_peak",
        inset_kwargs={},
        time_par="time",
        err=None,
        ttp_lines=True,
        ev_names=None,
        ev_colors=None,
        force_int=False,
        bold_title=False,
        bsl=None,
        onset_shade=True,
        cm="inferno",
        *args,
        **kwargs):

        self.obj = obj
        self.axs = axs
        self.title = title
        self.inset_par = inset_par
        self.err = err
        self.ev_names = ev_names
        self.force_int = force_int
        self.ev_colors = ev_colors
        self.bold_title = bold_title
        self.time_par = time_par
        self.bsl = bsl

        if not isinstance(self.obj, pd.DataFrame):
            # get profile timecourses
            if not hasattr(self.obj, "tc_condition"):
                self.obj.timecourses_condition()

            # get parameters
            if not hasattr(self.obj, "pars_subjects"):
                self.obj.parameters_for_tc_subjects(nan_policy=True)            

            # average parameters
            self.pars = self.obj.pars_subjects.groupby(["subject","event_type","vox"]).mean()
            self.profs = self.obj.tc_condition.copy()
            self.ev_ids = self.obj.cond.copy()
            self.df_sem = self.obj.sem_condition.copy()
            self.df_std = self.obj.std_condition.copy()
            self.add_lbls = False
        else:
            # deal with non-NideconvFitter object
            self.pars = None
            self.grouper = self.obj.groupby(["event_type", self.time_par])
            self.profs = self.grouper.mean()
            self.df_sem = self.grouper.sem()
            self.df_std = self.grouper.std()
            self.ev_ids = utils.get_unique_ids(self.profs, id="event_type")
            self.add_lbls = True

        # get depth 
        self.depth = np.linspace(0,100,num=self.profs.shape[-1])

        can_I_add_title = False
        if not isinstance(self.axs, (mpl.axes._axes.Axes,list,np.ndarray)):
            can_I_add_title = True

            figsize = (len(self.ev_ids)*4,4)
            self.fig,self.axs = plt.subplots(
                ncols=len(self.ev_ids), 
                figsize=figsize, 
                gridspec_kw={"wspace": 0.1}, 
                sharey=True, 
                sharex=True,
                constrained_layout=True
            )

            if len(self.ev_ids) == 1:
                self.axs = [self.axs]
        else:
            if isinstance(self.axs, (np.ndarray)):
                self.axs = list(self.axs)
            elif isinstance(self.axs, mpl.axes._axes.Axes):
                self.axs = [self.axs]

        for ix,ev in enumerate(self.ev_ids):
            
            if not isinstance(self.ev_names, list):
                self.ev_names = self.ev_ids

            lbl = None
            if self.add_lbls:
                if ix == 0:
                    try:
                        lbl = [str(round(float(i),2)) for i in self.depth]
                    except:
                        lbl = self.depth

                    if self.force_int:
                        lbl = [int(float(i)) for i in lbl]
    
            try:
                ev_title = f"{round(float(self.ev_names[ix]),2)} (dva)"
            except:
                ev_title = self.ev_names[ix]

            if isinstance(self.ev_colors, list):
                ev_title = {
                    "title": ev_title,
                    "color": self.ev_colors[ix]
                }

            if self.bold_title:
                if not isinstance(ev_title, dict):
                    ev_title = {
                        "title": ev_title,
                        "fontweight": "bold"
                    }
                else:
                    ev_title["fontweight"] = "bold"

            # actual profiles
            df = utils.select_from_df(self.profs, expression=f"event_type = {ev}")
            ev_prof = list(df.values.T)
            
            # check if we should add error shading
            ev_err = None
            if isinstance(self.err, str):
                if self.err.lower() == "sem":
                    df_err = self.df_sem.copy()
                elif self.err.lower() == "std":
                    df_err = self.df_std.copy()
                else:
                    raise ValueError(f"err must be one of 'std' (standard deviation) or 'sem' (standard error of mean), not '{self.err}'")

                # select event-specific error dataframe
                err = utils.select_from_df(df_err, expression=f"event_type = {ev}")

                # parse values into list
                ev_err = list(err.values.T)

            if isinstance(self.bsl, int):
                ev_prof = [correct_baseline(i, bsl=self.bsl) for i in ev_prof]

            # plot
            kwargs = utils.update_kwargs(kwargs, "line_width", 2)
            kwargs = utils.update_kwargs(
                kwargs,
                "label",
                lbl
            )

            pl = plotting.LazyPlot(
                ev_prof,
                xx=df.index.get_level_values(self.time_par).values,
                axs=self.axs[ix],
                title=ev_title,
                add_hline=0,
                cmap=cm,
                error=ev_err,
                *args,
                **kwargs
            )

            # add cute onset block
            if onset_shade:
                add_axvspan(self.axs[ix])

            if isinstance(self.pars, pd.DataFrame) & isinstance(inset_par, str):
                if ix == 0:
                    if self.depth.shape[0]<5:
                        try:
                            lbl = [str(round(float(i),2)) for i in self.depth]
                        except:
                            lbl = self.depth

                        if self.force_int:
                            lbl = [int(float(i)) for i in lbl]

                        inset_kwargs = utils.update_kwargs(
                            inset_kwargs,
                            "labels",
                            lbl
                        )

                    x_lbl = self.inset_par.replace("_"," ")
                    y_lbl = "%away from pial"
                    if "sns_ori" in list(inset_kwargs.keys()):
                        if inset_kwargs["sns_ori"] == "v":
                            y_lbl = self.inset_par.replace("_"," ")
                            x_lbl = "%away from pial"
                
                    add_lbl = True
                else:
                    y_lbl = x_lbl = None
                    add_lbl = False

                inset_kwargs = utils.update_kwargs(
                    inset_kwargs,
                    "add_labels",
                    y_lbl
                )

                left,bottom,width,height = 0.65,0.65,0.4,0.4
                ax2 = self.axs[ix].inset_axes([left, bottom, width, height])

                self.ev_pars = utils.select_from_df(self.pars, expression=f"event_type = {ev}")                
                self.bar_plot = plotting.LazyBar(
                    data=self.ev_pars,
                    x="vox",
                    y=self.inset_par,
                    axs=ax2,
                    font_size=pl.font_size/1.5,
                    label_size=pl.label_size/1.5,
                    x_label=x_lbl,
                    y_label=y_lbl,
                    error=self.err,
                    **inset_kwargs
                )

        if can_I_add_title:
            self.fig.supxlabel("time (s)", fontsize=pl.font_size)
            self.fig.supylabel("magnitude (%)", fontsize=pl.font_size)        
            
            if isinstance(self.title, str):
                self.fig.suptitle(
                    self.title, 
                    fontsize=pl.title_size, 
                    fontweight="bold", 
                    y=1.1
                )

def _save_figure(
    fig, 
    fname=None,
    fig_dir=None,
    subject=None,
    exts=["pdf","png"],
    overwrite=False,
    return_figdir=False,
    **kwargs):
    
    # define default path for figures in repository
    if not isinstance(fig_dir, str):
        fig_dir = opj(opd(opd(os.path.realpath(__file__))), "images")

    if return_figdir:
        return fig_dir
    
    # add subject if desired
    if isinstance(subject, str):
        fig_dir = opj(fig_dir, subject)

    # make directory if it doesn't exist
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)    
        
    # force extensions in list
    if isinstance(exts, str):
        exts = [exts]

    for ext in exts:

        # full file
        fullfile = opj(fig_dir, f"{fname}.{ext}")
        
        # decide execution rules; save if file doesn't exist or if overwrite == True
        save_as = False
        if not os.path.exists(fullfile):
            save_as = True
        else:
            if overwrite:
                save_as = True

        if save_as:
            print(f"Writing '{fullfile}'")
            fig.savefig(
                fullfile,
                bbox_inches="tight",
                dpi=300,
                facecolor="white",
                **kwargs
            )    

class StimPNGs(SubjectsDict):

    def __init__(self, subject, **kwargs):

        super().__init__(**kwargs)
        self.subject = subject
        self.png_files = self.fetch_files()
        self.cms = [utils.make_binary_cm(i) for i in self.get_colors()]

    def fetch_files(self):
        scr_dir = self.get_scr_dir(self.subject)
        return utils.get_file_from_substring(["png"], scr_dir)
    
    def load_png(self, file):
        img = (255*mpimg.imread(file)).astype('int')
        img_bin = np.zeros_like(img[...,0])
        img_bin[np.where(((img[..., 0] < 40) & (img[..., 1] < 40)) | ((img[..., 0] > 200) & (img[..., 1] > 200)))] = 1

        return {
            "bin": img_bin.copy(),
            "rgb": img.copy()
        }
    
    def plot_srf(
        self, 
        axs=None, 
        figsize=(5,5), 
        annot=True,
        circ_kw={},
        ticks=False,
        labels=False,
        normalize=False,
        **kwargs
        ):

        # get SRF object
        srf_obj = self.get_srf(self.subject, normalize=normalize)

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)

        # extract 
        max_val,max_dva = srf_obj["max_response"],srf_obj["max_size"]

        # plot
        if not ticks:
            kwargs["x_ticks"] = kwargs["y_ticks"] = []

        if labels:
            kwargs["x_label"] = "stimulus size"
            kwargs["y_label"] = "response"

        for key,val in zip(["color","line_width","x_lim"],["k",3,[0,5]]):
            if not key in list(kwargs.keys()):
                kwargs[key] = val

        plotting.LazyPlot(
            srf_obj["srf"],
            xx=srf_obj["stim_sizes"],
            y_lim=[0,max_val*1.1],
            axs=axs,
            **kwargs
        )

        if annot:
            x_limits,y_limits = axs.get_xlim(), axs.get_ylim()
            plotting.conform_ax_to_obj(
                ax=axs,
                add_vline={"pos": max_dva, "max": (1/y_limits[-1])*max_val},
                add_hline={"pos": max_val, "max": (1/x_limits[-1])*max_dva},
                **kwargs
            )

            # set some defaults
            if not "color" in list(circ_kw.keys()):
                circ_kw["color"] = self.get_colors()[0]

            if not "size" in list(circ_kw.keys()):
                circ_size = 40
            else:
                circ_size = circ_kw["size"]
                circ_kw.pop("size")

            axs.set_aspect(x_limits[-1]/y_limits[-1])
            axs.scatter(
                max_dva,
                max_val, 
                s=circ_size, 
                zorder=2, 
                **circ_kw
            )

        return axs

    def generate_composite(
        self, 
        axs=None, 
        figsize=(4*(1920/1080),4), 
        add_srf=False,
        srf_inset=[0.0,0.7,0.3,0.4],
        srf_kw={},
        **kwargs
        ):

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)

        for ix,png in enumerate(self.png_files[::-1]):
            cm = self.cms[::-1][ix]
            self.plot_image(
                png,
                axs=axs,
                cm=cm,
                **kwargs
            )

        # # add SRF inset
        if add_srf:
            ax2 = axs.inset_axes(srf_inset)
            self.plot_srf(
                axs=ax2,
                **srf_kw
            )

    def generate_screen_images(
        self, 
        axs=None, 
        figsize=(12,3),
        colors=None,
        **kwargs
        ):

        if not isinstance(axs, (np.ndarray,list,mpl.figure.SubFigure)):
            fig,use_axes = plt.subplots(
                ncols=len(self.png_files), 
                figsize=figsize,
                constrained_layout=True
            )
        else:
            if isinstance(axs, mpl.figure.SubFigure):
                use_axes = axs.subplots(ncols=len(self.png_files))
            else:
                use_axes = axs

        pop_title = False
        colors = self.get_colors()
        evs = self.get_evs()
        for ix,png in enumerate(self.png_files):

            # set title
            if not "title" in list(kwargs.keys()):
                kwargs = utils.update_kwargs(
                    kwargs,
                    "title",
                    {
                        "title": evs[ix],
                        "fontweight": "bold",
                        "color": colors[ix]
                    }
                )

                pop_title = True

            self.plot_image(
                png,
                axs=use_axes[ix],
                img_type="rgb",
                **kwargs
            )

            if pop_title:
                kwargs.pop("title")

    def plot_image(
        self, 
        img,
        figsize=(6,4), 
        cm=None, 
        rm_axs=True,
        annotate=True,
        axs=None, 
        img_type="bin",
        **kwargs
        ):

        if isinstance(img, str):
            img = self.load_png(img)[img_type]

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)
        
        # get visual field 
        SR_ = prf.SizeResponse()
        SR_.vf_extent

        # imshow
        axs.imshow(
            img, 
            cmap=cm, 
            extent=SR_.vf_extent[0]+SR_.vf_extent[1]
        )

        # rm axes
        if rm_axs:
            axs.axis("off")

        # annotate
        if annotate:
            for ii,val in zip(
                ["-10°","10°","-5°","5°"], 
                [(0,0.51),(0.96,0.51),(0.51,0),(0.51,0.96)]):

                axs.annotate(
                    ii,
                    val,
                    fontsize=14,
                    xycoords="axes fraction"
                )

        # format
        plotting.conform_ax_to_obj(
            ax=axs,
            add_hline=0,
            add_vline=0,
            **kwargs
        )

class BijanzadehFigures(SubjectsDict):

    def __init__(
        self, 
        axs=None,
        orig_fmt=True,
        ev_names=None,
        add_title=True,
        bold_title=True,
        color_title=True,
        **kwargs
        ):

        self.axs = axs
        self.orig_fmt = orig_fmt
        self.ev_names = ev_names
        self.add_title = add_title
        self.bold_title = bold_title
        self.color_title = color_title

        # init subjects toget list of imgs | imgs are in self.bijanzadeh_figures 
        super().__init__()
        
        n_plots = len(self.bijanzadeh_figures)
        if not isinstance(self.axs, list): 
            fig,self.axs = plt.subplots(
                ncols=n_plots, 
                figsize=(n_plots*5,5), 
                constrained_layout=True,
                sharey=True
            )
        else:
            if isinstance(self.axs, np.ndarray):
                self.axs = list(self.axs)

            if len(self.axs) != n_plots:
                raise ValueError(f"Number of axes ({len(self.axs)}) does not equal number of images in {self.bijanzadeh_figures} ({n_plots})")

        # plot
        self.fig_axs = {}
        for ix,(key,val) in enumerate(self.bijanzadeh_figures.items()):
            if not os.path.exists(val):
                raise FileNotFoundError(f"Could not find file '{val}'. This should not happen..?")
            
            if self.add_title:
                if isinstance(self.ev_names, list):
                    key = self.ev_names[ix]
                
                set_title = {
                    "title": key
                }

                if self.bold_title:
                    set_title["fontweight"] = "bold"

                if self.color_title:
                    set_title["color"] = self.ev_colors[ix]
            else:
                set_title = False

            if ix == 0:
                annot = True
            else:
                annot = False

            self.fig_axs[key] = self.plot_single_bijanzadeh(
                val,
                axs=self.axs[ix],
                title=set_title,
                annot=annot,
                **kwargs
            )
    
    @classmethod
    def plot_single_bijanzadeh(
        self, 
        img, 
        axs=None,
        orig_fmt=True,
        figsize=(5,5), 
        x_pos=[0,45,90,136,181,227,270],
        y_pos=[24,88,109,154,195],
        lw=1,
        ls="dashed",
        colors="k",
        onset_frame=[90,136],
        onset_color=["#2740ffff","#ff4bffff"],
        onset_lw=2,
        onset_ls="solid",
        annot=True,
        **kwargs
        ):

        if not isinstance(axs, mpl.axes._axes.Axes):
            fig,axs = plt.subplots(figsize=figsize)

        # function in holeresponse.utils to read pdf images
        img_data = read_pdf_image(img)

        # simple imshow
        axs.imshow(img_data, aspect="auto")

        # original formatting
        if orig_fmt:
            pl = plotting.conform_ax_to_obj(
                ax=axs,
                add_vline={
                    "pos": onset_frame,
                    "color": onset_color,
                    "lw": onset_lw,
                    "ls": onset_ls
                },
                add_hline={
                    "pos": y_pos,
                    "color": colors,
                    "lw": lw,
                    "ls": ls
                },
                **kwargs
            )
            
            # set y_ticks
            axs.set_yticklabels(["{:6.1f}".format(i) for i in np.arange(-0.2,1.4, step=0.2)])

            # set x_ticks
            axs.set_xticks(x_pos)
            axs.set_xticklabels(np.arange(-100,200*1.1, step=50, dtype=int))

        if annot:
            annotate_cortical_ribbon(
                axs,
                fontsize=pl[1].font_size,
                fontweight="bold",
                xycoords="axes fraction",
                color="k"
            )   
        else:
            plotting.conform_ax_to_obj(ax=axs, **kwargs)

        return axs
    
def annotate_cortical_ribbon(
    axs,
    pial_pos=(0.02,0.92), 
    wm_pos=(0.02,0.02), 
    lbls=["pial","wm"], 
    **kwargs
    ):

    if not "xycoords" in list(kwargs.keys()):
        kwargs["xycoords"] = "axes fraction"

    for pos,tag in zip([pial_pos,wm_pos],lbls):
        axs.annotate(
            tag,
            pos,
            **kwargs
        )

def make_wm_pial_ticks(data):
    x_ticks = [0,data.shape[0]//4, data.shape[0]//2,(data.shape[0]//2+data.shape[0]//4),data.shape[0]]
    x_labels = list(np.arange(0,120, step=25))

    return {
        "ticks": x_ticks,
        "labels": x_labels
    }

def add_axvspan(
    axs, 
    loc=(0,2), 
    color="#cccccc",
    alpha=0.3, 
    ymin=0,
    ymax=1,
    **kwargs
    ):

    axs.axvspan(
        *loc, 
        ymin=ymin,
        ymax=ymax, 
        alpha=alpha, 
        color=color,
        **kwargs
    )

class EpochMethod(SubjectsDict):

    def __init__(
        self,
        h5_file,
        subject=None,
        axs=None,
        wratios=[0.8,0.2],
        figsize=(14,4),
        data_kws={},
        plot_kws={},
        **kwargs
        ):

        self.h5_file = h5_file
        self.subject = subject
        self.figsize = figsize
        self.wratios = wratios
        self.axs = axs

        # initialize SubjectsDict
        super().__init__()

        if isinstance(self.h5_file, data.H5Parser):
            self.h5_obj = self.h5_file

            if not isinstance(self.subject, str):
                raise TypeError(f"Need a subject ID if specifying an {type(self.h5_file)}-object")
            
        else:
            if not isinstance(self.h5_file, str):
                raise TypeError("Please specify an h5-file from 'lsprep' folder to use as reference")
            else:
                if not os.path.exists(self.h5_file):
                    raise FileNotFoundError(f"Could not find specified file '{self.h5_file}'")
                
            self.bids_comps = utils.split_bids_components(self.h5_file)
            self.subject = f"sub-{self.bids_comps['sub']}"

            # load data object
            self.h5_obj = self.prepare_data(
                self.h5_file,
                **data_kws
            )

        # average tasks
        self.avg = self.average_tasks(
            self.h5_obj,
            self.subject,
            make_figure=False
        )

        # make plot
        self.plot(
            plot_kws=plot_kws,
            **kwargs
        )

    @classmethod
    def prepare_data(
        self, 
        h5_file, 
        **kwargs
        ):

        h5_tmp = data.H5Parser(
            h5_file,
            **kwargs
        )

        return h5_tmp
    
    @classmethod
    def average_tasks(
        self,
        obj,
        subject, 
        **kwargs
        ):

        avg = obj.plot_task_avg(
            orig=data.H5Parser.fetch_unique_ribbon(
                obj.func, 
                subject=subject
            ),
            filt=obj.dict_ribbon[subject],
            **kwargs
        )

        return avg
    
    def plot(
        self, 
        task=None,
        plot_kws={},
        **kwargs
        ):

        ncols = 2
        if not isinstance(self.axs, list):
            if isinstance(self.axs, mpl.figure.SubFigure):
                self.fig = self.axs
                axs = self.axs.subplots(
                    ncols=ncols,
                    gridspec_kw={
                        "width_ratios": self.wratios,
                        # "wspace": 0
                    }
                )
            else:
                self.fig,axs = plt.subplots(
                    ncols=ncols, 
                    figsize=self.figsize,
                    width_ratios=self.wratios,
                    constrained_layout=True
                )
        else:
            axs = self.axs

        # set default task ID
        if not isinstance(task, str):
            task = utils.get_unique_ids(self.avg, id="task")[0]

        # get the task-specific average
        sample_df = utils.select_from_df(
            self.avg, 
            expression=f"task = {task}"
        )

        # get time axis and filtered/unfiltered data
        for key,val in zip(
            ["x_label","y_label"],
            ["time (s)","magnitude (%)"]
            ):

            plot_kws = utils.update_kwargs(
                plot_kws,
                key,
                val
            )
            
        t = utils.get_unique_ids(sample_df, id="t")

        # average onsets
        use_onsets = utils.select_from_df(
            data.average_tasks(self.h5_obj.df_onsets),
            expression=f"task = {task}"
        )

        # epoch sample
        sample_ep = fitting.Epoch(
            sample_df,
            use_onsets,
            **kwargs
        )

        # plot profiles first so that the labels are placed appropriately
        ep_sample = sample_ep.df_epoch.copy()
        obj_gm = PlotEpochProfiles(
            ep_sample.groupby(["event_type","epoch","t"]).mean()["filtered"],
            axs=axs[1],
            ev_names=self.get_evs(),
            cm=self.get_colors(),
            title={
                "title": "profiles",
                "fontweight": "normal"
            },
            force_title=True,
            line_width=3,
            bsl=20,
            x_ticks=np.arange(
                sample_ep.interval[0],
                sample_ep.interval[-1]*1.1, 
                step=2
            ),
            **plot_kws
        )

        if "labels" in list(plot_kws.keys()):
            plot_kws.pop("labels")

        plotting.LazyPlot(
            [sample_df[i].values for i in list(sample_df.columns)],
            axs=axs[0],
            xx=t,
            add_hline=0,
            line_width=[0.5,2],
            color=["#cccccc", "k"],
            title="timecourse",
            **plot_kws
        )

        # find epochs and plot/color accordingly
        ev_order = use_onsets.reset_index()["event_type"].values
        for ix,(onset,ev) in enumerate(zip(use_onsets.values.squeeze(), ev_order)):
            if ev == "act":
                ev_color = self.get_colors()[0]
            elif ev == "suppr_1":
                ev_color = self.get_colors()[1]
            else:
                ev_color = self.get_colors()[2]

            t_df = utils.select_from_df(
                sample_df, 
                expression=(
                    f"t gt {onset-sample_ep.interval[0]}",
                    "&",
                    f"t lt {onset+sample_ep.interval[1]}"
                    )
                )
            
            axs[0].plot(
                utils.get_unique_ids(t_df, id="t"),
                t_df["filtered"].values,
                color=ev_color,
                lw=3
            )        

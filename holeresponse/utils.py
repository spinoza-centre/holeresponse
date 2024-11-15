import os
from linescanning import utils, prf
import holeresponse as hr
import pandas as pd
import PIL
import numpy as np
opj = os.path.join
opd = os.path.dirname

try:        
    from pdf2image import convert_from_path
except Exception as e:
    raise ImportError(f"Could not find package 'pdf2image', therefore some functions are not available")

class SubjectsDict():

    def __init__(
        self, 
        ev_colors=["#1B9E77","#D95F02","#4c75ff"],
        ev_names=[
            "center",
            "medium annulus",
            "large annulus"
        ],
        fig_dir=None,
        proj_dir=None):

        self.ev_colors = ev_colors
        self.fig_dir = fig_dir
        self.proj_dir = proj_dir
        self.ev_names = ev_names

        self.color_dict = {}
        for key,val in zip(self.ev_names, self.ev_colors):
            self.color_dict[key] = val

        self.orig_names = [
            "act",
            "suppr_1",
            "suppr_2"
        ]

        self.ev_mapper = {}
        for ix,i in enumerate(self.ev_names):
            self.ev_mapper[i] = self.orig_names[ix]

        # set project directory
        if not isinstance(self.proj_dir, str):
            self.proj_dir = "/data1/projects/MicroFunc/Jurjen/projects/VE-SRF"
        
        # set derivatives
        self.deriv_dir = opj(self.proj_dir, "derivatives")

        
        # default fig dir
        self.repo_dir = opd(opd(hr.__file__))
        if not isinstance(self.fig_dir, str):
            self.fig_dir = opj(self.repo_dir, "images")
        
        self.bijanzadeh_figures = {}
        for key,val in zip(
            self.ev_names,
            ["a","c","d"]
            ):

            self.bijanzadeh_figures[key] = opj(self.fig_dir, f"Fig3{val}_bijanzadeh_2018.pdf")

        # set ribbon voxels for each subject
        self.dict_data = {
            # "sub-001": {
            #     "ribbon": (360,367),
            #     "wm": (367,372),
            #     "hemi": "L",
            #     "line_ses": 3,
            #     "invert": False,
            #     "excl_runs": {
            #         "SRFa": [], # [1,2,3],
            #         "SRFb": [], #[1,3],
            #     },
            #     "stim_sizes": [1.64,3.57,5.1],
            #     "bottom_pixels": 160,
            #     "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-001_ses-3_task-SRFa_run-demo/sub-001_ses-3_task-SRFa_run-demo_Screenshots",
            #     "ref_slc": ["task-SRFa","run-1"]
            # },
            "sub-001": {
                "ribbon": (358,366),
                "rib_dict": {
                    "SRFa": {
                        "run-1": (358,366),
                        "run-2": (357,365),
                        "run-3": (359,367),
                        "run-4": (358,366),
                    },
                    "SRFb": {
                        "run-1": (358,366),
                        "run-2": (358,366),
                        "run-3": (358,366),
                    }
                },
                "ribbon_correction": 7,
                "wm": (367,372),
                "hemi": "L",
                "line_ses": 4,
                "invert": False,
                "excl_runs": {
                    "SRFa": [1], # [1,2,3],
                    "SRFb": [], #[1,3],
                },
                "stim_sizes": [1.64,3.57,5.1],
                "bottom_pixels": 45,
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-001_ses-4_task-SRFa_run-demo/sub-001_ses-4_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },
            "sub-002": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (358,366),
                        "run-2": (361,369),
                        "run-3": (364,372), # maybe remove?
                    },
                    "SRFb": {
                        "run-1": (361,369),
                        "run-2": (356,364),
                        "run-3": (362,370),
                        "run-4": (363,371) # maybe remove?
                    }
                },
                "ribbon_correction": -8,
                "ribbon": (358,366),
                "wm": (366,372),
                "hemi": "L",
                "line_ses": 3,
                "invert": False,
                "excl_runs": {
                    "SRFa": [3],
                    "SRFb": [4],
                },
                "stim_sizes": [2.64,4.47,6.69],
                "bottom_pixels": 165,
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-002_ses-3_task-SRFa_run-demo/sub-002_ses-3_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },
            "sub-003": {
                # "rib_dict": {
                #     "SRFa": {
                #         "run-1": (354,362),
                #         "run-2": (353,361),
                #         "run-3": (354,362),
                #         "run-4": (353,361),
                #         "run-5": (353,361),
                #     },
                #     "SRFb": {
                #         "run-1": (354,362),
                #         "run-2": (354,362),
                #         "run-3": (353,361),
                #         "run-4": (352,360),
                #         "run-5": (352,360),
                #     }
                # },
                # "ribbon": (354,363),
                # "wm": (349,355),
                # "hemi": "L",
                # "line_ses": 5,
                "rib_dict": {
                    "SRFa": {
                        "run-1": (357,364),
                        "run-2": (357,364),
                        "run-3": (357,364),
                        "run-4": (357,364),
                    },
                    "SRFb": {
                        "run-1": (357,364),
                        "run-2": (357,364),
                        "run-3": (357,364),
                    },
                },              
                "ribbon": (357,364),
                "ribbon_correction": 1,
                "wm": (349,355),
                "hemi": "R",
                "line_ses": 7,                
                "invert": False,
                "excl_runs": {
                    "SRFa": [1], #[3,4,5],
                    "SRFb": [], #[1,2,3,4,5],
                },
                "bottom_pixels": 105,
                "stim_sizes": [1.89,5.36,8.84],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-003_ses-7_task-SRFa_run-demo/sub-003_ses-7_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },           
            "sub-005": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (353,361), # wrong location due to geom
                        "run-2": (353,361), # wrong location due to geom
                        "run-3": (354,362), # wrong location due to geom
                        "run-4": (357,366),
                        "run-5": (359,368),
                        "run-5": (359,368),
                        "run-6": (362,371),
                        "run-7": (363,372),
                    },
                    "SRFb": {
                        "run-1": (354,362), # wrong location due to geom
                        "run-2": (354,362), # wrong location due to geom
                        "run-3": (359,368),
                        "run-4": (359,368),
                        "run-5": (361,370),
                        "run-6": (362,371),
                        "run-7": (362,371),                        
                    }
                },
                # "ribbon": (363,371),
                # "ribbon": (356,362),
                "ribbon": (353,361),
                "ribbon_correction": 9,
                "wm": (350,356),
                # "wm": (371,376),
                "hemi": "L",
                "line_ses": 2,
                "invert": False,
                "excl_runs": {
                    "SRFa": [1,2,3],# wrong location due to geom
                    "SRFb": [1,2],  # wrong location due to geom
                },
                "bottom_pixels": 105,
                "stim_sizes": [2.39,5.28,8.17],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/projects/VE-SRF/sourcedata/sub-005/ses-2/task/sub-005_ses-2_task-SRFa_run-demo/sub-005_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-4"],
                "foldover": "FH"
            },                 
            "sub-006": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (360,367),
                        "run-2": (360,367),
                        "run-3": (365,372),
                        "run-4": (358,365),
                    },
                    "SRFb": {
                        "run-1": (362,369),
                        "run-2": (363,370),
                        "run-3": (365,372),
                        "run-4": (361,368)
                    }
                },
                "ribbon_correction": 4,
                "ribbon": (360,367),
                "wm": (367,375),
                "hemi": "L",
                "line_ses": 3,
                "invert": False,
                "excl_runs": { 
                    "SRFa": [2,3],
                    "SRFb": [1,4],
                },
                "bottom_pixels": 195,
                "stim_sizes": [3.4,4.66,5.91],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-006_ses-3_task-SRFa_run-demo/sub-006_ses-3_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },
            "sub-008": {
                "ribbon": (357,368),
                "wm": (368,373),
                "hemi": "L",
                "line_ses": 3,
                "invert": False,
                "excl_runs": {
                    "SRFa": [2,3,4],
                    "SRFb": [1,2],
                },
                "ribbon_correction": 1,
                "bottom_pixels": 0,
                "stim_sizes": [1.89,5.56,9.23],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-008_ses-3_task-SR_run-demo20231110090124/sub-008_ses-3_task-SR_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"         
            },
            "sub-010": {
                # "ribbon": (357,368), # no motion; so no rib_dict
                # "ribbon": (360,369),
                # "wm": (369,375),
                "ribbon": (358,365),
                # "wm": (349,358),
                "wm": (316,322),
                "hemi": "L",
                "line_ses": 4,
                "invert": True,
                "excl_runs": {
                    "SRFa": [2],
                    "SRFb": [1,2],
                },
                "ribbon_correction": -2,
                "bottom_pixels": 0,
                "stim_sizes": [1.76,5.84,9.92],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-010_ses-4_task-SRFa_run-demo/sub-010_ses-4_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },            
            "sub-011": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (360,368),
                        "run-2": (358,366),
                        "run-3": (355,363),
                        "run-4": (356,364),
                    },
                    "SRFb": {
                        "run-1": (356,364),
                        "run-2": (355,363),
                        "run-3": (357,365),
                        "run-4": (357,365)
                    }
                },
                "ribbon": (360,368),
                "ribbon_correction": -1,
                "wm": (356,360),
                "hemi": "L",
                "line_ses": 2,
                "invert": True,
                "excl_runs": {
                    "SRFa": [3], # positive response for large annulus..
                    "SRFb": [],
                },
                "bottom_pixels": 190,
                "stim_sizes": [2.52,5.23,7.94],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-011_ses-2_task-SRFa_run-demo/sub-011_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },
            "sub-013": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (354,363),
                        "run-2": (354,363),
                        "run-3": (356,365),
                        "run-4": (356,365),
                    },
                    "SRFb": {
                        "run-1": (351,360),
                        "run-2": (355,364),
                        "run-3": (356,365),
                        "run-4": (357,366),
                        "run-5": (359,368)
                    }
                },                
                "ribbon": (354,363),
                "ribbon_correction": 2,
                "wm": (348,354),
                "hemi": "L",
                "line_ses": 2,
                "invert": True,
                "excl_runs": {
                    "SRFa": [],
                    "SRFb": [],
                },
                "bottom_pixels": 105,
                "stim_sizes": [1.64,5.13,8.63],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/projects/VE-SRF/sourcedata/sub-013/ses-2/task/sub-013_ses-2_task-SRFa_run-demo/sub-013_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },
            "sub-014": {          
                "ribbon": (356,361), # no motion; so no rib_dict
                "ribbon_correction": 0, #4,
                "wm": (351,356),
                "hemi": "L",
                "line_ses": 2,
                "invert": True,
                "excl_runs": {
                    "SRFa": [1,4],
                    "SRFb": [1],
                },
                "bottom_pixels": 140,
                "stim_sizes": [2.39,4.55,6.71],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-014_ses-2_task-SRFa_run-demo/sub-014_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-3"],
                "foldover": "FH"
            },
            "sub-015": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (359,367),
                        "run-2": (359,367),
                        "run-3": (360,368),
                        "run-4": (360,368),
                    },
                    "SRFb": {
                        "run-1": (360,368),
                        "run-2": (360,368),
                        "run-3": (361,369),
                    }
                },
                "ribbon_correction": 5,
                "wm": (367,375),
                "hemi": "L",
                "line_ses": 2,
                "invert": False,
                "excl_runs": {
                    "SRFa": [1],
                    "SRFb": [],
                },
                "bottom_pixels": 195,
                "stim_sizes": [3.4,4.66,5.91],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-015_ses-2_task-SRFa_run-demo/sub-015_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },
            "sub-016": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (355,361),
                        "run-2": (349,355),
                        "run-3": (344,350),
                    },
                    "SRFb": {
                        "run-1": (350,356),
                        "run-2": (344,350),
                    }
                },
                "ribbon_correction": 9,
                "wm": (367,375),
                "hemi": "L",
                "line_ses": 2,
                "invert": True,
                "excl_runs": {
                    "SRFa": [2],
                    "SRFb": [],
                },
                "bottom_pixels": 100,
                "stim_sizes": [3.4,4.66,5.91],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-016_ses-2_task-SRFa_run-demo/sub-016_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "LR"
            },
            "sub-017": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (352,359),
                        "run-2": (355,362),
                        "run-3": (355,362),
                    },
                    "SRFb": {
                        "run-1": (355,362),
                        "run-2": (355,362),
                        "run-3": (355,362),
                    }
                },
                "ribbon_correction": 3,
                "wm": (367,375),
                "hemi": "L",
                "line_ses": 2,
                "invert": True,
                "excl_runs": {
                    "SRFa": [2],
                    "SRFb": [1,2,3],
                },
                "bottom_pixels": 75,
                "stim_sizes": [3.4,4.66,5.91],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-017_ses-2_task-SRFa_run-demo/sub-017_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },            
            "sub-018": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (359,366),
                        "run-2": (358,365),
                        "run-3": (353,360),
                        "run-4": (355,362),
                    },
                    "SRFb": {
                        "run-1": (357,364),
                        "run-2": (355,362),
                        "run-3": (354,361),
                    }
                },
                "ribbon": (359,366),
                "ribbon_correction": 0,
                "wm": (367,375),
                "hemi": "L",
                "line_ses": 3,
                "invert": True,
                "excl_runs": {
                    "SRFa": [3,4],
                    "SRFb": [3],
                },
                "bottom_pixels": 90,
                "stim_sizes": [3.4,4.66,5.91],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-018_ses-3_task-SRFa_run-demo/sub-018_ses-3_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },
            "sub-019": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (359,367),
                        "run-2": (358,366),
                        "run-3": (358,366),
                        "run-4": (358,366),
                    },
                    "SRFb": {
                        "run-1": (358,366),
                        "run-2": (359,367),
                        "run-3": (358,366),
                        "run-4": (358,366),
                    }
                },
                "ribbon_correction": 3,
                "wm": (367,375),
                "hemi": "R",
                "line_ses": 2,
                "invert": True,
                "excl_runs": {
                    "SRFa": [],
                    "SRFb": [1,2,3],
                },
                "bottom_pixels": 90,
                "stim_sizes": [3.4,4.66,5.91],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-019_ses-2_task-SRFa_run-demo/sub-019_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },      
            "sub-020": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (357,365),
                        "run-2": (357,365),
                        "run-3": (359,367),
                    },
                    "SRFb": {
                        "run-1": (357,365),
                        "run-2": (358,366),
                        "run-3": (359,367),
                    }
                },
                "ribbon_correction": 3,
                "wm": (367,375),
                "hemi": "R",
                "line_ses": 2,
                "invert": False,
                "excl_runs": {
                    "SRFa": [2],
                    "SRFb": [1],
                },
                "bottom_pixels": 120,
                "stim_sizes": [3.4,4.66,5.91],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-020_ses-2_task-SRFa_run-demo/sub-020_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },  
            "sub-022": {
                "rib_dict": {
                    "SRFa": {
                        "run-1": (358,363),
                        "run-2": (358,363),
                        "run-3": (357,362),
                    },
                    "SRFb": {
                        "run-1": (358,363),
                        "run-2": (358,363),
                        "run-3": (358,363),
                    }
                },
                "ribbon_correction": 0, #4,
                "wm": (367,375),
                "hemi": "R",
                "line_ses": 2,
                "invert": True,
                "excl_runs": {
                    "SRFa": [],
                    "SRFb": [],
                },
                "bottom_pixels": 90,
                "stim_sizes": [3.4,4.66,5.91],
                "scr_dir": "/data1/projects/MicroFunc/Jurjen/programs/project_repos/LineExps/ActNorm3/logs/sub-022_ses-2_task-SRFa_run-demo/sub-022_ses-2_task-SRFa_run-demo_Screenshots",
                "ref_slc": ["task-SRFa","run-1"],
                "foldover": "FH"
            },                       
        }

    def has_ribdict(self, subject):
        has_dict = False
        if "rib_dict" in list(self.dict_data[subject].keys()):
            has_dict = True

        return has_dict

    def get_pars(self, subject):
        return self.get_coord_obj(subject).data
    
    def get_srf(self, subject, normalize=False):

        # get size response function
        pars = self.get_pars(subject)
        SR_ = prf.SizeResponse(params=pars.reset_index(), model="norm")

        # size response
        fill_cent, fill_cent_sizes = SR_.make_stimuli(
            factor=1,
            dt="fill"
        )

        sr_cent_act = SR_.batch_sr_function(
            SR_.params_df,
            center_prf=True,
            stims=fill_cent,
            sizes=fill_cent_sizes
        )

        max_dva, _ = SR_.find_stim_sizes(
            sr_cent_act[0].values,
            t="max",
            dt="fill",
            sizes=fill_cent_sizes,
            return_ampl=True
        )

        sr = sr_cent_act.iloc[:,0].values
        if normalize:
            sr /= sr.max()

        return {
            "srf": sr,
            "stim_sizes": fill_cent_sizes,
            "max_response": sr.max(),
            "max_size": max_dva
        }

    def get_coord_obj(self, subject):
        coord_file = self.get_coord_file(subject)
        return utils.VertexInfo(coord_file)
    
    def get_coord_file(self, subject):
        ls_ses = self.dict_data[subject]["line_ses"]
        pyc_path = opj(
            self.deriv_dir,
            "pycortex",
            subject,
            f"ses-{ls_ses}"
        )

        coord_file = opj(pyc_path, f"{subject}_ses-{ls_ses}_model-norm_desc-best_vertices.csv")

        if os.path.exists(coord_file):
            return coord_file
        else:
            raise FileNotFoundError(f"Could not find coordinate file '{coord_file}'")
    
    def get(self, subject, key):
        return self.dict_data[subject][key]

    def get_invert(self, subject):
        return self.get(subject, "invert")
    
    def get_ephys_figures(self):
        return self.bijanzadeh_figures
    
    def get_scr_dir(self, subject):
        return self.get(subject, "scr_dir")
    
    def get_files(self, subject, **kwargs):
        return utils.FindFiles(opj(self.proj_dir, subject), **kwargs).files
    
    def get_niftis(self, subject):
        return self.get_files(subject, extension="nii.gz")
    
    def get_slc_criteria(self, subject):
        return self.get(subject, "ref_slc")    
    
    def get_slc(self, subject):
        all_files = self.get_niftis(subject)
        ses = self.get_session(subject)
        crit = ["acq-1slice",f"ses-{ses}/"]+self.get_slc_criteria(subject)
        return utils.get_file_from_substring(crit, all_files, exclude="beam")
    
    def get_beam(self, subject):
        all_files = self.get_niftis(subject)
        ses = self.get_session(subject)
        crit = ["_bold.nii.gz",f"ses-{ses}/"]+self.get_slc_criteria(subject)
        return utils.get_file_from_substring(crit, all_files, exclude=["3DEPI"])    
    
    def get_subjects(self):
        return list(self.dict_data.keys())
    
    def get_colors(self):
        return self.ev_colors  

    def get_evs(self):
        return self.ev_names  

    def get_views(self, subject):
        return self.get(subject, "views")

    def get_session(self, subject):
        return self.get(subject, "line_ses")

    def get_hemi(self, subject):
        return self.get(subject, "hemi")

    def get_target(self, subject):
        ls_ses = self.get_session(subject)
        ls_hemi = self.get_hemi(subject)
        vert_file = opj(
            os.environ.get("DIR_DATA_DERIV"),
            "pycortex",
            subject,
            f"ses-{ls_ses}",
            f"{subject}_ses-{ls_ses}_desc-coords.csv"
        )
        vv = utils.VertexInfo(vert_file)
        
        return vv.get("index", hemi=ls_hemi)
    
    def get_ribbon(self, subject, from_unique=True):

        if from_unique:
            if self.has_ribdict(subject):
                ref = self.get(subject, "ref_slc")
                taskID = ref[0].split('-')[-1]
                runID = ref[1]

                rib_dict = self.get(subject, "rib_dict")
                ribbon = rib_dict[taskID][runID]
            else:
                ribbon = self.get(subject, "ribbon")
        else:
            ribbon = self.get(subject, "ribbon")

        return ribbon

    def get_wm(self, subject):
        return self.get(subject, "wm")

    def get_exclude(self, subject):
        return self.get(subject, "exclude")

    def get_screen_size(self, subject):
        return self.get(subject, "screen_size")

    def get_bounds(self, subject):
        return self.get(subject, "bounds")

    def get_extent(self, subject):
        return self.get(subject, "extent")

    def get_excl_runs(self, subject, task="SRFa"):
        return self.get(subject, "excl_runs")[task]

    def get_stimuli(self, subject):
        return self.get(subject, "stim_sizes")

    def create_stim_df(self, ev_names=["act","suppr_1","suppr_2"]):
        subjs = self.get_subjects()
        df = []
        for sub in subjs:
            stims = pd.DataFrame(self.get_stimuli(sub), columns=["stim_size"])
            stims["subject"],stims["event_type"] = sub.split('-')[-1], ev_names
            df.append(stims)

        df = pd.concat(df)
        df.set_index(["subject","event_type"], inplace=True)
        return df

def read_pdf_image(img):
    if isinstance(img, str):
        conv_img = convert_from_path(img)
        pil_img = conv_img[0]
        img_data = np.asarray(pil_img)
    elif isinstance(img, PIL.PpmImagePlugin.PpmImageFile):
        img_data = np.asarray(pil_img)
    else:
        img_data = img.copy()

    return img_data

def fetch_level_values(ddict, df):
    
    t_ax = list(ddict["orig"].columns)
    n_contours = len(ddict["contours"].allsegs)

    n_segments = []
    avg_t = []
    for ix,i in enumerate(ddict["contours"].allsegs):
        
        n_segs = len(i)
        if n_segs>0:
            avg_idx = int(round(i[-1][:,0].mean(),0))
            avg_t.append(t_ax[avg_idx])

    total_contours = sum(n_segments)
    lvl_vals = [df[i].mean() for i in avg_t]
    return lvl_vals

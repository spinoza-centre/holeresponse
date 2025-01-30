import os
from linescanning import utils, prf
import holeresponse as hr
import pandas as pd
import PIL
import numpy as np
import cv2
from scipy import stats
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

def get_vascular_weights(factor=None, **kwargs):

    # Original matrix as inferred from the table
    matrix = np.array([
        [1.00, 0.41, 0.59, 0.20, 0.26],
        [0.00, 1.00, 0.59, 0.20, 0.26],
        [0.00, 0.00, 1.00, 0.20, 0.32],
        [0.00, 0.00, 0.00, 1.00, 0.32],
        [0.00, 0.00, 0.00, 0.00, 1.00]
    ])

    if isinstance(factor, int):
        matrix_float = matrix.astype(np.float32)

        # set default interpolation method
        kwargs = utils.update_kwargs(
            kwargs,
            "interpolation",
            cv2.INTER_LANCZOS4
        )

        # resample
        resampled_matrix = cv2.resize(
            matrix_float, 
            (factor, factor),
            **kwargs
        )

        return resampled_matrix
    else:
        return matrix

def normalize_density(density):
    """Normalize vascular density to integrate to 1."""
    return density / np.sum(density)

def generate_microvascular_comp(n, offset=None, normalize=True):
    cortical_depth = np.linspace(0, 1, n)
    if not isinstance(offset, (int,float)):
        offset = n//2

    micro = np.exp(-offset * (cortical_depth - 0.5) ** 2)
    
    if normalize:
        # micro = normalize_density(micro)
        micro = micro/micro.max()

    return micro

def generate_macrovascular_comp(n, offset=None, normalize=True, fliplr=True):
    cortical_depth = np.linspace(0, 1, n)

    if not isinstance(offset, (int,float)):
        offset = n//4

    macro = 1 - np.exp(offset * cortical_depth)
    macro = normalize_density(macro)
    
    if normalize:
        macro = macro/macro.max()

    if fliplr:
        macro = macro[::-1]

    return macro

def generate_double_bump_model(n, fc=5, add_intercept=True, **kwargs):
    
    x = np.arange(0,n)
    y = np.linspace(-n//2,n//2,num=n)
    icpt = np.ones((n,1))

    # linear component (reverse so its CSF > WM)
    lin = x.copy()[::-1][...,np.newaxis]
    lin = lin/lin.max()

    # superficial bump
    y_1 = stats.norm.pdf(y,-fc)
    y_1 = (y_1/y_1.max())[...,np.newaxis]
    y_1d = np.gradient(y_1.squeeze())[...,np.newaxis]
    
    # deep bump
    y_2 = stats.norm.pdf(y,fc)
    y_2 = (y_2/y_2.max())[...,np.newaxis]
    y_2d = np.gradient(y_2.squeeze())[...,np.newaxis]

    # vascular
    micro = generate_microvascular_comp(n, **kwargs)[...,np.newaxis]
    macro = generate_macrovascular_comp(n, **kwargs)[...,np.newaxis]

    # construct dictionary
    reg_dict = {
        "linear": lin,
        "double_bump": np.concatenate(
            [
                lin,
                y_1,
                y_2
            ],
            axis=1
        ),
        "double_bump_deriv": np.concatenate(
            [
                lin,
                y_1,
                y_2,
                y_1d,
                y_2d
            ],
            axis=1
        ),
        "double_bump_deriv_macro": np.concatenate(
            [
                macro,
                y_1,
                y_2,
                y_1d,
                y_2d
            ],
            axis=1
        ),
        "double_bump_deriv_micro": np.concatenate(
            [
                micro,
                y_1,
                y_2,
                y_1d,
                y_2d
            ],
            axis=1
        ),
        "double_bump_deriv_vascular": np.concatenate(
            [
                macro,
                micro,
                y_1,
                y_2,
                y_1d,
                y_2d
            ],
            axis=1
        )            

    }

    c_vecs = [
        np.array([[1]]),
        np.array([[0,1,1]]),
        np.array([[0,1,1,1,1]]),
        np.array([[0,1,1,1,1]]),
        np.array([[0,1,1,1,1]]),
        np.array([[0,0,1,1,1,1]]),
    ]

    skip_for_context = np.array([0,1,1,1,1,2])

    if add_intercept:
        for ix,(key,val) in enumerate(reg_dict.items()):
            reg_dict[key] = np.concatenate([icpt, val], axis=-1)
            c_vecs[ix] = np.insert(c_vecs[ix], 0, 0)[np.newaxis,...]

        skip_for_context +=1
    return reg_dict, c_vecs, skip_for_context

def add_level_to_multiindex(
    df, 
    new_level_values, 
    insert_position, 
    new_level_name="new_level"):

    """
    Add a new level to the MultiIndex of a dataframe at a specified position.

    Parameters:
    - df: pandas DataFrame with a MultiIndex.
    - new_level_values: List of values for the new level, or a single value to apply to all rows.
    - insert_position: Integer specifying where to insert the new level in the MultiIndex.
    - new_level_name: String specifying the name of the new level (default is "new_level").

    Returns:
    - df: DataFrame with updated MultiIndex including the new level.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("The dataframe must have a MultiIndex.")
    
    # Check if the new level name already exists in the MultiIndex
    if new_level_name in df.index.names:
        print(f"The level name '{new_level_name}' already exists in the MultiIndex. Skipping process.")
        return df  # Return the original dataframe without modification

    # If new_level_values is a single value, replicate it to match the length of the MultiIndex
    if not isinstance(new_level_values, list):
        new_level_values = [new_level_values] * len(df.index)
    
    # Ensure the new level values match the length of the index
    if len(new_level_values) != len(df.index):
        raise ValueError("The length of new_level_values must match the number of rows in the MultiIndex.")
    
    # Create new tuples by inserting the new level values
    new_tuples = [
        tuple(idx[:insert_position]) + (new_value,) + tuple(idx[insert_position:])
        for idx, new_value in zip(df.index, new_level_values)
    ]
    
    # Get the original index names and insert the new level name
    new_index_names = list(df.index.names)
    new_index_names.insert(insert_position, new_level_name)
    
    # Construct the new MultiIndex
    new_multi_index = pd.MultiIndex.from_tuples(new_tuples, names=new_index_names)
    
    # Assign the updated MultiIndex to the dataframe
    df.index = new_multi_index
    
    return df
    
def run_models(df, **kwargs):

    reg_dict, c_vecs, skip_idx = generate_double_bump_model(df.shape[-1])
    beta_full,beta_sum,preds = [],[],[]
    for ix,(key,X) in enumerate(reg_dict.items()):

        print(f"model #{ix+1} ('{key}'); skipping index: {skip_idx[ix]}")
        res = run_model_for_events(
            X,
            df,
            model_name=key,
            n_skip_reg=skip_idx[ix],
            **kwargs
        )

        beta_full.append(res["beta_full"])
        beta_sum.append(res["beta_sum"])
        preds.append(res["predictions"])

    res = {}
    for i,lbl in zip([beta_full,beta_sum,preds],["beta_full","beta_sum","predictions"]):
        if len(i)>1:
            i = pd.concat(i)
        else:
            i = i[0]

        res[lbl] = i.copy()

    return res

def run_model_for_events(X,df,**kwargs):
    evs = utils.get_unique_ids(df, id="event_type", sort=False)
    beta_full,beta_sum,preds = [],[],[]
    for ev in evs:
        ev_df = utils.select_from_df(df, expression=f"event_type = {ev}")
        res = run_model_glm(
            X,
            ev_df,
            **kwargs
        )

        beta_full.append(res["model_full"])
        beta_sum.append(res["model_sum"])
        preds.append(res["predictions"])

    res = {}
    for i,lbl in zip([beta_full,beta_sum,preds],["beta_full","beta_sum","predictions"]):
        if len(i)>1:
            i = pd.concat(i)
        else:
            i = i[0]

        res[lbl] = i.copy()

    return res

def run_model_glm(
    X, 
    df, 
    transpose=True, 
    n_skip_reg=1,
    model_name=None,
    **kwargs
    ):

    if transpose:
        Y = df.values.T
    else:
        Y = df.values.copy()

    beta,sse,rank,s = np.linalg.lstsq(X,Y)
    reduced_model = beta[n_skip_reg:,:].sum(axis=0)

    # format full model so that subjects become columns and regressors in rows
    df_full = pd.DataFrame(beta, index=df.index[:X.shape[-1]])
    idx = [i for i in list(df_full.index.names) if i !=  "subject"]
    df_full.reset_index(inplace=True)
    df_full.drop(["subject"], axis=1, inplace=True)
    df_full.set_index(idx, inplace=True)
    
    # reduce model
    df_reduced = pd.DataFrame(reduced_model, columns=["beta"], index=df.index)

    # generate predictions
    preds = np.dot(X,beta)
    df_pred = pd.DataFrame(preds.T, index=df.index)
    
    if isinstance(model_name, (str,int)):
        df_full = add_level_to_multiindex(df_full, model_name, -1, "model")
        df_reduced = add_level_to_multiindex(df_reduced, model_name, -1, "model")
        df_pred = add_level_to_multiindex(df_pred, model_name, -1, "model")

    return {
        "model_full": df_full, 
        "model_sum": df_reduced,
        "predictions": df_pred
    }

def generate_boxcar(x_shape=20, centers=[5, 15], width=5, y_max=0.2):
    """
    Generates a boxcar function with specified box widths and centers.

    Parameters:
    - x_shape (int): Total number of points in x-axis.
    - centers (list): List of center positions for the boxcar peaks.
    - width (int): Width of each box.
    - y_max (float): Maximum y value of the boxcar function.

    Returns:
    - x (numpy array): X-axis values.
    - y (numpy array): Corresponding Y-axis values (boxcar function).
    """
    x = np.arange(x_shape)
    y = np.zeros_like(x, dtype=float)

    for center in centers:
        start = max(0, center - width // 2)
        end = min(x_shape, center + width // 2 + 1)
        y[start:end] = y_max

    return x, y

def generate_model_example(n=20, **kwargs):

    reg_dict,_,_ = generate_double_bump_model(n)
    x, box_car = generate_boxcar(x_shape=n, **kwargs)

    return [
        box_car.squeeze(),
        reg_dict["double_bump_deriv"][:,2],
        reg_dict["double_bump_deriv"][:,3],
    ]

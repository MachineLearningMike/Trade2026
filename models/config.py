import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import platform

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

config = {
    "dual_drive": False,
    "train_start_dt": datetime(2018, 1, 1, 0),
    "vali_start_dt": datetime(2025, 7, 1, 0),
    "test_start_dt": datetime(2026, 1, 1, 0),
    "test_end_dt_inclusive": datetime(2026, 3, 30, 0),
    "seed": 523,
    
    "train_hours_limit": 240,
    "dollarsPerHour": 0.1,
}
projects = {}

#----------------------------------- platform-specific params ---------------------------
if platform.system() == "Windows":  
        config["dataset_dir"] = "d:/trade_data/history"
        config["management_dir"] = "d:/trade_data/management"
else:
    pass

#----------------------------------- projects, historical --------------------------------

projects["M18"] = {
    "management": {
        "model_module" : "model_m18",
        "metrics" : "rmse", # "mae", "rmse"
        "steps_to_save": 100,
        
        'choose_best_loss_model': False,
        
        
        "nSubstepsPerStep": 2,
        'initial_steps': 200,
        'half_life' : int(10228 * 80),    # covers  epochs
        'overwrite_saved_lr': False,
    },
    
    "data": {
        "factor10": 10,
        "org_height": 384,
        "batch_size": 10,
    },
    
    "model": {
        "optimizer": "adamw",
        'block_dicts' : [

            {
            }
        ]
    }
}

#------------------------------------- generate params -------------------------------

project = "M18"
config["bayesProject"] = project

config.update(projects[project]["management"])
config.update(projects[project]["data"])
config.update(projects[project]["model"])

params = SimpleNamespace(**config)

print(f"Project: {params.bayesProject}")
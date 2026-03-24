import torch
import numpy as np
import json, tzlocal
from torch.amp import autocast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import sys, gc, copy, time, math
from importlib import import_module
from datetime import datetime, timedelta

# import matplotlib
# matplotlib.use("Agg")  # non-GUI backend, safe in threads
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

# from modules.utils import check_dir, check_gradient
import modules.utils as utils
from modules.zeus_tqdm import zeus_tqdm
from models.config import params
from modules.read_cursor_keys_linux_windows import KeyReader

from models.model_dom import Model_Dom
from modules.Zeus_Dataset import Zeus_Dataset

kr = KeyReader()
kr.init_key_listener()


class Model_Trainer():
    def __init__(self, pm, bayes=None, trainable=True):
        self.pm = pm
        self.bayes = bayes
        self.project_path = Path(pm.management_dir) / pm.bayesProject
        self.setup_logging(self.project_path)

        self.dataset = Zeus_Dataset(pm)
        # self.dataset = Zeus_DataLoader(Zeus_Dataset(self.pm, dataset_path, management_path), batch_size=None, shuffle=None)     # A bit slower?
        
        self.dataset.load_data()
        logging.info("Zeus_Dataset created and loaded.")
        self.device_type, self.nDevices, self.device = self._acquire_device()
        self.devices = [ self.device_type + ':' + str(id) for id in range(self.nDevices)]
        
        self.md = Model_Dom(self.devices[0], pm, self.dataset, project_dir, bayes=bayes, trainable=trainable)
            
        self.focus_md_id = 0
        self.transient_step = 0
        self.train_bombs = 1
        
        self.elapse = 0.0
        self.train_start_time = time.time()
        
    def setup_logging(self, log_dir):
        """
        logging.info("This is an info message")
        logging.warning("Watch out! This is a warning")
        logging.error("An error happened")
        logging.debug("Debug details here (won't show unless level is DEBUG)")
        """
        # Remove all handlers associated with the root logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        log_path = log_dir / 'log.txt'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler (append mode)
        utils.check_dir(log_path.parent)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(formatter)

        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
        
    def _acquire_device(self):
        if torch.cuda.is_available():
            device_type = 'cuda'
            default_device = torch.device('cuda:0')
            nDevices = torch.cuda.device_count()
        else:
            device_type = 'cpu'
            default_device = torch.device('cpu:0')
            nDevices = torch.cpu.device_count()
                   
        return device_type, nDevices, default_device
    
    def save_state(self, name=None):
        if name is None: name = "train_state_trainer"  # + '-' + self.get_train_id_string()
        checkpoint = {
            'dataset': self.dataset.pack_state(),
            'transient_step': self.transient_step,
            'elapse': self.elapse,
        }
        utils.check_dir(self.project_path)
        torch.save(checkpoint, self.project_path / (name + '.pth'))
        torch.save(checkpoint, self.project_path / (name + '.pth.bak'))   # Create a backup, as creating takes a long time and may fail by the cloud Spot service.
        return
    
    def try_load_state(self, name=None):    # Pairs with save_state
        if name is None: name = name = "train_state_trainer" #+ self.get_train_id_string()
        
        file_found = True
        
        try:
            checkpoint_path = self.project_path / (name + '.pth')
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            logging.info("Trainer checkpoint loaded from the .pth file")
        except Exception as e:
            logging.info(str(e))
            try:
                checkpoint_path = self.project_path / (name + '.pth.bak')     # Maybe the previous save.
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                logging.info("Trainer checkpoint loaded from the .pth.bak file")
            except Exception as e:
                logging.info(str(e))
                file_found = False
                
        if file_found:        
            if checkpoint.get('dataset', None) is not None:
                self.dataset.unpack_state(checkpoint['dataset'])
                logging.info("dataset state loaded with saved state")

            if checkpoint.get('transient_step', None) is not None:
                self.transient_step = checkpoint['transient_step']
                logging.info("transient_step loaded: {}".format(self.transient_step))
            if checkpoint.get('elapse', 0.0) != 0.0:
                self.elapse = checkpoint['elapse']
                logging.info("elpase loaded: {}".format(self.elapse))
        else:
            pass
        
    def train(self):        

        # Clear everything
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Reset cudnn
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.md.zero_losses_metrics()
        self.md.try_load_state()
        self.try_load_state()        
        
        ran_epochs = False
        best_metric_yet = 1e10
                
        logging.info("============= Starting train ... ==============")
                
        for epoch in range(0, self.pm.epochs):
            ran_epochs = True

            with torch.set_grad_enabled(True):
                train_losses, train_metrics = self.run_epoch("train_set", trainable=True)   # (,), (C,)
 
            self.md.get_epoch_display_batch(train_losses, train_metrics)

            logging.info("Finished train epoch {}, {}".format(self.md.epoch, (self.md.display)))
    
            keep_going = True
            best_metric = None
    
            if self.md.pm.skip_validation:
                logging.info("Validation skipped as configured with 'skip_validation'")
            else:
                with torch.set_grad_enabled(False):
                    vali_losses, vali_metrics = self.run_epoch("vali_set", trainable=False)   # (,), (C,)
                
                self.md.get_epoch_display_batch(vali_losses, vali_metrics)
                logging.info("Finished vali epoch {}, {}".format(self.md.epoch, self.md.display))

                keep_going, best_metric = self.md.save_best_train(vali_metrics)                

            self.md.save_state(epoch_finished=True)           
            self.save_state()
            
            best_metric_yet = best_metric or self.md.vali_metric_old
            if self.bayes is not None:  self.bayes.on_epoch_end()
                    
            if not keep_going: break
            
        self.md.save_state(train_finished=True)
        self.save_state()
        logging.info("!!! Train finished")

        return - best_metric_yet
    

    def show_data(self, batch_x):
        B, T, C, H, W = batch_x.shape
        # Pick random b, t, c indices
        b = torch.randint(0, B, (1,)).item()
        t = torch.randint(0, T, (1,)).item()
        c = torch.randint(0, C, (1,)).item()
        print(f"Selected slice: b={b}, t={t}, c={c}")

        # Extract 2D slice
        slice_2d = batch_x[b, t, c].cpu().numpy()  # shape (H, W)
        mean = np.mean(slice_2d)
        std = np.std(slice_2d)

        # Create coordinate grid
        x = np.arange(W)
        y = np.arange(H)
        Xg, Yg = np.meshgrid(x, y)

        # Plot as 3D surface
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(Xg, Yg, slice_2d, cmap='viridis', edgecolor='none')

        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Value')
        ax.set_title(f"3D Surface of X[b={b}, t={t}, c={c}, mean={mean}, std={std}]")

        plt.tight_layout()
        plt.show()
    
    def clean_gpu_every(self, n_iter, step):
        if step % n_iter == 0:
            gc.collect()
            torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()  # only for inter-process communicatoin
    
    def respond_to_cursor(self):
        direction, count = kr.read_key()
        factor = None; step = math.pow(10.0, 1/10)

        if direction is None:
            return
        elif direction == "up":
            factor = math.pow(step, count)         
        elif direction == "down":
            factor = math.pow(1 / step, count)
        elif direction == "left":
            self.train_bombs = self.train_bombs - count if self.train_bombs - count >= 1 else 1
            return
        elif direction == "right":
            self.train_bombs = self.train_bombs + count
            return
        elif direction == "0":
            self.focus_md_id = 0
            return
        elif direction == "1":
            self.focus_md_id = 1
            return
        elif direction == "2":
            self.focus_md_id = 2
            return
        elif direction == "3":
            self.focus_md_id = 3
            return
        elif direction == "4":
            self.focus_md_id = 4
            return
        else: return

        focus_md = self.md

        if factor:
            lr = focus_md.get_learning_rate()
            assert lr is not None
            lr *= factor
            focus_md.set_learning_rate(lr)
            # lr = focus_md.get_learning_rate()
            # print(f"lr = {lr}")

        return
    
    
    def run_epoch(self, x_set_label, trainable=True):
               
        # self.dataset.set_data_slots(data_slots)
        renewed = self.dataset.refresh_x_set(x_set_label, seed=self.pm.seed, renew_if_new_epoch=True)
        # logging.info("{} {}".format(x_set_label, "renewed" if renewed else "left over from previous run"))

        task = "training" if trainable else "validation"
        Doing = "Continuing" if trainable else "Starting"
        
        if self.md.epoch_finished:
            logging.info(f"--------- Starting {task} on new epoch... {self.md.epoch}")
        else:            
            logging.info(f"--------- {Doing} {task} on saved epoch... {self.md.epoch}")

        self.md.zero_losses_metrics()
                        
        progress_bar = zeus_tqdm(self.dataset)
        progress_bar.start_prefetch_thread()

        if self.nDevices > 1: gpu_executor = ThreadPoolExecutor(max_workers=len(self.devices))
        else: gpu_executor = None
        
        if not progress_bar.no_more_data():  progress_bar.prefetch(x_set_label)        # Fetch one batch asynchronously

        train_dict = {
            "type_metrics":              self.pm.metrics,
        }

        step = 1
        start = time.time()
        start1 = time.time()
        for batch in progress_bar:              # Wait for the asynchronous .prefetch
            end = time.time()
            
            if (end-self.train_start_time) > self.pm.train_hours_limit * 3600: break

            if not progress_bar.no_more_data():  progress_bar.prefetch(x_set_label)        # Prefetch ensures getting new bacthes.
            
            if False:
                self.show_data(batch_x)
                continue

            self.md.optimizer.zero_grad(set_to_none=True)
                              
            epoch_step_loss = []; epoch_step_metrics = []; substeps_per_step = 0

            # with autocast(device_type='cuda', dtype=torch.bfloat16):    # time: 97 -> 82, memory: little
            #     #------------------------ this algorithm is studied in zeus_train_6_hourly.py -----------
                        
            substep_loss, substep_metrics = self.md.forward_backward_no_optimizer_step(batch, train_dict, trainable)
            
            if trainable:
                substeps_per_step += 1
                self.md.accum_substeps += 1

            epoch_step_loss.append(substep_loss)          # [substep_loss...]
            epoch_step_metrics.append(substep_metrics)    # [substep_metrics...]                    
                
            # #------------------------------------------------ The set of steps finished.
            self.md.store_step_loss_and_metrics(epoch_step_loss, epoch_step_metrics) # [substep_loss...], [substep_metrics...]
            
            if trainable:
                utils.check_gradient(self.md.model)
                
                self.md.optimizer.step()     # # instead of the below lines
                # scaler doesn't support bf16?  ---- this use of scaler is coupled with the use of scaler in  forward_backward_no_optimizer_step
                # self.md.scaler.step(self.md.optimizer)
                # self.md.scaler.update()
                
                if self.pm.activate_scheduler:  self.md.scheduler.step()

            end1 = time.time()
            ms_wait = (end-start) * 1000
            ms_step = (end1-start1) * 1000
            self.elapse = self.elapse + (end1-start1)
            elapse = f"{int(self.elapse) // 3600}:{(int(self.elapse) % 3600) // 60 :02d}"
            # compute = (ms_step - ms_wait) / self.train_bombs
            steps_per_hour = 3600/(end1-start1)
            substepsPerDollar = int(substeps_per_step * steps_per_hour / self.pm.dollarsPerHour)
            lr = self.md.get_learning_rate()
            progress_bar.set_description(f"{self.md.pbar_description}, elapse {elapse} wait/step {ms_wait:.0f}/{ms_step:.0f} lr {lr:1.3e}")
            start1 = time.time()

            # This is to save the model in case it suddenly deviates to diverge.
            if trainable is True:
                self.md.save_best_loss_model(self.md.step_loss, elapse)
            
            if trainable is True and step % self.pm.steps_to_save == 0:
                self.md.pack_losses_metrics()
                self.md.save_state()
                self.save_state()
            self.clean_gpu_every(1, step)
            self.respond_to_cursor()
            
            start = time.time()
        
            step += 1
        
        self.md.get_pbar_description_epoch()
        progress_bar.set_description(self.md.pbar_description)

        progress_bar.stop_prefetch_thread()
        
        if trainable is True: 
            self.md.pack_losses_metrics()
            self.md.save_state()
            self.save_state()
        if gpu_executor is not None: gpu_executor.shutdown()

        epoch_losses = self.md.epoch_losses
        epoch_metrics = self.md.epoch_metrics

        return epoch_losses, epoch_metrics  # (,), (C,)
    

    def play_test(self):
        
        from modules.utils import show_two_arrays, initialize_for_show_array, finalize_for_show_array
        
        def get_key():
            """
            Get the most recent key press without buffering.
            Returns None if no key is currently pressed.
            """
            import keyboard
            import time
            
            # Check all keys we care about
            keys_to_check = {
                'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
                'esc': 'quit',
            }
            
            # Check if any key is currently pressed
            for key, action in keys_to_check.items():
                if keyboard.is_pressed(key):
                    # Wait for key release to ensure we don't get multiple triggers
                    while keyboard.is_pressed(key):
                        time.sleep(0.01)
                    return action
            
            return None
        
        initialize_for_show_array()
        
        #----------------------------------------------------------------------
               
        # self.dataset.set_data_slots(data_slots)
        renewed = self.dataset.refresh_x_set('test_set')
        # logging.info("{} {}".format(x_set_label, "renewed" if renewed else "left over from previous run"))

        logging.info(f"--------- Starting interactive testing...")

        self.md.zero_losses_metrics()
                         
        step = 1
        start = time.time()
        start1 = time.time()
        
        dt = inclusive_left
                
        old_path = None; key_stroked = True
        while inclusive_left <= dt and dt <= inclusive_right:            
            
            if key_stroked:

                file_path = self.get_file_path_npz(dt.year, dt.month, dt.day, dt.hour, modulo)
                if file_path != old_path:
                    dict_data = load_and_decode(file_path)
                    old_path = file_path

                variable = self.pm.meteo_variables[var_id]
                arr_y = dict_data[variable]  # (C, H, W)

            #--------------------------------------------------------------
            
            singleton_batch = self.dataset.take_singleton_batch(dt)
            
            if singleton_batch is not None:
                pred, true = self.md.predict(singleton_batch)            
                pred, true = pred[0], true[0]

                # Display the array
                show_two_arrays(arr1=arr_y, arr2=None, variable1=variable, variable2="", dt=dt, modulo=modulo)

            #---------------------------------------------------------------
            
            # Wait for key input
            key = None
            start_time = time.time()
            while key is None and (time.time() - start_time < 0.5):
                key = get_key()
                if key is None:
                    time.sleep(0.1)  # Small sleep to prevent CPU overuse
            
            # Process key
            key_stroked = True
            
            if key == "up": 
                var_id += 1
                var_id = 0 if var_id >= len(self.pm.meteo_variables) else var_id
            elif key == "down": 
                var_id -= 1
                var_id = len(self.pm.meteo_variables) - 1 if var_id < 0 else var_id
            elif key == "left": 
                dt = dt - timedelta(hours=modulo)
            elif key == "right": 
                dt = dt + timedelta(hours=modulo)
            elif key == 'sep':
                h = dt.hour
                dict = Snapshot.get_season_events(dt.year)
                dt = dict['sep_equinox'].replace(hour=dt.hour, minute=0)
            elif key == "quit":
                print("Quitting...")
                break
            elif key is None:
                # No key pressed, continue to next iteration
                key_stroked = False
                continue

        finalize_for_show_array()
            

  
    
    
if __name__ == "__main__":
    
    from models.config import params
    
    pm = params
    # utils.check_dir(management_path)
    # project_path = management_path / "projects" / pm.bayesProject
    # utils.check_dir(project_path)
    
    project_dir = pm.management_dir + "/" + pm.bayesProject
    
    model_sequence_trainer = Model_Trainer(pm, bayes=None, trainable=True)
    model_sequence_trainer.train()
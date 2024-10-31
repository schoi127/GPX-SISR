import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
############################################################################
from datetime import datetime
import schedule
import time
import os
from pathlib import Path
from time import sleep
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
############################################################################

class FileCreateHandler(FileSystemEventHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback  # store callback function

    def on_created(self, event):
        if not event.is_directory:
            # full file path
            file_path = event.src_path
            # extract only filename
            file_name = os.path.basename(file_path)

            # transfer the current filename to the call-back function
            self.callback(file_name)

def monitor_folder(path_to_watch, callback):
    event_handler = FileCreateHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)  # Keep file monitoring every this seconds
            print(f"Start file monitoring every 1s. Path: {path_to_watch}")
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def mkdir_and_rename(path):
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
        
    os.makedirs(path, exist_ok=True)
def get_list_dir(mother_dir):
    folder_list = []
    mother_dir = Path(mother_dir)
    for entry in os.listdir(mother_dir):
        if os.path.isdir(os.path.join(mother_dir, entry)):
            folder_list.append(entry)
    return folder_list
def get_files_all_subfolder(main_dir, extensions):
    file_paths = []
    for (root, directories, files) in os.walk(main_dir):
        for file in files:
            if extensions in file:
                file_path = Path(os.path.join(root, file))
                file_paths.append(file_path)
    return file_paths
############################################################################


def run_validation(opt):    
    print(f"Validation check...")

    torch.backends.cudnn.benchmark = True
    opt['path']['results_root'] = os.path.dirname(opt['datasets']['test_1']['dataroot_lq'])
    opt['path']['log'] = opt["path"]["results_root"]
    opt['path']['visualization'] = opt['path']['results_root']

    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    # Run validation
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        #####
        break


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    mother_dir = opt['datasets']['test_1']['dataroot_lq']

    # callback function to process filenames each time a file is created
    def on_new_file_created(file_name):
        if 'processed' in file_name or 'Spectrum' in file_name or '.log' in file_name:
            print(f"'{file_name}'은 'processed' 또는 '.log' 를 포함하므로 후속 작업을 건너뜁니다.")
            return  # skip the rest and continue

        print(f"Newly created file: {file_name}")
        opt['datasets']['test_1']['dataroot_lq'] = os.path.join(opt['datasets']['test_1']['dataroot_lq'],
                                                                file_name)
        run_validation(opt)
        # Keep file monitoring after new file has been created
        opt['datasets']['test_1']['dataroot_lq'] = os.path.dirname(
            os.path.dirname(opt['datasets']['test_1']['dataroot_lq'])
        )
    # re-start monitoring    
    monitor_folder(mother_dir, on_new_file_created)

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)

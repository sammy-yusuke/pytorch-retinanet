import os
import json
from datetime import datetime
from tensorboardX import SummaryWriter
import socket

def create_training_output(task_root_path, comment=''):
    result_root_path = \
        os.path.expanduser(os.path.join(
            task_root_path, datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(result_root_path, exist_ok=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(os.path.join(task_root_path, "runs", current_time + '_' + socket.gethostname() + comment))
    tr = TrainingOutput(result_root_path, writer)
    return tr

class TrainingOutput(object):
    def __init__(self, result_root_path, writer):
        self.result_root_path = result_root_path
        self.experiment_configuration = None
        self.writer = writer

    def write(self):
        if self.experiment_configuration is not None:
            json.dump(self.experiment_configuration, open(os.path.join(self.result_root_path, "experiment_configuration.json")))
        self.writer.export_scalars_to_json(os.path.join(self.result_root_path, "all_scalars.json"))

    def read(self):
        pass
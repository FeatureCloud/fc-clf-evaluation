import os
import shutil
import threading
import time

import joblib
import jsonpickle
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error, mean_absolute_error, \
    mean_absolute_percentage_error, median_absolute_error
from sklearn.model_selection import train_test_split

from app.algo import roc_plot


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.y_test_filename = None
        self.y_pred_filename = None
        self.output_format = None
        self.y_test = None
        self.y_proba = None
        self.plt = None

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_roc']
            self.y_test_filename = config['files']['y_test']
            self.y_pred_filename = config['files']['y_pred']
            self.output_format = config['files']['output_format']

        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        print(f'Read config file.', flush=True)

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_plot = 3
        state_writing_results = 4
        state_finishing = 5

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:
            if state == state_initializing:
                print("[CLIENT] Initializing")
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input
                    print("[CLIENT] Coordinator", self.coordinator)

            if state == state_read_input:
                print('[CLIENT] Read input and config')
                self.read_config()
                self.y_test = pd.read_csv(self.y_pred_filename)
                self.y_proba = pd.read_csv(self.y_proba_filename)

                state = state_plot
            if state == state_plot:
                self.plt = roc_plot(self.y_test, self.y_proba)
            if state == state_writing_results:
                self.plt.savefig("roc_plot.png")
                state = state_finishing
            if state == state_finishing:
                print("[CLIENT] Finishing")
                self.progress = 'finishing...'
                if self.coordinator:
                    time.sleep(3)
                self.status_finished = True
                break

            time.sleep(1)


logic = AppLogic()

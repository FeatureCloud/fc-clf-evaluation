import pickle
import shutil
import threading
import time

import jsonpickle
import pandas as pd
import yaml

from app.algo import check, aggregate_confusion_matrices, create_score_df, compute_confusion_matrix


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
        self.y_test = None
        self.y_pred = None

        self.confusion_matrix_local = None
        self.confusion_matrix_global = None

        self.df = None
        self.score_df = None

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
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_class_evaluation']
            self.y_test_filename = config['files']['y_test']
            self.y_pred_filename = config['files']['y_pred']

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

        state_compute_confusion_matrix = 6
        state_aggregate_confusion_matrices = 7
        state_wait_for_confusion_matrices = 8
        state_compute_scores = 9
        state_writing_results = 10
        state_finishing = 11

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:

            # Local computations

            if state == state_initializing:
                print("[CLIENT] Initializing", flush=True)
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input
                    print("[CLIENT] Coordinator", flush=True)

            if state == state_read_input:
                print('[CLIENT] Read input and config', flush=True)
                self.read_config()
                if self.y_test_filename.endswith(".csv"):
                    self.y_test = pd.read_csv(self.INPUT_DIR + "/" + self.y_test_filename, sep=",")
                elif self.y_test_filename.endswith(".tsv"):
                    self.y_test = pd.read_csv(self.INPUT_DIR + "/" + self.y_test_filename, sep="\t")
                else:
                    self.y_test = pickle.load(self.INPUT_DIR + "/" + self.y_test_filename)

                if self.y_pred_filename.endswith(".csv"):
                    self.y_pred = pd.read_csv(self.INPUT_DIR + "/" + self.y_pred_filename, sep=",")
                elif self.y_pred_filename.endswith(".tsv"):
                    self.y_pred = pd.read_csv(self.INPUT_DIR + "/" + self.y_pred_filename, sep="\t")
                else:
                    self.y_pred = pickle.load(self.INPUT_DIR + "/" + self.y_pred_filename)
                self.y_test, self.y_pred = check(self.y_test, self.y_pred)
                state = state_compute_confusion_matrix

            if state == state_compute_confusion_matrix:
                self.confusion_matrix_local = compute_confusion_matrix(self.y_test, self.y_pred)

                data_to_send = jsonpickle.encode(self.confusion_matrix_local)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_aggregate_confusion_matrices
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_confusion_matrices
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_confusion_matrices:
                print("[CLIENT] Wait for confusion matrix", flush=True)
                self.progress = 'wait for confusion matrix'
                if len(self.data_incoming) > 0:
                    print("[CLIENT] Received aggregated confusion matrix from coordinator.", flush=True)
                    self.confusion_matrix_global = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []

                    state = state_compute_scores

            if state == state_compute_scores:
                print('[CLIENT] Compute scores parameters', flush=True)
                self.score_df = create_score_df(self.confusion_matrix_global)
                state = state_writing_results

            if state == state_writing_results:
                print('[CLIENT] Save results')
                self.score_df.to_csv(self.OUTPUT_DIR + "/scores.csv", index=False)

                if self.coordinator:
                    self.data_incoming = ['DONE']
                    state = state_finishing
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break

            if state == state_finishing:
                print("Finishing", flush=True)
                self.progress = 'finishing...'
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            # GLOBAL AGGREGATIONS

            if state == state_aggregate_confusion_matrices:
                print("[CLIENT] Wait for confusion matrices", flush=True)
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    print("[CLIENT] Aggregate confusion matrices", flush=True)
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    self.confusion_matrix_global = aggregate_confusion_matrices(data)
                    data_to_broadcast = jsonpickle.encode(self.confusion_matrix_global)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_compute_scores
                    print(f'[CLIENT] Broadcasting aggregated confusion matrix to clients', flush=True)

            time.sleep(1)


logic = AppLogic()

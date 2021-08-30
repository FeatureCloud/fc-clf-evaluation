import os
import pickle
import shutil
import threading
import time

import jsonpickle
import pandas as pd
import yaml

from app.algo import check, aggregate_confusion_matrices, create_score_df, compute_confusion_matrix, \
    create_cv_accumulation, plot_boxplots


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
        self.sep = ","
        self.mode = None
        self.dir = "."
        self.splits = {}

        self.confusion_matrices_local = {}
        self.confusion_matrices_global = {}

        self.score_dfs = {}
        self.cv_averages = None

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
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_classification_evaluation']
            self.y_test_filename = config['input']['y_true']
            self.y_pred_filename = config['input']['y_pred']
            self.sep = config['format']['sep']
            self.mode = config['split']['mode']
            self.dir = config['split']['dir']

        if self.mode == "directory":
            self.splits = dict.fromkeys([f.path for f in os.scandir(f'{self.INPUT_DIR}/{self.dir}') if f.is_dir()])
        else:
            self.splits[self.INPUT_DIR] = None

        for split in self.splits.keys():
            os.makedirs(split.replace("/input/", "/output/"), exist_ok=True)
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

                for split in self.splits.keys():
                    path = split + "/" + self.y_test_filename
                    if self.y_test_filename.endswith(".csv"):
                        y_test = pd.read_csv(path, sep=",")
                    elif self.y_test_filename.endswith(".tsv"):
                        y_test = pd.read_csv(path, sep="\t")
                    else:
                        y_test = pickle.load(path)
                    path = split + "/" + self.y_pred_filename
                    if self.y_pred_filename.endswith(".csv"):
                        y_pred = pd.read_csv(path, sep=",")
                    elif self.y_pred_filename.endswith(".tsv"):
                        y_pred = pd.read_csv(path, sep="\t")
                    else:
                        y_pred = pickle.load(path)
                    y_test, y_pred = check(y_test, y_pred)
                    self.splits[split] = [y_test, y_pred]
                state = state_compute_confusion_matrix

            if state == state_compute_confusion_matrix:
                for split in self.splits.keys():
                    y_test = self.splits[split][0]
                    y_pred = self.splits[split][1]
                    self.confusion_matrices_local[split] = compute_confusion_matrix(y_test, y_pred)

                data_to_send = jsonpickle.encode(self.confusion_matrices_local)

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
                    print("[CLIENT] Received aggregated confusion matrices from coordinator.", flush=True)
                    self.confusion_matrices_global = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []

                    state = state_compute_scores

            if state == state_compute_scores:
                print('[CLIENT] Compute scores parameters', flush=True)
                sens = []
                specs = []
                accs = []
                precs = []
                recs = []
                fs = []
                mccs = []

                for split in self.splits.keys():
                    self.score_dfs[split], data = create_score_df(self.confusion_matrices_global[split])
                    sens.append(data[0])
                    specs.append(data[1])
                    accs.append(data[2])
                    precs.append(data[3])
                    recs.append(data[4])
                    fs.append(data[5])
                    mccs.append(data[6])
                if len(self.splits.keys()) > 1:
                    self.cv_averages = create_cv_accumulation(accs, fs, mccs, precs, recs)

                state = state_writing_results

            if state == state_writing_results:
                print('[CLIENT] Save results')
                for split in self.splits.keys():
                    self.score_dfs[split].to_csv(split.replace("/input", "/output") + "/scores.csv", index=False)

                if len(self.splits.keys()) > 1:

                    self.cv_averages.to_csv(self.OUTPUT_DIR + "/cv_evaluation.csv", index=False)

                    print("[CLIENT] Plot images")
                    plt = plot_boxplots(self.cv_averages, title=f'{len(self.splits)}-fold Cross Validation')

                    for format in ["png", "svg", "pdf"]:
                        try:
                            plt.write_image(self.OUTPUT_DIR + "/boxplot." + format, format=format, engine="kaleido")
                        except Exception as e:
                            print("Could not save plot as " + format + ".")
                            print(e)

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
                    for split in self.splits.keys():
                        split_data = []
                        for client in data:
                            split_data.append(client[split])

                        self.confusion_matrices_global[split] = aggregate_confusion_matrices(split_data)
                    data_to_broadcast = jsonpickle.encode(self.confusion_matrices_global)
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    state = state_compute_scores
                    print(f'[CLIENT] Broadcasting aggregated confusion matrix to clients', flush=True)

            time.sleep(1)


logic = AppLogic()

import os
import pickle
import shutil
import threading
import time

import jsonpickle
import pandas as pd
import yaml

from FeatureCloud.app.engine.app import AppState, app_state, Role
from algo import check, aggregate_confusion_matrices, create_score_df, compute_confusion_matrix, \
    create_cv_accumulation, plot_boxplots


# Local computations
@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """

    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Initializing")
        if self.id is not None:  # Test if setup has happened already
            self.log(f"[CLIENT] Coordinator {self.is_coordinator}")
        
        return 'read input'


@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """

    def register(self):
        self.register_transition('compute confusion matrix', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Read input and config")
        self.read_config()

        splits = self.load('splits')
        for split in splits.keys():
            path = split + "/" + self.load('y_test_filename')
            if self.load('y_test_filename').endswith(".csv"):
                y_test = pd.read_csv(path, sep=",")
            elif self.load('y_test_filename').endswith(".tsv"):
                y_test = pd.read_csv(path, sep="\t")
            else:
                y_test = pickle.load(path)
            path = split + "/" + self.load('y_pred_filename')
            if self.load('y_pred_filename').endswith(".csv"):
                y_pred = pd.read_csv(path, sep=",")
            elif self.load('y_pred_filename').endswith(".tsv"):
                y_pred = pd.read_csv(path, sep="\t")
            else:
                y_pred = pickle.load(path)
            y_test, y_pred = check(y_test, y_pred)
            splits[split] = [y_test, y_pred]
        
        return 'compute confusion matrix'
        
    def read_config(self):
        self.store('INPUT_DIR', "/mnt/input")
        self.store('OUTPUT_DIR', "/mnt/output")
        self.store('sep', ",")
        self.store('dir', ".")
        self.store('confusion_matrices_local', {})
        self.store('confusion_matrices_global', {})
        self.store('score_dfs', {})
        splits = {}
        
        with open(self.load('INPUT_DIR') + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_classification_evaluation']
            self.store('y_test_filename', config['input']['y_true'])
            self.store('y_pred_filename', config['input']['y_pred'])
            self.store('sep', config['format']['sep'])
            self.store('mode', config['split']['mode'])
            self.store('dir', config['split']['dir'])

        if self.load('mode') == "directory":
            splits = dict.fromkeys([f.path for f in os.scandir(f"{self.load('INPUT_DIR')}/{self.load('dir')}") if f.is_dir()])
        else:
            splits[self.load('INPUT_DIR')] = None

        for split in splits.keys():
            os.makedirs(split.replace("/input/", "/output/"), exist_ok=True)
        
        shutil.copyfile(self.load('INPUT_DIR') + '/config.yml', self.load('OUTPUT_DIR') + '/config.yml')
        
        self.store('splits', splits)
        self.log(f'Read config file.')


@app_state('compute confusion matrix', Role.BOTH)
class ComputeConfusionMatrixState(AppState):
    """
    Compute local confusion matrix and send computation data to coordinator.
    """

    def register(self):
        self.register_transition('aggregate confusion matrices', Role.COORDINATOR)
        self.register_transition('wait for confusion matrices', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        splits = self.load('splits')
        confusion_matrices_local = self.load('confusion_matrices_local')
        for split in splits.keys():
            y_test = splits[split][0]
            y_pred = splits[split][1]
            confusion_matrices_local[split] = compute_confusion_matrix(y_test, y_pred)

        data_to_send = jsonpickle.encode(confusion_matrices_local)
        self.send_data_to_coordinator(data_to_send)
        self.log(f'[CLIENT] Sending computation data to coordinator')
        if self.is_coordinator:
            return 'aggregate confusion matrices'
        else:
            return 'wait for confusion matrices'


@app_state('wait for confusion matrices', Role.PARTICIPANT)
class WaitForConfusionMatrixState(AppState):
    """
    Wait for aggregated confusion matrices from coordinator.
    """

    def register(self):
        self.register_transition('compute scores', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Wait for confusion matrix")
        data = self.await_data()
        self.log("[CLIENT] Received aggregated confusion matrices from coordinator.")
        confusion_matrices_global = jsonpickle.decode(data)
        self.store('confusion_matrices_global', confusion_matrices_global)
        return 'compute scores'


@app_state('compute scores', Role.BOTH)
class ComputeScoresState(AppState):
    """
    Compute scores.
    """

    def register(self):
        self.register_transition('writing results', Role.BOTH)
        
    def run(self) -> str or None:
        self.log('[CLIENT] Compute scores parameters')
        sens = []
        specs = []
        accs = []
        precs = []
        recs = []
        fs = []
        mccs = []
        score_dfs = self.load('score_dfs')
        for split in self.load('splits').keys():
            score_dfs[split], data = create_score_df(self.load('confusion_matrices_global')[split])
            sens.append(data[0])
            specs.append(data[1])
            accs.append(data[2])
            precs.append(data[3])
            recs.append(data[4])
            fs.append(data[5])
            mccs.append(data[6])
        if len(self.load('splits').keys()) > 1:
            cv_averages = create_cv_accumulation(accs, fs, mccs, precs, recs)
            self.store('cv_averages', cv_averages)
            
        return 'writing results'


@app_state('writing results', Role.BOTH)
class WritingResultsState(AppState):
    """
    Write the results of the scores.
    """

    def register(self):
        self.register_transition('finishing', Role.COORDINATOR)
        self.register_transition('terminal', Role.PARTICIPANT)
        
    def run(self) -> str or None:
        self.log('[CLIENT] Save results')
        for split in self.load('splits').keys():
            self.load('score_dfs')[split].to_csv(split.replace("/input", "/output") + "/scores.csv", index=False)

        if len(self.load('splits').keys()) > 1:

            self.load('cv_averages').to_csv(self.load('OUTPUT_DIR') + "/cv_evaluation.csv", index=False)

            self.log("[CLIENT] Plot images")
            plt = plot_boxplots(self.load('cv_averages'), title=f"{len(self.load('splits'))}-fold Cross Validation")

            for format in ["png", "svg", "pdf"]:
                try:
                    plt.write_image(self.load('OUTPUT_DIR') + "/boxplot." + format, format=format, engine="kaleido")
                except Exception as e:
                    print("Could not save plot as " + format + ".")
                    print(e)
        self.send_data_to_coordinator('DONE')

        if self.is_coordinator:
            return 'finishing'
        else:
            return 'terminal'


@app_state('finishing', Role.COORDINATOR)
class FinishingState(AppState):

    def register(self):
        self.register_transition('terminal', Role.COORDINATOR)
        
    def run(self) -> str or None:
        self.gather_data()
        self.log("Finishing")
        return 'terminal'


# GLOBAL AGGREGATIONS
@app_state('aggregate confusion matrices', Role.COORDINATOR)
class AggregateConfusionMatricesState(AppState):
    """
    The coordinator receives the local confusion matrices from each client and aggregates the confusion matrices.
    The coordinator broadcasts the aggregated confusion matrices to the clients.
    """
    
    def register(self):
        self.register_transition('compute scores', Role.COORDINATOR)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Wait for confusion matrices")
        data_incoming = self.gather_data()
        self.log("[CLIENT] Aggregate confusion matrices")
        data = [jsonpickle.decode(client_data) for client_data in data_incoming]
        confusion_matrices_global = self.load('confusion_matrices_global')

        for split in self.load('splits').keys():
            split_data = []
            for client in data:
                split_data.append(client[split])
            confusion_matrices_global[split] = aggregate_confusion_matrices(split_data)
        data_to_broadcast = jsonpickle.encode(confusion_matrices_global)
        self.broadcast_data(data_to_broadcast, send_to_self=False)
        self.log(f'[CLIENT] Broadcasting aggregated confusion matrix to clients')

        return 'compute scores'

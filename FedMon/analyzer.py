import pandas as pd
import copy

class Analyzer:

    def __init__(self, trials = {
        "TensorFlow":{
                "trial_postfix_metrics":"-MNIST-tensorflow-FedAvg-false-10-config-A",
                "trial_postfix_profile":"-tensorflow-MNIST-10-config-A",
                "trial_prefix_profile": "server_metrics/new/profile/clients/",
                "trial_prefix_metrics": "server_metrics/metrics/new/client-",
                "trial_server_metrics": "server_metrics/new/profile/server/tensorflow-MNIST-FedAvg-false-10-config-A/"
            }
        }, number_of_clients=10):
        
        self.trials = trials
        self.number_of_clients = number_of_clients
        self.clients_profiles = {}
        self.clients_utilization_metrics = {}
        self.server_profiles = {}


    def create_structures(self):
        prefix = '/home/jovyan/work'
        number_of_clients = self.number_of_clients
        clients_profiles = {i:{} for i in self.trials}
        clients_utilization_metrics = {i:{} for i in self.trials}
        server_profiles = {i: {} for i in self.trials}
        for trial in self.trials:
            path = self.trials[trial]["trial_server_metrics"] + "profile.jl"
            profile_data = pd.read_json(path, lines=True)
            profile_data["round"] = profile_data.index
            
            path = self.trials[trial]["trial_server_metrics"] + "history_losses_distributed.jl"
            history_loss_data = pd.read_json(path, lines=True)[["round", "loss"]]
            server_profiles[trial] = profile_data.merge(history_loss_data, on='round')
            
            for client in range(number_of_clients):
                
                trial_postfix_metrics = self.trials[trial]["trial_postfix_metrics"]
                trial_postfix_profile = self.trials[trial]["trial_postfix_profile"]
                trial_prefix_metrics = self.trials[trial]["trial_prefix_metrics"]
                trial_prefix_profile = self.trials[trial]["trial_prefix_profile"]
                
                
                path = f"{trial_prefix_profile}{client}{trial_postfix_profile}/profile.jl"

                print(path)
                clients_profiles[trial][f"client_{client}"] = pd.read_json(path, lines=True)
                clients_profiles[trial][f"client_{client}"]['timestamp'] = pd.to_datetime(clients_profiles[trial][f"client_{client}"]['timestamp'], unit='ms')
                if "prev_accuracy" in clients_profiles[trial][f"client_{client}"].columns:
                    clients_profiles[trial][f"client_{client}"]["accuracy"] = clients_profiles[trial][f"client_{client}"]["prev_accuracy"] 
                    del clients_profiles[trial][f"client_{client}"]["prev_accuracy"]
                
                path = f"{trial_prefix_metrics}{client}{trial_postfix_metrics}.csv"

                print(path)
                util_metrics = pd.read_csv(path)
                    
                util_metrics = util_metrics[['network_rx_flnet', 'timestamp', 'memory', 'network_tx_flnet', 'cpu', 'count', 'memory_util']]
                util_metrics['network_rx'] = util_metrics['network_rx_flnet'].diff()
                util_metrics['network_tx'] = util_metrics['network_tx_flnet'].diff()
                util_metrics['cpu_time'] = util_metrics['cpu'].diff()
                util_metrics['timestamp'] = pd.to_datetime(util_metrics['timestamp'], unit='ms')
                util_metrics = util_metrics[(util_metrics['timestamp'] >= clients_profiles[trial][f"client_{client}"].timestamp.min()) & (util_metrics['timestamp'] < clients_profiles[trial][f"client_{client}"].timestamp.max())]
                

                clients_utilization_metrics[trial][f"client_{client}"] = util_metrics
        self.server_profiles = server_profiles
        self.clients_utilization_metrics = clients_utilization_metrics
        self.clients_profiles = clients_profiles

    def get_all_metrics(self):
        overall_data = {}
        out = None
        for trial in self.trials:
            for client in range(self.number_of_clients):
                df_util = self.clients_utilization_metrics[trial][f"client_{client}"]
                df_fl = self.clients_profiles[trial][f"client_{client}"]
                
                df_fl['s_timestamp'] = pd.to_datetime(df_fl['timestamp'], unit='ms').round('1s')
                df_fl["round"] = df_fl.index
                
                df_util['s_timestamp'] = pd.to_datetime(df_util['timestamp'], unit='ms').round('1s')
                df_util = df_util[(df_util["s_timestamp"] <= df_fl.s_timestamp.max()) & (df_util["s_timestamp"] >= df_fl.s_timestamp.min())]
                
                
                res = pd.merge(df_fl, df_util, on='s_timestamp', how = "outer")
                res = res.sort_values("s_timestamp").fillna(method = 'bfill').fillna(method = 'ffill')
                res["Node"] =  f"client_{client}"
                res["Trial"] = trial
                out = pd.concat([out, res])
        return out

    
        
    def get_overall_FL_metrics(self):
        return self.clients_profiles

    def get_overall_utilization_metrcs(self):
        return self.clients_utilization_metrics

    def get_concatenated_utilization_metrics(self):
        clients_utils = copy.deepcopy(self.clients_utilization_metrics)
        for trial in self.trials:
            for i in self.clients_utilization_metrics[trial]:
                clients_utils[trial][i]["Node"] =  i
                clients_utils[trial][i]["Trial"] =  trial
        return pd.concat([clients_utils[trial][i] for i in clients_utils[trial] for trial in self.trials])

    def get_concatenated_fl_metrics(self):
        for trial in self.trials:
            for i in self.clients_profiles[trial]:
                self.clients_profiles[trial][i]["Node"] = i
                self.clients_profiles[trial][i]["Trial"] = trial
        return pd.concat([self.clients_profiles[trial][i] for i in self.clients_profiles[trial] for trial in self.clients_profiles])




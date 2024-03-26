import os
import time

os.unsetenv("http_proxy")
os.unsetenv("https_proxy")
from FedMon.profiler import Profiler
import flwr as fl
# import torch
import json

# from clients.utils.profiler import Profiler, ClientsLogger
# from evaluation import get_eval_fn



# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

strategy_name = os.getenv("FL_STRATEGY")

number_of_rounds = int(os.getenv("FL_NUM_OF_ROUNDS", 3))

fraction_fit = float(os.getenv("FL_FRACTION_FIT", 0.1))
fraction_eval = float(os.getenv("FL_FRACTION_EVAL", 0.1))
min_eval_clients = int(os.getenv("FL_MIN_EVAL_CLIENTS", 2))
min_fit_clients = int(os.getenv("FL_MIN_FIT_CLIENTS", 2))
min_available_clients = int(os.getenv("FL_MIN_AVAILABLE_CLIENTS", 2))
strategy_extra_params = json.loads(os.getenv("FL_STRATEGY_EXTRA_PARAMS", "{}"))

dataset_name = os.getenv("FL_DATASET", 'CIFAR10')

eval_dataset = os.getenv("FL_EVAL_DATASET", 'false').lower() == 'true'

profiler_fit = Profiler("fit.jl", store_per_run=True)
profiler_fit_run = Profiler("fit_run.jl", store_per_run=True)

profiler_history_losses_centralized = Profiler("history_losses_centralized.jl")

fl.server.Server.fit = profiler_fit(fl.server.Server.fit)
fl.server.Server.fit_round = profiler_fit_run(fl.server.Server.fit_round)


if __name__ == "__main__":

    # get strategy from library based on user's preference, default is FedAvg
    # possible values are     "FastAndSlow", "FaultTolerantFedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgAndroid",
    #     "FedAvgM", "FedFSv0", "FedFSv1", "FedYogi", "QFedAvg"
    Strategy = fl.server.strategy.FedAvg
    if strategy_name in ["FastAndSlow", "FaultTolerantFedAvg", "FedAdagrad",
                         "FedAdam", "FedAvg", "FedAvgAndroid",
                         "FedAvgM", "FedFSv0", "FedFSv1", "FedYogi", "QFedAvg", "FedProx"]:
        Strategy = getattr(fl.server.strategy, strategy_name)

        # Strategy.configure_evaluate = clogger.log_selected_clients(Strategy.configure_evaluate)

    strategy = Strategy(
                            fraction_fit=fraction_fit,
                            fraction_evaluate=fraction_eval,
                            min_fit_clients=min_fit_clients,
                            min_evaluate_clients=min_eval_clients,
                            min_available_clients=min_available_clients,
                            # evaluate_fn=get_eval_fn() if eval_dataset else None,
                            **strategy_extra_params
                        )
    
    start_time = time.time()
    
    history = fl.server.start_server(server_address="[::]:8080",
                           config=fl.server.ServerConfig(**{"num_rounds": number_of_rounds, "round_timeout": 3600}), strategy=strategy
                           )
    
    end_time = time.time()

    profiler_fit.store_metrics()
    profiler_fit_run.store_metrics()


    for metric in history.losses_centralized:
        profiler_history_losses_centralized.log_metric("round", metric[0])
        profiler_history_losses_centralized.log_metric("loss", metric[1])
        profiler_history_losses_centralized.store_metrics("history_losses_centralized.jl")


    for metric in history.losses_distributed:
        profiler_history_losses_centralized.log_metric("round", metric[0])
        profiler_history_losses_centralized.log_metric("loss", metric[1])
        profiler_history_losses_centralized.store_metrics("history_losses_distributed.jl")


    for metric in history.metrics_centralized:
        for round in history.metrics_centralized[metric]:
            Profiler.log_metric("round", round[0])
            Profiler.log_metric(metric, round[1])
            Profiler.store_metrics("history_metrics_centralized.jl")

    print("history.losses_centralized: ", history.losses_centralized)
    print("history.metrics_centralized: ", history.metrics_centralized)
    print("history.losses_distributed: ", history.losses_distributed)
    print("history.metrics_distributed: ", history.metrics_distributed)
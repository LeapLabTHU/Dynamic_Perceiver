import numpy as np
import mlflow
from models.jeidnn.utils import get_abs_path

def get_display(key, cum_metric):
    if 'correct' in key:
            return  100*np.mean(cum_metric)
    else:
        return np.mean(cum_metric)

def we_want_to_see_it(metric_key):
    if 'pow' in metric_key:
        return False
    if 'list' in metric_key:
        return False
    if 'score' in metric_key:
        return False
    if 'cheating' in metric_key:
        return False
    return True

# here we change the name of the metric to be displayed
def key_to_name_display(metric_key):
    metric_name = metric_key.replace('correct', 'acc')
    metric_name = metric_name.replace('count_', '')
    metric_name = metric_name.replace('count', '')
    metric_name = metric_name.replace('per_gate', '')
    if metric_name[-1] == '_':
        metric_name = metric_name[:-1]
    return metric_name


def log_aggregate_metrics_mlflow(prefix_logger, metrics_dict, gates_count):
    log_dict = {}
    for metric_key, val in metrics_dict.items():
        if we_want_to_see_it(metric_key):
            metric_name_display = key_to_name_display(metric_key)
            cumul_metric, total = val
            if total > 0:
                if type(cumul_metric) is list: 
                    if (len(cumul_metric) == gates_count or len(cumul_metric) == gates_count+1) and 'per_gate' in metric_key:# if the length is the number of gates we want to see all of them
                        for g, cumul_metric_per_gate in enumerate(cumul_metric):
                            log_dict[prefix_logger+'/'+metric_name_display+ str(g)]  = get_display(metric_key, cumul_metric_per_gate)/total
                    else:
                        log_dict[prefix_logger+'/'+metric_name_display]  = get_display(metric_key, np.mean(cumul_metric))/total
                elif type(cumul_metric) is dict :
                    for g, cumul_metric_per_gate in cumul_metric.items():
                            log_dict[prefix_logger+'/'+metric_name_display+ str(g)]  = get_display(metric_key, cumul_metric_per_gate)/total
                else:
                    log_dict[prefix_logger+'/'+metric_name_display] = get_display(metric_key, cumul_metric)/total
    return log_dict



def setup_mlflow(run_name: str, cfg, experiment_name):
    print(run_name)
    project = experiment_name
    mlruns_path = get_abs_path(["mlruns"])
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(project)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(cfg)



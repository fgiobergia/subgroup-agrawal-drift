from drift_agrawal import SubgroupDriftAgrawal
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import argparse
from river.drift import binary as drift
from sklearn.model_selection import ParameterGrid
import json
import pickle
import os
from tqdm import tqdm


def to_Xy(ds, n : int = 10_000, drift_info : bool = False):
    X = []
    y = []
    sg_list = []
    drifted_list = []
    for vals in ds.take(n, drift_info=drift_info):
        if drift_info:
            x_val, y_val, sg, drifted = vals
        else:
            x_val, y_val = vals
        X.append(list(x_val.values()))
        y.append(y_val)
        if drift_info:
            sg_list.append(sg)
            drifted_list.append(drifted)
    
    if drift_info:
        return np.array(X), np.array(y), np.array(sg_list), np.array(drifted_list)
    return np.array(X), np.array(y)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--sg-size", type=float, default=0.1)
    argparser.add_argument("--perturbation", type=float, default=0.05)
    argparser.add_argument("--n-runs", type=int, default=10)
    argparser.add_argument("--models-file", type=str)

    args = argparser.parse_args()
    print(args)

    # set seed
    np.random.seed(args.seed)

    with open(args.models_file) as f:
        models = json.load(f)

    model_lookup = {
        "DDM": drift.DDM,
        "EDDM": drift.EDDM,
        "FHDDM": drift.FHDDM,
        "HDDM_A": drift.HDDM_A,
        "HDDM_W": drift.HDDM_W
    }

    train_size = 10_000
    batch_size = 1000
    test_batches=200
    bootstrap = 5 # batches where detection is disabled
    sg_size = args.sg_size
    perturbation = args.perturbation
    n_runs = 10
    outfile = f"results/sg_{sg_size}_pert_{perturbation}_nruns_{n_runs}.pkl"

    if os.path.isfile(outfile):
        print("File exists, skipping")
        exit()


    results = {}

    for model_name, params in models.items():
        print(model_name)
        model_func = model_lookup[model_name]

        for config in ParameterGrid(params):
            print(config)

            key = (model_name, str(config))
            results[key] = {
                "y_pred": [],
                "y_true": [],
            }
            with tqdm(range(n_runs)) as bar:
                for run in bar:
                    for with_drift in [True, False]:
                        ds = SubgroupDriftAgrawal(sg_size=sg_size if with_drift else 0.0,
                                                perturbation=perturbation,
                                                position=train_size + (test_batches // 2) * batch_size,
                                                width = (test_batches // 4) * batch_size)

                        X_train, y_train = to_Xy(ds, train_size)

                        clf = DecisionTreeClassifier(max_depth=3)
                        clf.fit(X_train, y_train)

                        detector = model_func(**config)

                        detections = []
                        pos = 0 # position in the stream
                        for batch in range(test_batches):
                            X_test, y_test, is_sg, _ = to_Xy(ds, batch_size, drift_info=True)
                            y_pred = clf.predict(X_test)

                            # 0 : correct
                            # 1 : error
                            for p in y_test != y_pred:
                                detector.update(p)

                                if batch > bootstrap and detector.drift_detected:
                                    detections.append(pos)
                                
                                pos += 1

                        results[key]["y_pred"].append(len(detections))
                        results[key]["y_true"].append(with_drift)
                        
                        # store json
                        with open(outfile, "wb") as f:
                            pickle.dump(results, f)

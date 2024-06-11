from drift_agrawal import SubgroupDriftAgrawal
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import argparse
from river.drift import binary as drift
from sklearn.model_selection import ParameterGrid
import json


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

    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("--seed", type=int, default=42)
    # argparser.add_argument("--sg-size", type=float, default=0.01)
    # argparser.add_argument("--perturbation", type=float, default=0.1)
    # args = argparser.parse_args()

    models = {
        "DDM": (drift.DDM, {
            "warm_start": [ 10, 50, 100, 500, 1000, 5000 ],
            "drift_threshold": [ 0.1, 0.5, 1, 3, 5, 10]
        }),
        "EDDM": (drift.EDDM, {
            "warm_start": [ 10, 50, 100, 500, 1000, 5000 ],
            "beta": np.arange(0.0, 1.1, 0.1),
            "alpha": [1.],
        }),
        "FHDDM": (drift.FHDDM, {
            "sliding_window_size": [10, 50, 100, 500, 1000, 5000],
            "confidence_level": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        }),
        "HDDM_A": (drift.HDDM_A, {
            "drift_confidence": [1e-4, 1e-3, 1e-2, 1e-1],
        }),
        "HDDM_W": (drift.HDDM_W, {
            "drift_confidence": [1e-4, 1e-3, 1e-2, 1e-1],
            "lambda_val": [ 0.01, 0.05, 0.1, 0.5 ],
        })
    }

    train_size = 10_000
    batch_size = 1000
    test_batches=200
    bootstrap = 5 # batches where detection is disabled
    sg_size = 0.1
    perturbation = 0.05
    n_runs = 10

    results = {}

    for model_name, (model_func, params) in models.items():
        print(model_name)

        for config in ParameterGrid(params):

            key = f"{model_name}|{config}"
            results[key] = {
                "y_pred": [],
                "y_true": [],
            }
            for run in range(n_runs):
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
                    with open(f"results.json", "w") as f:
                        json.dump(results, f)

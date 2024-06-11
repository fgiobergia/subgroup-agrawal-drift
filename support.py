from drift_agrawal import SubgroupDriftAgrawal
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import argparse
from river.drift import binary as drift
from sklearn.model_selection import ParameterGrid
import pickle
from run import to_Xy


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--perturbation", type=float, default=0.05)
    args = argparser.parse_args()

    # set seed
    np.random.seed(args.seed)

    models = {
        "DDM": (drift.DDM, {
            "warm_start": 5000,
            "drift_threshold": 5,
        }),
        "EDDM": (drift.EDDM, {
            "warm_start": 5000,
            "beta": 1.,
            "alpha": 1., 
        }),
        "FHDDM": (drift.FHDDM, {
            "sliding_window_size": 100,
            "confidence_level": 0.1, 
        }),
        "HDDM_A": (drift.HDDM_A, {
            "drift_confidence": 0.001,
        }),
        "HDDM_W": (drift.HDDM_W, {
            "drift_confidence": 0.1, 
            "lambda_val": 0.5, 
        })
    }

    train_size = 10_000
    batch_size = 1000
    test_batches=200
    bootstrap = 5 # batches where detection is disabled
    sg_size = args.sg_size
    perturbation = args.perturbation
    n_runs = 10

    results = {}

    for model_name, (model_func, config) in models.items():
        print(model_name)

        for sg_size in np.logspace(-2, 0, base=10):

            # key = f"{model_name}|{config}|{sg_size}"
            key = (model_name, sg_size)
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
                    with open(f"results.pkl", "w") as f:
                        pickle.dump(results, f)

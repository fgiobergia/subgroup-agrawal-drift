from river.datasets import synth
import numpy as np
class SubgroupAgrawalDrift:

    def pick_subgroup(self, target : float, tol : float = 1e-2, n_iter : int = 1_000, max_len : None | int = None):
        # identify a subgroup that has probability approx. the same as target

        # known features + range of values (start, stop, "value is in thousands")
        # (needed to return the correct ranges and avoid producing overly fine-grained subgroups)

        ranges = {
            "salary": [20, 150, True],
            "age": [20, 80, False],
            "elevel": [0, 4, False],
            "car": [1, 20, False],
            "zipcode": [0, 8, False],
            "hyears": [1,30, False],
            "loan": [0, 500, True],
        }

        curr = 1.0 # initial probability (support) -- all instances are in the subgroup
        taken = {}

        features = list(ranges.keys())
        np.random.shuffle(features) # features are shuffled to add some more randomness (not strictly necessary)

        while n_iter > 0:
            for feature in features:

                if len(taken) == max_len:
                    return taken
            
                if feature in taken:
                    continue
                    
                start, stop, is_k = ranges[feature]

                a, b = np.random.randint(start, stop + 1, size=2)
                if b < a:
                    a, b = b, a
                
                prob = (b-a) / (stop - start + 1)
                if is_k:
                    a *= 1_000
                    b *= 1_000

                if curr * prob >= target:
                    # can consider this feature (does not go below target probability)
                    # in this way, we guarantee that the extracted subgroup is always
                    # at least as big as requested (may be larger).

                    if abs(curr * prob - target) < abs(curr - target): # check whether we get closer to the target (greedy policy!)
                        # take!
                        curr *= prob
                        taken[feature] = (a, b)
                    
            if abs(curr - target) < tol:
                return taken, curr # converged (according to tol)
            n_iter -= 1
        return taken, curr # return the best subgroup found so far

    def __init__(self, sg_size : int = 0.1, perturbation : float = 0.0, position : int = 5000, width : int = 1000):
        self.ds = synth.Agrawal(perturbation=perturbation)
        self.sg_size = sg_size
        if self.sg_size > 0:
            self.sg, self.sg_size = self.pick_subgroup(sg_size)
        self.t = 0
        self.position = position
        self.width = width
    

    def take(self, n : int, drift_info : bool = False):
        for x, y in self.ds.take(n):
            drifted = False
            # should drift
            in_drifting_sg = all(a <= x[k] < b for k, (a, b) in self.sg.items()) if self.sg_size > 0 else False
            drift_proba = 1 / (1 + np.exp(-4 * (self.t - self.position) / self.width))
            if in_drifting_sg and np.random.random() < drift_proba:
                # should drift!
                y = 1 - y
                drifted = True
            self.t += 1

            if drift_info:
                yield x, y, in_drifting_sg, drifted
            else:
                yield x, y
from pystream.algorithms.base import VFDT, SVFDT, SVFDT_II
from pystream.utils import read_arff_meta, instance_gen
from pystream.evaluation import EvaluatePrequential
import logging
from pathlib import Path


def run():
    DEBUG = True
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    methods = ["entropy", "bgst_entropy", "budget_entropy"]
    log_dir = Path("log/")  # Change this if you want to save the logs somewhere else
    log_dir.mkdir(parents=True, exist_ok=True)

    z_values = [0.1, 0.2, 0.5, 0.9]
    z_val_str = z_values

    # Make sure all these files are in the datasets folder
    commands = [
        ("datasets/elecNormNew.csv"),
        ("datasets/hyper.csv"),
        ("datasets/sea.csv"),
        ("datasets/usenet.csv"),
        ("datasets/random_rbf_500k.csv"),
        ("datasets/covTypeNorm.csv"),
        ("datasets/led24_0_1kk.csv"),
        ("datasets/led24_10_1kk.csv"),
        ("datasets/led24_20_1kk.csv"),
        ("datasets/random_rbf_1kk.csv"),
        ("datasets/random_rbf_250k(50).csv"),
        ("datasets/poker-lsn.csv"),
        ("datasets/airlines_parsed.csv"),
        ("datasets/CTU_1.csv"),
        ("datasets/CTU_2.csv"),
        ("datasets/CTU_3.csv"),
        ("datasets/CTU_4.csv"),
        ("datasets/CTU_5.csv"),
        ("datasets/CTU_6.csv"),
        ("datasets/CTU_7.csv"),
        ("datasets/CTU_8.csv"),
        ("datasets/CTU_9.csv"),
        ("datasets/CTU_10.csv"),
        ("datasets/CTU_11.csv"),
        ("datasets/CTU_12.csv"),
        ("datasets/CTU_13.csv"),
    ]
    for fname in commands:
        dataset_name = fname.split("/")[-1].split(".csv")[0]
        meta_file = f"datasets/metas/{dataset_name}.meta"
        dtype, types, classes = read_arff_meta(meta_file)
        n_classes = len(classes)
        only_binary_splits = False
        base_learners_n_args = [
            (
                "vfdt",
                VFDT,
                {
                    "gp": 100,
                    "split_criterion": "infogain",
                    "tiebreaker": 0.05,
                    "only_binary_splits": only_binary_splits,
                },
            ),
            (
                "svfdt",
                SVFDT,
                {
                    "gp": 100,
                    "split_criterion": "infogain",
                    "tiebreaker": 0.05,
                    "only_binary_splits": only_binary_splits,
                },
            ),
            (
                "svfdt_ii",
                SVFDT_II,
                {
                    "gp": 400,
                    "split_criterion": "infogain",
                    "tiebreaker": 0.05,
                    "only_binary_splits": only_binary_splits,
                },
            ),
        ]
        for method in methods:
            for name, base_learner, kwargs in base_learners_n_args:
                csv_results = []
                for z, confidence in zip(z_values, z_val_str):
                    algorithm = base_learner(types, n_classes, **kwargs)
                    log_file = log_dir / f"{dataset_name}_{confidence}.csv"
                    evaluator = EvaluatePrequential(
                        n_classes, algorithm, algorithm_type="tree"
                    )
                    stream = instance_gen(fname, dtype)  # chunksize=500000

                    evaluator.train_test_prequential_entropy(
                        stream,
                        DEBUG,
                        10,
                        log_file=log_file,
                        active=True,
                        z=z,
                        method=method,
                    )

                    csv_results.append(
                        {
                            "Z Value": str(confidence),
                            "Accuracy": evaluator.stats.accuracy,
                            "Hits": evaluator.stats["hits"],
                            "Miss": evaluator.stats["miss"],
                            "Queried": evaluator.stats["train_truelabel"],
                        }
                    )

                with open(log_dir / f"{method}_{dataset_name}_{name}.csv", "w") as f:
                    f.write("Z Value,Accuracy,Queries\n")
                    for result in csv_results:
                        f.write(
                            (
                                f"{result['Z Value']},"
                                f"{result['Accuracy']},"
                                f"{result['Queried']}\n"
                            )
                        )


if __name__ == "__main__":
    run()

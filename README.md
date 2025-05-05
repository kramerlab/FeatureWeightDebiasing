# Feature-weighted maximum representative sampling (FW-MRS)

## Install Requirements
All requirements are listed in the requirements.txt file and can be installed via pip:
```shell
python3 -m pip install -r requirements.txt
```

## Reproduce Results
Run the scripts individually:
```shell
./downstream_task_experiment.sh
./decomposition_experiments.sh
./temperature_comparison.sh
```

or all together:
```shell
./all_experiments.sh
```

## Reproduce Diagrams
To reproduce the diagrams the Jupyter notebooks are in the directory "/notebooks".

## Feature-weighted Decision Trees

We cloned the source code of scikit-learn to add the feature weights to the decision tree implementation because we wanted to use the advantages of scikit-learn (e.g.parallelization and runtime performance). To make it easier to inspect our implementation we extracted the changed source code to the file src/feature_weighted_tree/extracted_source_code.
# RoughSetsConsensusClustering

## Running matrix.py

## Prerequisites

> Ensure you have Python 3 installed on your system. You can download it from [here](https://www.python.org/downloads/).

## Setup

1. Clone the repository to your local machine.

2. Navigate to the project directory.

```sh
python3 -m pip install virtualenv
virtualenv envCDLib
```

3. Activate the Python virtual environment.

```sh
source envCDLib/bin/activate
```

4. Install the required Python packages.

```sh
pip install -r cdlib_requirements.txt
```

## Running the Script

Run `matrix.py` using the following command:

```sh
python3 src/matrix.py
```

## Into matrix.py

### Main Steps

1. **Set the Folder Version**: Specify the folder version for the networks. Options are `'NetsType_1.4'` or `'NetsType_1.6'`.

```python
folder_version = 'NetsType_1.4'
```

2. **Run Rough Clustering Algorithm**: Execute the Rough Clustering algorithm for all networks in the specified folder.

```python
runRoughClusteringCDLib(m = Matrix([], {}, []), folder_version = folder_version, gamma=0.8)
```

3. **Evaluate Overlapping Normalized Mutual Information (NMI)**: Assess the overlapping NMI for the Rough Clustering algorithm for all networks in the specified folder.

```python
nmi_overlapping_evaluate(folder_version)
```

4. **Calculate Core Nodes Percentage in Ground Truth (GT) Communities**: Determine the percentage of core nodes that are in the GT communities. The result is saved in `RC_cores_percent.txt`.

```python
compare_cores_with_GT_simple(folder_version)
```

5. **Calculate Partition Coefficient for Rough Clustering**: Compute the Partition Coefficient for the Rough Clustering algorithm for all networks in the specified folder.

```python
apply_PC_to_RC(folder_version, overlap=True)
```

6. **Calculate Partition Coefficient for Ground Truth Communities**: Compute the Partition Coefficient for the Ground Truth communities for all networks in the specified folder.

```python
apply_PC_to_GT(folder_version, overlap=True)
```

7. **Export Overlapping Nodes Scores**: Export the scores of overlapping nodes by the Partition Coefficient for both the Rough Clustering algorithm and the Ground Truth.

```python
export_pc_overlaping_nodes_gt(folder_version)
```

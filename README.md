# Placeit!

## Before starting

### Requirement

* python 3.10

### Recommandations

* larger number of cpu cores => faster exploration



## Install
Create a virtual env, and source it:
```
python3.10 -m venv qdp
source qdp/bin/activate
```
Launch the installer:
```
./launchers/installer.sh
```



## Launch
Always make sure to be in the virtual env:
```
source qdp/bin/activate
```

Then run one of the following examples:

### PLace generation

Display mode, to visualize each evaluation in a sequential run: 
```
python3 run_qd_place.py -a cma_mae -nbr 2000 -p 20 -o ycb_mug -t table -ii -d
```

Longer run with parallelization:
```
python3 run_qd_place.py -a cma_mae -nbr 2000 -p 20 -o ycb_mug -t table -ii -ll
```

Run with different z initial positions:
```
python3 run_qd_place.py -a cma_mae -nbr 2000 -p 20 -o ycb_mug -t table -ii -ll -n 3
```



### Visualizing output

To replay poses from a completed run:
```
python3 visualization/replay_poses.py -r path_to_run_folder/
```
Shuffle places and replay poses from a completed run:
```
python3 visualization/replay_poses.py -r path_to_run_folder/ -si







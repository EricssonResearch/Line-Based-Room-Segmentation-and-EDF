Home of Wiki for Erik's master thesis project.

## Description
This project implements a method for real-time construction of continuous Euclidean distance fields (EDF), specifically for large indoor environments, using Gaussian process (GP) regression. The proposed approach focuses on leveraging the inherent structural information of indoor spaces by partitioning them into rooms and constructing a local GP-EDF model for each room. This approach significantly reduces the computational cost associated with large matrix operations in GPs. The method also exploits the geometric regularities commonly found in indoor spaces by detecting walls and representing them as line segments. This information is integrated into the models' priors to both improve accuracy and further reduce computational expense. 

The method is divided into three modules:

1. **Line Segment Detection**: This module identifies and represents the walls within an indoor environment as line segments. The algorithm operates sequentially, continually updating the line segments as new sensor data is acquired.

2. **Room Segmentation**: This module utilizes the detected line segments to partition the indoor environment into separate rooms. It first processes the raw line segments, then constructs a visibility graph with the line segments as nodes, and finally performs graph clustering to identify individual rooms.

3. **Room-based GP-EDF**: This module creates a local GP-EDF model for each detected room. It incorporates the line segments into the prior mean functions and represents residual measurements with inducing points.

## Simulation Environments

The project uses two different indoor environments, simulated in Gazebo, for testing and validation and a mobile robot, Turtlebot3 Waffle, equipped with a 2D laser range sensor, was steered through these environments to collect data.

See more in the wiki!

## Repository Structure
- `data/`: contains pickle files of sequences of sensor readings collected by the Turtlebot3_waffle in the two different indoor environments. The robot traversed multiple routes through these environments under varying levels of sensor noise.
- `models/`: Contains the code for the three modules of the thesis implementation: line segment detection, room segmentation, and room-based GP-EDF.
- `main.py`: Can be used to run and visualize all the models. All parameters for the models can be set through this script.
- `gpedf_experiments.py`: Used to compare the room-based GP-EDF model with two baselines: the "standard global model" that constructs a single global GP-EDF using a zero prior mean function, and the "line-based global model" that constructs a single global GP-EDF using the line segment-based prior mean function.

## Installation
The project can be installed by running the following command in your terminal:
```bash
pip install -r requirements.txt
```

## Expected Outcome
See wiki!


## Common Issues

If you encounter an error saying that the required dependency `pytz` is missing when trying to import pandas, you can fix this by running the following command in your terminal:

```bash
python3 -m pip install pytz
```

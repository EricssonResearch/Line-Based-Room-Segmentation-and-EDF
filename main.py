import numpy as np
import networkx as nx
import pandas as pd

from shapely.geometry import MultiPoint
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import time

from models.room_based_gpedf import RoomBasedGPEDF
from models.line_segment_detection import LineSegDetection
from models.room_segmentation import RoomSegmentation


def get_linesegs_within_range(le, robot_pos, range):
    """Determine line segments within range from the robot

    Args:
        le (LineSegDetection): line segment detection object
        robot_pos (array): current robot position
        range (float): range of sensor

    Returns:
        list: line segments within range
    """
    lines = list(le.line_graph.nodes())
    endpoints = np.array([le.line_graph.nodes[line]["line"].endpoints for line in lines])
    a = endpoints[:, 0, :] 
    b = endpoints[:, 1, :] 
    ab = b - a  
    ab2 = np.sum(ab**2, axis=-1) 
    ap = robot_pos - a  
    abap = np.sum(ab * ap, axis=-1)  
    t = np.clip(abap / ab2, 0.0, 1.0) 
    pt = ap - t[:, np.newaxis] * ab  
    pt2 = np.sum(pt**2, axis=-1)
    within_range = list(np.array(lines)[pt2 <= range**2])
    return within_range


def live_plot(ax, gpedf, rs, le, model_name):
    """Live plot of the selected module

    Args:
        ax (matplotlib AxesSubplot): matplotlib ax
        gpedf (RoomBasedGPEDF): room-based GP-EDF object
        rs (RoomSegmentation): room segmentation object
        le (LineSegDetection): line segment detection object
        model_name (string): name of model to plot
    """
    ax.clear()
    _colors = ["blue", "red", "green", "pink", "orange", "lime", "cyan", "yellow", "purple", "brown"]
    #_colors = list(colors.cnames.keys())
    
    if model_name == "Line Segment Detection":
        points_array = np.array(list(le.points.values()))
        ax.scatter(points_array[:,0], points_array[:,1], c='black', s=5, alpha=0.5)
        endpoints = [le.line_graph.nodes[p]["line"].endpoints for p in le.line_graph.nodes()]  
        lc = LineCollection(endpoints, colors='red', linewidths=2.0)
        ax.add_collection(lc)
    
    elif model_name == "Room Segmentation":
        col = [_colors[rs.le.line_graph.nodes[l]["label"]] for l in rs.le.line_graph.nodes()]
        points_array = np.array(list(rs.le.points.values()))
        ax.scatter(points_array[:,0], points_array[:,1], c='black', alpha=0.5)
        endpoints = [rs.le.line_graph.nodes[p]["line"].endpoints for p in rs.le.line_graph.nodes()]  
        lc = LineCollection(endpoints, colors=col, linewidths=3.0)
        ax.add_collection(lc)
        nx.draw(rs.room_graph, pos=nx.get_node_attributes(rs.room_graph, 'pos'), node_size=30, node_color="brown", edge_color="purple")
        ax.scatter(rs.le.robot_pos[0], rs.le.robot_pos[1], c='purple',marker='o',s=100)
    
    elif model_name == "Room-based GP-EDF":
        col = [_colors[gpedf.rs.le.line_graph.nodes[l]["label"]] for l in gpedf.rs.le.line_graph.nodes()]
        np_points = np.array(list(gpedf.rs.le.points.values()))
        ax.scatter(np_points[:,0], np_points[:,1], c='black', alpha=0.5)
        endpoints = [gpedf.rs.le.line_graph.nodes[p]["line"].endpoints for p in gpedf.rs.le.line_graph.nodes()]  
        lc = LineCollection(endpoints, colors=col, linewidths=2.5)
        ax.add_collection(lc)
        nx.draw(gpedf.rs.room_graph, pos=nx.get_node_attributes(gpedf.rs.room_graph, 'pos'), node_size=30, node_color="brown", edge_color="purple")
        if gpedf.rs.room_graph.nodes[gpedf.rs.current_room]["rect"] is None:
            room_bounds = MultiPoint(list(gpedf.rs.le.points.values())).bounds
        else:
            room_bounds = gpedf.rs.room_graph.nodes[gpedf.rs.current_room]["rect"]
        inflation = 1
        x_grid = np.linspace(room_bounds[0]-inflation, room_bounds[2]+inflation, 30)
        y_grid = np.linspace(room_bounds[1]-inflation, room_bounds[3]+inflation, 30)
        X_test = np.meshgrid(x_grid, y_grid)
        test = np.column_stack((np.ravel(X_test[0]), np.ravel(X_test[1])))
        mean = gpedf.get_dist(test, gpedf.rs.current_room)
        CS = ax.contour(X_test[0] , X_test[1], np.reshape(mean, (X_test[0].shape[0], X_test[1].shape[0])), np.linspace(0.25, 2.25, 9))
        ax.clabel(CS, inline=1, fontsize=8)
        ax.scatter(gpedf.rs.le.robot_pos[0], gpedf.rs.le.robot_pos[1], c='red',edgecolor='red',marker='o',s=80)
        #inducing_points = gpedf.rs.room_graph.nodes[gpedf.rs.current_room]["GP"].gp.variational_strategy.inducing_points.detach()
        #ax.scatter(inducing_points[:,0], inducing_points[:,1], c='pink', s=40, marker="+")

    ax.axis('square')
    plt.pause(0.05)


def scan_to_coords(scan, max_sensor_range, std_dev=0.0):
    """Transforms a sensor reading to a sequence of rectangular coordinates.

    Args:
        scan (dict): dictionary of a sensor range reading
        max_sensor_range (float): maximum range of the sensor
        std_dev (float, optional): standard variance of additive Gaussian noise

    Returns:
        sensor data
    """
    robot_yaw = scan['robot_pose'][2] 
    robot_pos = scan['robot_pose'][0:2] 
    scan_yaw = scan['angle_min'] + robot_yaw
    points = []
    angles = [] 
    ranges = [] 
    delta_theta = scan['angle_increment']
    for range in scan['ranges']:
        if scan['range_min'] < range < scan['range_max'] and range < max_sensor_range and range > 0.2:
            noise = np.random.normal(0, std_dev)
            range = range + noise
            x_scan = range*np.cos(scan_yaw)
            y_scan = range*np.sin(scan_yaw)
            
            x_map = robot_pos[0] + x_scan 
            y_map = robot_pos[1] + y_scan

            points.append(np.array([x_map, y_map]))
            angles.append(scan_yaw)
            ranges.append(range)
            
        scan_yaw += delta_theta
    return points, angles, ranges, delta_theta, np.array(robot_pos)


def main():
    """ Available model names:
        * Room-based GP-EDF
        * Room Segmentation
        * Line Segment Detection
    """
    
    model_name = "Room-based GP-EDF" 
    file_name = 'scans_simple_hospital_route1.pkl'  # File name of the collected sensor readings

    with open("data/"+file_name, 'rb') as f:
        scans = pickle.load(f) #[0:300]
        
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(1, 1, 1)
    
    # Model parameters
    
    max_sensor_range = 3.0  # Maximum range of sensor
    
    k = 7  # Magnification factor for adaptive breakpoint detection
    least_thresh = 0.05 # Maximum orthogonal distance between points at fitted line
    min_line_length = 0.5  # Minimum length of a line segment
    min_line_points = 20  # Minimum number of points used to fit a full line segment
    line_seed_points = 20  # Number of points used to fit a line seed
    corner_thresh = 0.5  # Distance threshold to form a corner between two line segments
    door_width_interval = [0.8, 3.0]  # Interval of potential doorway widths
    gap_thresh = 0.5  # Distance threshold of maximum gap width to join two collinear line segments
    delta_d = 0.2  # Maximum orthogonal distance to mergetwo line segments
    le = LineSegDetection(k, least_thresh, min_line_length, min_line_points, 
                           line_seed_points, corner_thresh, door_width_interval, 
                           gap_thresh, delta_d)
    
    gamma_dist = 0.02  # Tuning parameter for the distance-based edge weight
    gamma_rob = 0.005  # Tuning parameter for the robot position-based edge weight
    fiedler_thresh = 0.18  # Threshold for the Fiedler value
    vis_dist_thresh = 8.0  # Maximum distance for creating an edge between two line segments
    remerge_thresh = 0.5  # Threshold for remerging a room that has been split
    rs = RoomSegmentation(le, max_sensor_range, gamma_dist, gamma_rob, 
                        fiedler_thresh, vis_dist_thresh, remerge_thresh)
    
    lamb = 100  # Characteristic length scale
    inducing_thresh = 0.5e-6  # threshold for adaptive inducing point selection
    gpedf = RoomBasedGPEDF(rs, lamb, inducing_thresh)
    
    if model_name == "Line Segment Detection":
        room_graph = nx.Graph()
        room_graph.add_node(0, lines=[], pos=None, rect=None, GP=None)
    
    tot_points = []
    last_robot_pos = None
    num_points = 0
    num_points_list = []
    durations = []
    for i, scan in enumerate(scans): 
        print("****************** scan",i,"******************")
        points, angles, ranges, delta_theta, robot_pos = scan_to_coords(scan, max_sensor_range=3.0)
        tot_points.extend(points)
        if last_robot_pos is not None and np.linalg.norm(robot_pos-last_robot_pos) < 0.2:
            continue 
        last_robot_pos = robot_pos
        
        if model_name == "Line Segment Detection":
            start = time.time()
            if le.line_graph.number_of_nodes() > 0:
                linesegs_within_range = get_linesegs_within_range(le, robot_pos, max_sensor_range+0.2)
            else:
                linesegs_within_range = []
            le.update(points, angles, ranges, robot_pos, 0, linesegs_within_range, delta_theta, room_graph, room_processing=False)
            end = time.time()
            print("Line segment detection time:",end-start)
            
        elif model_name == "Room Segmentation":
            start = time.time()
            rs.update(robot_pos, points, angles, ranges, delta_theta)
            end = time.time()
            print("Room segmentation time:",end-start)
            num_points += len(points)
            if rs.le.isUpdated:
                num_points_list.append(num_points)
                durations.append(end-start)
            
        elif model_name == "Room-based GP-EDF":
            start = time.time()
            gpedf.update(robot_pos, points, angles, ranges, delta_theta)
            end = time.time()
            print("GP-EDF update time:",end-start)
            num_points += len(points)
            if gpedf.rs.le.isUpdated:
                num_points_list.append(num_points)
                durations.append(end-start)
            
        if i > 5:
            live_plot(ax, gpedf, rs, le, model_name)
    
    if model_name == "Room Segmentation" or model_name == "Room-based GP-EDF":
        plt.figure()
        df = pd.DataFrame({
                'x': num_points_list,
                'y': durations
                })
        df['smoothed_y'] = df['y'].ewm(span=50).mean()
        plt.plot(df['x'], df['smoothed_y'])
        plt.title(model_name + " Computation Time")
        plt.xlabel("Number of data points")
        plt.ylabel("Computation time (seconds)")   
        
    plt.show()

    
    
    
if __name__ == '__main__':
    main()
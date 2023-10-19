import matplotlib.pyplot as plt
import numpy as np
import torch
import gpytorch
from models.room_based_gpedf import RoomBasedGPEDF
from gpedf_utils.online_sgpr import OnlineSGPRegression as osgpr
from models.line_segment_detection import LineSegDetection
from models.room_segmentation import RoomSegmentation
import networkx as nx
from gpedf_utils.streaming_sgpr import LineSegMean
import time
import open3d as o3d
import pickle
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib import patches


torch.set_default_dtype(torch.double)

class LineGPEDF:
    """ Class for the line segment-based global GP-EDF
    """
    
    def __init__(self, le, lamb=100, max_sensor_range=3.0, inducing_thresh=0.5e-6):
        self.le = le
        self.lamb = lamb
        self.room_graph = nx.Graph()
        self.room_graph.add_node(0, lines=[], pos=None, rect=None, GP=None)
        self.model = self.init_model(inducing_thresh)
        self.last_robot_pos = None
        self.max_sensor_range = max_sensor_range
        self.radius = self.max_sensor_range + 0.2
        self.pcd = o3d.open3d.geometry.PointCloud()
        
    def init_model(self, inducing_thresh):
        inducing_points = torch.empty((0,2)) 
        covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
        covar_module.lengthscale = 1/self.lamb
        mean_module = LineSegMean(0, self.room_graph, self.le.line_graph)
        model = osgpr(covar_module=covar_module, mean_module=mean_module,
                            inducing_points=inducing_points, 
                            learn_inducing_locations=True, 
                            jitter=1e-6, inducing_thresh=inducing_thresh)
        return model
    
    def get_dist(self, X):
        X = torch.from_numpy(np.atleast_2d(X))
        mean = self.model.predict(X)
        EDF_mean = - (1/self.lamb) * torch.log(mean)
        return EDF_mean.detach()

    def get_target_value(self, X):
        dist = torch.zeros((X.shape[0], 1))
        phi = torch.exp(-dist*self.lamb) 
        return phi

    def downsample(self, x, r_voxel=0.15):
        zeros = np.zeros((x.shape[0],1))
        XYZ = np.concatenate((x, zeros), 1)
        self.pcd.points = o3d.utility.Vector3dVector(XYZ)
        downXY = np.asarray(self.pcd.voxel_down_sample(voxel_size=r_voxel).points)[:,0:2]
        return downXY
    
    def linesegs_within_range(self, robot_pos):
        line_keys = self.le.line_graph.nodes()
        endpoints = np.array([self.le.line_graph.nodes[line]["line"].endpoints for line in line_keys])
        a = endpoints[:, 0, :] 
        b = endpoints[:, 1, :] 
        ab = b - a  
        ab2 = np.sum(ab**2, axis=-1) 
        ap = robot_pos - a  
        abap = np.sum(ab * ap, axis=-1)  
        t = np.clip(abap / ab2, 0.0, 1.0) 
        pt = ap - t[:, np.newaxis] * ab  
        pt2 = np.sum(pt**2, axis=-1)
        within_range = list(np.array(line_keys)[pt2 <= self.radius**2])
        return within_range
    
    def update(self, robot_pos, points, angles, ranges, delta_theta):
        self.last_robot_pos = robot_pos
        
        within_range = self.linesegs_within_range(robot_pos) if self.le.line_graph.number_of_nodes() > 0 else []
        
        self.le.update(points, angles, ranges, robot_pos, 0, within_range, delta_theta, self.room_graph, line_processing=False)

        if len(self.le.new_remaining_points) > 0:
            new_points = np.array(list(self.le.new_remaining_points.values()))
            down_points = torch.from_numpy(self.downsample(new_points))
            target = self.get_target_value(down_points)
            self.model.update(down_points, target) 
            return True
        return False
        
        
class StandardGPEDF:
    """ Class for the standard global GP-EDF
    """
    def __init__(self, lamb=100, max_sensor_range=3.0, inducing_thresh=0.5e-6):
        self.lamb = lamb
        self.model = self.init_model(inducing_thresh)
        self.last_robot_pos = None
        self.max_sensor_range = max_sensor_range
        self.pcd = o3d.open3d.geometry.PointCloud()
        self.input_points = None
        
    def init_model(self, inducing_thresh):
        inducing_points = torch.empty((0,2)) 
        covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
        covar_module.lengthscale = 1/self.lamb
        mean_module = gpytorch.means.ZeroMean()
        model = osgpr(covar_module=covar_module, mean_module=mean_module,
                            inducing_points=inducing_points, 
                            learn_inducing_locations=True, 
                            jitter=1e-6, inducing_thresh=inducing_thresh)
        return model
    
    def get_dist(self, X):
        X = torch.from_numpy(np.atleast_2d(X))
        mean = self.model.predict(X)
        EDF_mean = - (1/self.lamb) * torch.log(mean)
        return EDF_mean.detach()

    def get_target_value(self, X):
        dist = torch.zeros((X.shape[0], 1))
        phi = torch.exp(-dist*self.lamb) 
        return phi

    def downsample(self, x, r_voxel=0.3):
        zeros = np.zeros((x.shape[0],1))
        XYZ = np.concatenate((x, zeros), 1)
        self.pcd.points = o3d.utility.Vector3dVector(XYZ)
        downXY = np.asarray(self.pcd.voxel_down_sample(voxel_size=r_voxel).points)[:,0:2]
        self.input_points = downXY
        return downXY
    
    def update(self, robot_pos, points):
        # if self.last_robot_pos is not None and np.linalg.norm(robot_pos-self.last_robot_pos) < 0.3:
        #     return 
        self.last_robot_pos = robot_pos

        down_points = torch.from_numpy(self.downsample(points))
        target = self.get_target_value(down_points)
        self.model.update(down_points, target) 
    
        

def init_models(lamb=100, max_sensor_range=3.0, inducing_thresh=0.5e-6):
    # Room based local GP-EDF model
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
    
    room_gpedf = RoomBasedGPEDF(rs, lamb, inducing_thresh)
    
    # Standard global GP-EDF model
    standard_gpedf = StandardGPEDF(lamb, max_sensor_range, inducing_thresh)

    # Line Segment based global GP-EDF model
    le = LineSegDetection(k, least_thresh, min_line_length, min_line_points, 
                           line_seed_points, corner_thresh, door_width_interval, 
                           gap_thresh, delta_d)
    line_gpedf = LineGPEDF(le, lamb, max_sensor_range, inducing_thresh)
    
    return room_gpedf, standard_gpedf, line_gpedf


def visualize(SIM_MODELS, points, room_gpedf, standard_gpedf, line_gpedf, line_of_points):
    x_grid = np.linspace(-0.9, 3.6, 30)
    y_grid = np.linspace(0.0, 6.6,  30)
    X_test = np.meshgrid(x_grid, y_grid)
    test = np.column_stack((np.ravel(X_test[0]), np.ravel(X_test[1])))
    plot_dists = [None, None, None]
    if SIM_MODELS[0]:
        fig = plt.figure(figsize=(6.5, 16))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) # change the height ratio here
        fig.suptitle('Room-based model', fontsize=10)
        
        ax = plt.subplot(gs[0])
        ax.scatter(points[:,0], points[:,1], c='gray', alpha=0.5)
        mean_pred = np.zeros(len(test))
        for i, point in enumerate(test):
            room = room_gpedf.rs.identify_room(list(point), visualize_GP=True)
            if type(room) == list:
                mean_pred[i] = min([room_gpedf.get_dist(point, label) for label in room])
            else:
                mean_pred[i] = room_gpedf.get_dist(point, room)
        CS = ax.contour(X_test[0] , X_test[1], np.reshape(mean_pred, (X_test[0].shape[0], X_test[1].shape[0])), np.linspace(0.25, 2.25, 9))
        ax.clabel(CS, inline=1, fontsize=8)
        ax.axis("square")
        
        plot_dists_room = np.zeros(len(line_of_points))
        for i, point in enumerate(line_of_points):
            room = room_gpedf.rs.identify_room(list(point), visualize_GP=True)
            if type(room) == list:
                plot_dists_room[i] = min([room_gpedf.get_dist(point, label) for label in room])
            else:
                plot_dists_room[i] = room_gpedf.get_dist(point, room)
        plot_dists[0] = plot_dists_room
        ax.scatter(line_of_points[:,0], line_of_points[:,1], c='red', s=20)
        ax.set_xlim(-10,10)
        ax.set_ylim(-7.7,7.6)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)

        ax_line = plt.subplot(gs[1], sharex=ax)
        ax_line.plot(line_of_points[:,0],plot_dists_room, "blue")
        ax_line.xaxis.set_visible(False) 
        ax_line.set_xlim(-10,10)
        
        xyA1 = (line_of_points[0,0], line_of_points[0,1])
        xyB1 = (line_of_points[0,0], plot_dists_room[0])
        arrow1 = patches.ConnectionPatch(
            xyA1,
            xyB1,
            coordsA=ax.transData,
            coordsB=ax_line.transData,
            color="black",
            arrowstyle="-|>",  # "normal" arrow
            mutation_scale=15,  # controls arrow head size
            linewidth=2.5,
        )
        fig.patches.append(arrow1)
        
        xyA2 = (line_of_points[-1,0], line_of_points[-1,1])
        xyB2 = (line_of_points[-1,0], plot_dists_room[-1])
        arrow2 = patches.ConnectionPatch(
            xyA2,
            xyB2,
            coordsA=ax.transData,
            coordsB=ax_line.transData,
            color="black",
            arrowstyle="-|>",  # "normal" arrow
            mutation_scale=15,  # controls arrow head size
            linewidth=2.5,
        )
        fig.patches.append(arrow2)
        
        plt.subplots_adjust(hspace=0)
        
    if SIM_MODELS[1]:
        fig = plt.figure(figsize=(6.5, 16))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) # change the height ratio here
        fig.suptitle('Standard model', fontsize=10)
        
        ax = plt.subplot(gs[0])
        ax.scatter(points[:,0], points[:,1], c='gray', alpha=0.5)
        mean = standard_gpedf.get_dist(test)
        CS = ax.contour(X_test[0] , X_test[1], np.reshape(mean, (X_test[0].shape[0], X_test[1].shape[0])), np.linspace(0.25, 2.25, 9))
        ax.clabel(CS, inline=1, fontsize=8)
        ax.axis("square")
        
        plot_dists_standard = standard_gpedf.get_dist(line_of_points)
        plot_dists[1] = plot_dists_standard
        ax.scatter(line_of_points[:,0], line_of_points[:,1], c='red', s=20)
        ax.set_xlim(-10,10)
        ax.set_ylim(-7.7,7.6)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax_line = plt.subplot(gs[1], sharex=ax)
        ax_line.plot(line_of_points[:,0],plot_dists_standard, "orange")
        ax_line.xaxis.set_visible(False)
        ax_line.set_xlim(-10,10)
        xyA1 = (line_of_points[0,0], line_of_points[0,1])
        xyB1 = (line_of_points[0,0], plot_dists_standard[0])
        arrow1 = patches.ConnectionPatch(
            xyA1,
            xyB1,
            coordsA=ax.transData,
            coordsB=ax_line.transData,
            color="black",
            arrowstyle="-|>",  # "normal" arrow
            mutation_scale=15,  # controls arrow head size
            linewidth=2.5,
        )
        fig.patches.append(arrow1)
        
        xyA2 = (line_of_points[-1,0], line_of_points[-1,1])
        xyB2 = (line_of_points[-1,0], plot_dists_standard[-1])
        arrow2 = patches.ConnectionPatch(
            xyA2,
            xyB2,
            coordsA=ax.transData,
            coordsB=ax_line.transData,
            color="black",
            arrowstyle="-|>",  # "normal" arrow
            mutation_scale=15,  # controls arrow head size
            linewidth=2.5,
        )
        fig.patches.append(arrow2)
        
        plt.subplots_adjust(hspace=0)
        
    if SIM_MODELS[2]:
        fig = plt.figure(figsize=(6.5, 16))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1]) # change the height ratio here
        fig.suptitle('Line-based model', fontsize=10)
        
        ax = plt.subplot(gs[0])
        ax.scatter(points[:,0], points[:,1], c='gray', alpha=0.5)
        mean = line_gpedf.get_dist(test)
        CS = ax.contour(X_test[0] , X_test[1], np.reshape(mean, (X_test[0].shape[0], X_test[1].shape[0])), np.linspace(0.25, 2.25, 9))
        ax.clabel(CS, inline=1, fontsize=8)
        ax.axis("square")
        
        plot_dists_line = line_gpedf.get_dist(line_of_points)
        plot_dists[2] = plot_dists_line
        ax.scatter(line_of_points[:,0], line_of_points[:,1], c='red', s=20)
        ax.set_xlim(-10,10)
        ax.set_ylim(-7.7,7.6)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax_line = plt.subplot(gs[1], sharex=ax)
        ax_line.plot(line_of_points[:,0],plot_dists_line, "green")
        ax_line.xaxis.set_visible(False)
        ax_line.set_xlim(-10,10)
        
        xyA1 = (line_of_points[0,0], line_of_points[0,1])
        xyB1 = (line_of_points[0,0], plot_dists_line[0])
        arrow1 = patches.ConnectionPatch(
            xyA1,
            xyB1,
            coordsA=ax.transData,
            coordsB=ax_line.transData,
            color="black",
            arrowstyle="-|>",  # "normal" arrow
            mutation_scale=15,  # controls arrow head size
            linewidth=2.5,
        )
        fig.patches.append(arrow1)
        
        xyA2 = (line_of_points[-1,0], line_of_points[-1,1])
        xyB2 = (line_of_points[-1,0], plot_dists_line[-1])
        arrow2 = patches.ConnectionPatch(
            xyA2,
            xyB2,
            coordsA=ax.transData,
            coordsB=ax_line.transData,
            color="black",
            arrowstyle="-|>",  # "normal" arrow
            mutation_scale=15,  # controls arrow head size
            linewidth=2.5,
        )
        fig.patches.append(arrow2)
        
        plt.subplots_adjust(hspace=0)
    return plot_dists

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
    """
    SIM_MODELS[0] = 1 to simulate room based model, else 0
    SIM_MODELS[1] = 1 to simulate standard model, else 0
    SIM_MODELS[2] = 1 to simulate line based model, else 0
    """
    SIM_MODELS = [1,1,1]
    
    VISUALIZE = 1 # Set to 1 to visualize the EDF of the models, otherwise 0
    
    PLOT = 0 # Set to 1 to plot the computation time graph of the models, otherwise 0
    
    file_name = "scans_simple_hospital_route1.pkl"  # File name of the collected sensor readings
    with open("data/"+file_name, 'rb') as f:
        scans = pickle.load(f)[0:300]

    lamb = 100  # Characteristic length scale
    inducing_thresh = 0.5e-6  # threshold for adaptive inducing point selection
    max_sensor_range = 3.0  # Maximum range of sensor
    room_gpedf, standard_gpedf, line_gpedf = init_models(lamb, max_sensor_range, inducing_thresh)
    
    num_points = 0
    num_points_list = []
    num_points_list_room = []
    num_points_list_line = []
    
    room_gpedf_update_durations = []
    standard_gpedf_update_durations = []
    line_gpedf_update_durations = []
    
    room_gpedf_prediction_durations = []
    standard_gpedf_prediction_durations = []
    line_gpedf_prediction_durations = []
    
    last_robot_pos = None
    tot_points = []
    for i, scan in enumerate(scans): 
        print("****************** scan",i,"******************")
        points, angles, ranges, delta_theta, robot_pos = scan_to_coords(scan, max_sensor_range=max_sensor_range)
        
        if len(points) == 0 or (last_robot_pos is not None and np.linalg.norm(robot_pos-last_robot_pos) < 0.23):
            continue 
        last_robot_pos = robot_pos
        num_points += len(points)
        print("Number of points:",num_points)
        num_points_list.append(num_points)
        tot_points.extend(points)
        if SIM_MODELS[0]:
            start = time.time()
            isUpdated = room_gpedf.update(robot_pos, np.array(points), angles, ranges, delta_theta)
            end = time.time()
            print("current room:",room_gpedf.rs.current_room)
            print("Room GP-EDF update duration:", end-start)
            if i > 10:
                if isUpdated:
                    room_gpedf_update_durations.append(end-start)
                    num_points_list_room.append(num_points)
                start = time.time()
                room_gpedf.get_dist(robot_pos, room_gpedf.rs.current_room)
                end = time.time()
                print("Room GP-EDF prediction duration:", end-start)
            room_gpedf_prediction_durations.append(end-start)
            print("Number of inducing points for room model:", room_gpedf.rs.room_graph.nodes[room_gpedf.rs.current_room]["GP"].gp.variational_strategy.inducing_points.numel())
            print("------------------------------------")
        if SIM_MODELS[1]:
            start = time.time()
            standard_gpedf.update(robot_pos, np.array(points))
            end = time.time()
            print("Standard GP-EDF update duration:", end-start)
            standard_gpedf_update_durations.append(end-start)
            if i > 5:
                start = time.time()
                standard_gpedf.get_dist(robot_pos)
                end = time.time()
                print("Standard GP-EDF prediction duration:", end-start)
            standard_gpedf_prediction_durations.append(end-start)
            print("Number of inducing points for standard model:", standard_gpedf.model.gp.variational_strategy.inducing_points.numel())
            print("------------------------------------")
        if SIM_MODELS[2]:
            start = time.time()
            isUpdated = line_gpedf.update(robot_pos, np.array(points), angles, ranges, delta_theta)
            end = time.time()
            print("Line Segment GP-EDF update duration:", end-start)
            if i > 5:
                if isUpdated:
                    line_gpedf_update_durations.append(end-start)
                    num_points_list_line.append(num_points)
                start = time.time()
                line_gpedf.get_dist(robot_pos)
                end = time.time()
                print("Line Segment GP-EDF prediction duration:", end-start)
            line_gpedf_prediction_durations.append(end-start)
            print("Number of inducing points for line model:", line_gpedf.model.gp.variational_strategy.inducing_points.numel())
            print("------------------------------------")
        
        
    if PLOT:
        plt.figure()
        if SIM_MODELS[0]:
            df_room_update = pd.DataFrame({
            'x': num_points_list_room,
            'y': room_gpedf_update_durations
            })
            df_room_update['smoothed_y'] = df_room_update['y'].ewm(span=150).mean()
            plt.plot(df_room_update['x'], df_room_update['smoothed_y'], 'blue', label='Room-based model')
            plt.ylim(0.0,0.16)
        if SIM_MODELS[1]:
            df_standard_update = pd.DataFrame({
            'x': num_points_list,
            'y': standard_gpedf_update_durations
            })
            df_standard_update['smoothed_y'] = df_standard_update['y'].rolling(window=20).mean()
            plt.plot(df_standard_update['x'], df_standard_update['smoothed_y'], 'orange',label='Global model')
        if SIM_MODELS[2]:
            df_line_update = pd.DataFrame({
            'x': num_points_list_line,
            'y': line_gpedf_update_durations
            })
            df_line_update['smoothed_y'] = df_line_update['y'].ewm(span=20).mean()
            plt.plot(df_line_update['x'], df_line_update['smoothed_y'], 'green', label='Global line-based model')
        plt.title("GP-EDF Update")
        plt.xlabel("Number of data points")
        plt.ylabel("Computation time (seconds)")
        plt.legend()
        
        plt.figure()
        if SIM_MODELS[0]:
            df_room_pred = pd.DataFrame({
            'x': num_points_list,
            'y': room_gpedf_prediction_durations
            })
            df_room_pred['smoothed_y'] = df_room_pred['y'].ewm(span=150).mean()
            plt.plot(df_room_pred['x'], df_room_pred['smoothed_y'], 'blue', label='Room-based model')
            plt.ylim(0.0,0.012)
        if SIM_MODELS[1]:
            df_standard_pred = pd.DataFrame({
            'x': num_points_list,
            'y': standard_gpedf_prediction_durations
            })
            df_standard_pred['smoothed_y'] = df_standard_pred['y'].rolling(window=20).mean()
            plt.plot(df_standard_pred['x'], df_standard_pred['smoothed_y'], 'orange', label='Global model')
        if SIM_MODELS[2]:
            df_line_pred = pd.DataFrame({
            'x': num_points_list,
            'y': line_gpedf_prediction_durations
            })
            df_line_pred['smoothed_y'] = df_line_pred['y'].ewm(span=20).mean()
            plt.plot(df_line_pred['x'], df_line_pred['smoothed_y'], 'green', label='Global line-based model')

        plt.title("GP-EDF Prediction")
        plt.xlabel("Number of data points")
        plt.ylabel("Computation time (seconds)")
        plt.legend()
        
    if VISUALIZE:
        line_of_points = np.column_stack((np.linspace(-6.7, 6.6,100), 2.9*np.ones(100)))
        plot_dists = visualize(SIM_MODELS, np.array(tot_points), room_gpedf, standard_gpedf, line_gpedf, line_of_points)
        plt.figure()
        if SIM_MODELS[0]:
            plt.plot(line_of_points[:,0], plot_dists[0], 'blue', label='Room-based model')
        if SIM_MODELS[1]:
            plt.plot(line_of_points[:,0], plot_dists[1], 'orange', label='Global model')
        if SIM_MODELS[2]:
            plt.plot(line_of_points[:,0], plot_dists[2], 'green', label='Global line-based model')
        plt.title("EDF plot", fontsize=20)
        plt.ylabel("distance (m)", fontsize=14)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        plt.legend()
        
    plt.show()


if __name__ == '__main__':
    main()
    

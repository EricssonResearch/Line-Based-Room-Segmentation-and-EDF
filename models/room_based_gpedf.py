from gpedf_utils.online_sgpr import OnlineSGPRegression as osgpr
import numpy as np
import torch
import gpytorch
import open3d as o3d
from gpedf_utils.streaming_sgpr import LineSegMean
import copy

torch.set_default_dtype(torch.double)

class RoomBasedGPEDF:
    """Class for the room-based GP-EDF module
    """
    def __init__(self, rs, lamb=100, inducing_thresh=0.5e-6):
        self.rs = rs
        self.lamb = lamb  # Characteristic length scale
        self.room_test_data = dict()
        self.pcd = o3d.open3d.geometry.PointCloud()
        model = self.init_gpedf(0, inducing_thresh)
        self.rs.room_graph.add_node(0, lines=[], pos=None, rect=None, GP=model, visibility_graph=None)

    def get_dist(self, X, room, returnGrad=False):  
        """Predict the distance of the GP-EDF

        Args:
            X (array): any point
            room (int): room label
            returnGrad (bool, optional): whether to return the gradient of the prediction. Defaults to False.

        Returns:
            list: prediction
        """
        X = torch.from_numpy(np.atleast_2d(X))
        X.requires_grad = True
        model = self.rs.room_graph.nodes[room]["GP"]
        mean = model.predict(X)
        EDF_mean = - (1/self.lamb) * torch.log(mean)
        if returnGrad:
            EDF_mean.sum().backward()
            grad_norm = torch.nn.functional.normalize(X.grad)
            prediction = [EDF_mean.detach(), grad_norm]
        else:
            prediction = EDF_mean.detach()
        return prediction

    def get_target_value(self, X):
        """Compute target value of input point 

        Args:
            X (array): input point

        Returns:
            float: target value
        """
        dist = torch.zeros((X.shape[0], 1))
        phi = torch.exp(-dist*self.lamb) 
        return phi

    def downsample(self, x, r_voxel=0.15):
        """Downsample input points

        Args:
            x (list): input points
            r_voxel (float, optional): radius of voxel

        Returns:
            list: list of downsampled input points
        """
        zeros = np.zeros((x.shape[0],1))
        XYZ = np.concatenate((x, zeros), 1)
        self.pcd.points = o3d.utility.Vector3dVector(XYZ)
        downXY = np.asarray(self.pcd.voxel_down_sample(voxel_size=r_voxel).points)[:,0:2]
        return downXY

    def init_gpedf(self, room, inducing_thresh):
        """Intialize first GP-EDF model

        Args:
            room (int): room label
            inducing_thresh (float): threshold for adaptive inducing point selection

        Returns:
            OnlineSGPR: online SGPR model
        """
        inducing_points = torch.empty((0,2)) 
        covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
        covar_module.lengthscale = 1/self.lamb
        mean_module = LineSegMean(room, self.rs.room_graph, self.rs.le.line_graph)
        model = osgpr(covar_module=covar_module, mean_module=mean_module,
                            inducing_points=inducing_points, 
                            learn_inducing_locations=True, 
                            jitter=1e-6, inducing_thresh=inducing_thresh)
        return model
    
    def nearest_lineseg(self, points, endpoints):
        """Determines the nearest line segment to each of point in points

        Args:
            points (list): list of points
            endpoints (list): endpoints of line segments

        Returns:
            int: indices of the nearest line segments of each point
            float: shortest distance to the nearest line segment
        """
        points = np.atleast_2d(points)
        endpoints = np.array(endpoints)
        points = points[:, np.newaxis, :] 
        endpoints = endpoints[np.newaxis, :, :, :] 
        a = endpoints[:, :, 0, :] 
        b = endpoints[:, :, 1, :] 
        ab = b - a
        ab2 = np.sum(ab**2, axis=-1)
        ax = points - a 
        abax = np.sum(ab * ax, axis=-1)
        t = np.clip(abax / (ab2+1e-10), 0.0, 1.0)
        xt = ax - t[..., np.newaxis] * ab 
        xt2 = np.sum(xt**2, axis=-1) 
        min_indices = np.argmin(xt2, axis=1)
        return min_indices, np.sqrt(xt2[np.arange(xt2.shape[0]), min_indices])
    
    def split_inducing_points(self, room_pair, inducing_points):
        """Divides the inducing points between two rooms

        Args:
            room_pair (tuple): pair of room labels
            inducing_points (array): array of inducing points

        Returns:
            arrays: inducing points of room 1 and inducing points of room 2
        """
        lines_per_room = [self.rs.room_graph.nodes[room]["lines"] for room in room_pair]
        endpoints_per_room = [[self.rs.le.line_graph.nodes[line]["line"].endpoints for line in room_lines] for room_lines in lines_per_room]
        
        bboxes = [self.rs.room_graph.nodes[room]["rect"] for room in room_pair]

        # Masks for points in bounding boxes
        masks = [(inducing_points[:, 0] >= bbox[0]) & (inducing_points[:, 0] <= bbox[2]) & (inducing_points[:, 1] >= bbox[1]) & (inducing_points[:, 1] <= bbox[3]) for bbox in bboxes]
        combined_mask = np.column_stack(masks)
        inducing_indices1 = np.where(combined_mask[:, 0] & ~combined_mask[:, 1])[0]
        inducing_indices2 = np.where(~combined_mask[:, 0] & combined_mask[:, 1])[0]

        # Masks for points in residual
        residual_mask =  np.all(combined_mask, axis=1) | np.all(~combined_mask, axis=1) 
        residual_points = inducing_points[residual_mask]
        residual_indices = np.where(residual_mask)[0]
        
        if len(residual_indices) > 0:
            # For the remaining points, find the nearest lines for each room
            nearest_lines_and_distances = [self.nearest_lineseg(residual_points, room_endpoints) for room_endpoints in endpoints_per_room]
            # Get nearest line indices and distances for each room
            nearest_lines_indices = np.array([nearest_lines_and_distances[i][0] for i in range(2)]).T
            distances = np.array([nearest_lines_and_distances[i][1] for i in range(2)]).T

            # Get corresponding line for each room and point
            nearest_lines = np.array([[lines_per_room[i][nearest_lines_indices[j][i]] for i in range(2)] for j in range(len(residual_points))])
            
            # Check if points are on the positive side of lines
            positive_sides = np.array([[self.is_point_on_positive_side(point, line) for line in nearest_lines[i]] for i, point in enumerate(residual_points)])
            num_positive_side = positive_sides.astype(int).sum(axis=1)
            
            # Room assignment for points on the positive side of only one line
            positive_side_mask = num_positive_side == 1
            room_indices_positive_side = positive_sides[positive_side_mask].argmax(axis=1)

            # Room assignment for points not on the positive side of any line
            
            no_or_all_positive_side_mask = (num_positive_side == 0) | (num_positive_side > 1)  
            room_indices_no_positive_side = distances[no_or_all_positive_side_mask].argmin(axis=1)

            # Combine room assignments
            room_indices = np.empty(len(residual_points), dtype=int)
            room_indices[positive_side_mask] = room_indices_positive_side
            room_indices[no_or_all_positive_side_mask] = room_indices_no_positive_side

            # Assign points to rooms based on room_indices
            inducing_indices1 = np.concatenate((inducing_indices1, residual_indices[room_indices == 0]))
            inducing_indices2 = np.concatenate((inducing_indices2, residual_indices[room_indices == 1]))

        return inducing_indices1, inducing_indices2

    def is_point_on_positive_side(self, point, line_key):
        """Checks if a point is on the positive side of a line

        Args:
            point (array): any point
            line_key (int): key of a line segment

        Returns:
            bool: true if point is on positive side, false otherwise
        """
        line = self.rs.le.line_graph.nodes[line_key]["line"]
        return self.rs.le.signed_dist(point, line.endpoints[0], line.normal) > 0
        
    def merge_gp_models(self, room_pair):
        """Merges two local GP-EDF models after two rooms have merged

        Args:
            room_pair (tuple): pair of room labels
        """
        with torch.no_grad():
            other_model = self.rs.room_graph.nodes[room_pair[0]]["GP"]
            merged_model = self.rs.room_graph.nodes[room_pair[1]]["GP"] 
            merged_inducing_points = torch.cat((merged_model.gp.variational_strategy.inducing_points, other_model.gp.variational_strategy.inducing_points))
            old_merged_inducing_points = torch.cat([ merged_model.gp._old_strat.inducing_points, other_model.gp._old_strat.inducing_points])

            merged_old_strat_mean = torch.cat([merged_model.gp._old_strat._variational_distribution.variational_mean, other_model.gp._old_strat._variational_distribution.variational_mean])
            merged_old_strat_covar = torch.block_diag(merged_model.gp._old_strat._variational_distribution.chol_variational_covar, other_model.gp._old_strat._variational_distribution.chol_variational_covar)
            old_strat_dist = gpytorch.variational.CholeskyVariationalDistribution(
                                                old_merged_inducing_points.size(-2)
                                                )
            merged_model.gp._old_strat = gpytorch.variational.UnwhitenedVariationalStrategy(
                                                    merged_model,
                                                    old_merged_inducing_points,
                                                    old_strat_dist,
                                                    learn_inducing_locations=True)
            merged_model.gp._old_strat._variational_distribution.variational_mean.data.copy_(merged_old_strat_mean)
            merged_model.gp._old_strat._variational_distribution.chol_variational_covar.data.copy_(merged_old_strat_covar)
            merged_model.gp._old_strat.variational_params_initialized.fill_(1)
            
            merged_model.gp._old_C_matrix = torch.block_diag(merged_model.gp._old_C_matrix.evaluate(), other_model.gp._old_C_matrix.evaluate()) 

            merged_variational_mean = torch.cat([merged_model.gp.variational_strategy._variational_distribution.variational_mean, other_model.gp.variational_strategy._variational_distribution.variational_mean])
            merged_variational_covar = torch.block_diag(merged_model.gp.variational_strategy._variational_distribution.chol_variational_covar, other_model.gp.variational_strategy._variational_distribution.chol_variational_covar)
            
            merged_dist = gpytorch.variational.CholeskyVariationalDistribution(
                                        merged_inducing_points.size(-2)
                                        )
            merged_model.gp.variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
                                                    merged_model,
                                                    merged_inducing_points,
                                                    merged_dist,
                                                    learn_inducing_locations=True)
            merged_model.gp.variational_strategy._variational_distribution.variational_mean.data.copy_(merged_variational_mean)
            merged_model.gp.variational_strategy._variational_distribution.chol_variational_covar.data.copy_(merged_variational_covar)
            merged_model.gp.variational_strategy.variational_params_initialized.fill_(1)

            self.rs.room_graph.remove_node(room_pair[0])
            self.rs.room_graph.nodes[room_pair[1]]["GP"] = merged_model
    
    def split_gp_model(self, room_pair):
        """Splits a local GP-EDF model into two child models after a room split has occured.

        Args:
            room_pair (tuple): pair of room labels
        """
        with torch.no_grad():
            split_model = copy.copy(self.rs.room_graph.nodes[room_pair[0]]["GP"].gp)
            split_inducing_points = copy.copy(split_model.variational_strategy.inducing_points.detach())
            inducing_indices1, inducing_indices2 = self.split_inducing_points(room_pair, split_inducing_points)
            split_variational_mean = copy.copy(split_model.variational_strategy._variational_distribution.variational_mean.detach())
            split_variational_covar = copy.copy(split_model.variational_strategy._variational_distribution.chol_variational_covar.detach())
            new_variational_mean1 = split_variational_mean[inducing_indices1]
            new_variational_covar1 = split_variational_covar[inducing_indices1,:][:,inducing_indices1]
            
            model1 = osgpr(covar_module=copy.copy(split_model.covar_module), mean_module=copy.copy(split_model.mean_module),
                        inducing_points=split_inducing_points[inducing_indices1], 
                        learn_inducing_locations=True, jitter=1e-6)

            model1.gp.variational_strategy._variational_distribution.variational_mean.data.copy_(new_variational_mean1)
            model1.gp.variational_strategy._variational_distribution.chol_variational_covar.data.copy_(new_variational_covar1)
            
            model1.gp.mean_module.room_graph = self.rs.room_graph
            model1.gp.mean_module.line_graph = self.rs.le.line_graph
            model1.gp._old_strat = copy.copy(split_model._old_strat)
            model1.gp._old_kernel = copy.copy(split_model._old_kernel)
            model1.gp._old_C_matrix = copy.copy(split_model._old_C_matrix)
            model1.gp.variational_strategy.variational_params_initialized.fill_(1)

            self.rs.room_graph.nodes[room_pair[0]]["GP"] = model1
            
            new_variational_mean2 = split_variational_mean[inducing_indices2]
            new_variational_covar2 = split_variational_covar[inducing_indices2,:][:,inducing_indices2]
                
            model2 = osgpr(covar_module=copy.copy(split_model.covar_module), mean_module=copy.copy(split_model.mean_module),
                        inducing_points=split_inducing_points[inducing_indices2], 
                        learn_inducing_locations=True, jitter=1e-6)
            
            model2.gp.variational_strategy._variational_distribution.variational_mean.data.copy_(new_variational_mean2)
            model2.gp.variational_strategy._variational_distribution.chol_variational_covar.data.copy_(new_variational_covar2)
            model2.gp.mean_module.room_label = room_pair[1] 
            model2.gp.mean_module.room_graph = self.rs.room_graph
            model2.gp.mean_module.line_graph = self.rs.le.line_graph
            model2.gp._old_strat = copy.copy(split_model._old_strat)
            model2.gp._old_kernel = copy.copy(split_model._old_kernel)
            model2.gp._old_C_matrix = copy.copy(split_model._old_C_matrix)
            model2.gp.variational_strategy.variational_params_initialized.fill_(1)
            self.rs.room_graph.nodes[room_pair[1]]["GP"] = model2
        
    def adjust_gp_models(self,room_pair):
        """Adjust two local GP-EDF models after their room boundary has been altered.

        Args:
            room_pair (tuple): pair of room labels
        """
        with torch.no_grad():
            inducing_points1 = copy.copy(self.rs.room_graph.nodes[room_pair[0]]["GP"].gp.variational_strategy.inducing_points)
            line_keys1 = self.rs.room_graph.nodes[room_pair[0]]["lines"]
            endpoints = [self.rs.le.line_graph.nodes[line]["line"].endpoints for line in line_keys1] 
            endpoints += [self.rs.le.line_graph.nodes[line]["line"].endpoints for line in self.rs.room_changes[room_pair]]  
            min_indices, _ = self.nearest_lineseg(inducing_points1, endpoints)
            moved_ip = np.where(min_indices > len(line_keys1)-1)[0]
            if len(moved_ip) > 0:
                model1 = self.rs.room_graph.nodes[room_pair[0]]["GP"].gp
                variational_mean1 = copy.copy(model1.variational_strategy._variational_distribution.variational_mean)
                variational_covar1 = copy.copy(model1.variational_strategy._variational_distribution.chol_variational_covar)
                
                keep_indices = [i for i in range(len(inducing_points1)) if i not in moved_ip]
                new_inducing_points1 = inducing_points1[keep_indices]
                variational_distribution1 = gpytorch.variational.CholeskyVariationalDistribution(
                                            new_inducing_points1.size(-2)
                                            )
                model1.variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
                                            model1,
                                            new_inducing_points1,
                                            variational_distribution1,
                                            learn_inducing_locations=True)
                model1.variational_strategy._variational_distribution.variational_mean.data.copy_(variational_mean1[keep_indices])
                model1.variational_strategy._variational_distribution.chol_variational_covar.data.copy_(variational_covar1[keep_indices,:][:,keep_indices])
                model1.variational_strategy.variational_params_initialized.fill_(1)
                
                
                model2 = self.rs.room_graph.nodes[room_pair[1]]["GP"].gp
                inducing_points2 = model2.variational_strategy.inducing_points
                model2._old_C_matrix = model2.current_C_matrix(inducing_points1[moved_ip])
                model2._old_strat = model2.variational_strategy
                if len(inducing_points2) > 0:
                    new_inducing_points2 = torch.cat([inducing_points2, inducing_points1[moved_ip]])
                else:
                    new_inducing_points2 = inducing_points1[moved_ip]
                
                variational_distribution2 = gpytorch.variational.CholeskyVariationalDistribution(
                                            new_inducing_points2.size(-2)
                                            )
                model2.variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
                                            model2,
                                            new_inducing_points2,
                                            variational_distribution2,
                                            learn_inducing_locations=True)
                x_new = inducing_points1[moved_ip]
                y_new = self.get_target_value(x_new)
                model2.update_variational_distribution(x_new, y_new)
                
    def update_local_models(self):
        """Performs updates of local GP-EDF models based on changes in the room configuration.
        """
        for room_pair in self.rs.room_changes:
            if self.rs.room_changes[room_pair] == "new_room":
                self.split_gp_model(room_pair)
            elif self.rs.room_changes[room_pair] == "merged_rooms":
                self.merge_gp_models(room_pair)
            else:
                self.adjust_gp_models(room_pair)
                
    def assign_points(self):
        """Distributes residual points from line segment detection module 
        between neighbouring GP-EDF models

        Returns:
            dict: dictionary of the points to add to each room
        """
        centroids = np.array([np.mean(list(cluster.values()), axis=0) for cluster in self.rs.le.remaining_cluster_points])
        bboxes = [self.rs.room_graph.nodes[room]["rect"] for room in self.rs.local_rooms]

        # Masks for points in bounding boxes
        masks = [(centroids[:, 0] >= bbox[0]) & (centroids[:, 0] <= bbox[2]) & (centroids[:, 1] >= bbox[1]) & (centroids[:, 1] <= bbox[3]) for bbox in bboxes]
        combined_mask = np.column_stack(masks)
        num_in_one_bbox = combined_mask.astype(int).sum(axis=1)
        cluster_indices_per_room = {room: list(np.where((combined_mask[:, i]) & (num_in_one_bbox == 1))[0]) for i, room in enumerate(self.rs.local_rooms)}
        # Masks for points in residual
        residual_mask = num_in_one_bbox != 1
        residual_centroids = centroids[residual_mask]
        residual_indices = np.where(residual_mask)[0]
        if len(residual_indices) > 0:
            lines_per_room = [self.rs.room_graph.nodes[room]["lines"] for room in self.rs.local_rooms]
            endpoints_per_room = [[self.rs.le.line_graph.nodes[line]["line"].endpoints for line in room_lines] for room_lines in lines_per_room]
            nearest_lines_and_distances = [self.nearest_lineseg(residual_centroids, room_endpoints) for room_endpoints in endpoints_per_room]
            
            # Get nearest line indices and distances for each room
            nearest_lines_indices = np.array([nearest_lines_and_distances[i][0] for i in range(len(nearest_lines_and_distances))]).T
            distances = np.array([nearest_lines_and_distances[i][1] for i in range(len(nearest_lines_and_distances))]).T

            # Get corresponding line for each room and point
            nearest_lines = np.array([[lines_per_room[i][nearest_lines_indices[j][i]] for i in range(len(lines_per_room))] for j in range(len(residual_centroids))])
            
            # Check if points are on the positive side of lines
            positive_sides = np.array([[self.is_point_on_positive_side(point, line) for line in nearest_lines[i]] for i, point in enumerate(residual_centroids)])

            # Assign points to rooms based on room_indices
            room_indices = np.empty(len(residual_centroids), dtype=int)
            
            # Room assignment for points on the positive side of one line
            positive_side_mask = positive_sides.astype(int).sum(axis=1) == 1
            if sum(positive_side_mask) > 0:
                room_indices_positive_side = positive_sides[positive_side_mask].argmax(axis=1)
                room_indices[positive_side_mask] = room_indices_positive_side
            # Room assignment for points on the positive side of multiple lines
            multiple_positive_side_mask = positive_sides.astype(int).sum(axis=1) > 1
            if sum(multiple_positive_side_mask) > 0:
                distances_multiple_positive = distances[multiple_positive_side_mask].copy()
                positive_sides_multiple_positive = positive_sides[multiple_positive_side_mask]
                
                # Set the distances to the lines that the points are not on the positive side of to infinity
                distances_multiple_positive[~positive_sides_multiple_positive] = np.inf
                room_indices_multiple_positive_side = np.argmin(distances_multiple_positive, axis=1)
                room_indices[multiple_positive_side_mask] = room_indices_multiple_positive_side
            
            # Room assignment for points not on the positive side of any line
            no_positive_side_mask = positive_sides.astype(int).sum(axis=1) == 0
            if sum(no_positive_side_mask) > 0:
                room_indices_no_positive_side = distances[no_positive_side_mask].argmin(axis=1)
                room_indices[no_positive_side_mask] = room_indices_no_positive_side
                
            for i, room in enumerate(self.rs.local_rooms):
                indices_in_room = list(residual_indices[room_indices == i])
                if len(indices_in_room) > 0:
                    cluster_indices_per_room[room] += indices_in_room
        return cluster_indices_per_room
        

    def update(self, robot_pos, points, angles, ranges, delta_theta):
        """ Runs the room-based GP-EDF module for each arriving sensor reading

        Args:
            robot_pos (array): current position of the robot
            points (list): list of measurement points
            angles (list): list of angles of the measurements
            ranges (list): list of ranges to the points
            delta_theta (float): angular resolution of the sensor
        """
        self.rs.update(robot_pos, points, angles, ranges, delta_theta)
        
        if len(self.rs.room_changes) > 0:
            self.update_local_models()
        
        
        if len(self.rs.le.new_remaining_points) > 0:
            if len(self.rs.local_rooms) > 1:
                cluster_indices_per_room = self.assign_points()

                cluster_points_array = np.array(list(map(lambda cluster: list(map(list,cluster.values())), self.rs.le.remaining_cluster_points)))
                for label in cluster_indices_per_room:
                    room_indices = cluster_indices_per_room[label]
                    if len(room_indices) > 0:
                        room_cluster_points = cluster_points_array[room_indices].tolist()
                        if len(room_cluster_points) > 1:
                            new_points = np.array(sum(room_cluster_points, []))
                        else:
                            new_points = np.array(room_cluster_points[0])
                        down_points = torch.from_numpy(self.downsample(new_points))
                        target = self.get_target_value(down_points)
                        self.rs.room_graph.nodes[label]["GP"].update(down_points, target) 
            else:
                new_points = np.array(list(self.rs.le.new_remaining_points.values()))
                down_points = torch.from_numpy(self.downsample(new_points))
                target = self.get_target_value(down_points)
                self.rs.room_graph.nodes[self.rs.current_room]["GP"].update(down_points, target)
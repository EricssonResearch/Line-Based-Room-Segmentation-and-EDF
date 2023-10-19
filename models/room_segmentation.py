import numpy as np
import networkx as nx
from shapely.geometry import LineString, MultiLineString
from rtree import index
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster._spectral import cluster_qr
from collections import Counter, defaultdict
import copy
           
class RoomSegmentation:
    """Class the room segmentation module
    """
    def __init__(self, le, max_sensor_range=3.0, gamma_dist=0.02, 
                 gamma_rob=0.005, fiedler_thresh=0.18, 
                 vis_dist_thresh=8.0, remerge_thresh=0.5):   
        
        self.max_sensor_range = max_sensor_range  # Maximum range of sensor
        self.radius = self.max_sensor_range + 0.4
        self.room_graph = nx.Graph()
        self.le = le
        
        p = index.Property()
        p.dimension = 2 # 2D data
        p.variant = index.RT_Star # Use R*-tree variant
        self.rtree = index.Index(properties=p)
        
        self.next_room_label = 1
        self.sorted_lines = dict()
        self.previous_relabeling = []
        self.previous_clusters = []
        self.current_room = 0
        self.num_clusters = 1
        self.prev_room_lines = None
        self.room_changes = []
        self.new_room = None
        self.robot_pos = None
        self.visibility_graph = None
        self.local_rooms = []
        
        self.fiedler_thresh = fiedler_thresh  # Threshold for the Fiedler value
        self.gamma_dist = gamma_dist  # Tuning parameter for the distance-based edge weight
        self.gamma_rob = gamma_rob  # Tuning parameter for the robot position-based edge weight
        self.vis_dist_thresh = vis_dist_thresh  # Maximum distance for creating an edge between two line segments
        self.remerge_thresh = remerge_thresh  # Threshold for remerging a room that has been split

    def update(self, robot_pos, points, angles, ranges, delta_theta):
        """ Runs the room segmentation process for each arriving sensor reading

        Args:
            robot_pos (array): current position of the robot
            points (list): list of measurement points
            angles (list): list of angles of the measurements
            ranges (list): list of ranges to the points
            delta_theta (float): angular resolution of the sensor
        """
        self.room_changes = dict()
        self.new_room = None
        self.robot_pos = robot_pos 
        local_lines = []
        self.local_rooms = [self.current_room]
        if self.room_graph.number_of_nodes() > 0:
            local_lines += self.room_graph.nodes[self.current_room]["lines"]
            edges = self.room_graph.edges(self.current_room)
            if len(edges) > 0:
                for e in edges:
                    self.local_rooms.append(e[1])
                    local_lines += self.room_graph.nodes[e[1]]["lines"]
        else:
            self.room_graph.add_node(0, lines=[], pos=self.robot_pos, rect=None, GP=None, visibility_graph=None)
        if len(local_lines) > 0:
            linesegs_within_range = self.linesegs_within_range(local_lines, self.radius)
            self.local_rooms = list(np.unique(list(nx.get_node_attributes(self.le.line_graph.subgraph(linesegs_within_range), "label").values())))
            if self.current_room not in self.local_rooms:
                self.local_rooms.insert(0, self.current_room)
        else:
            linesegs_within_range = []
        
        self.le.update(points, angles, ranges, self.robot_pos, self.current_room, linesegs_within_range, delta_theta, self.room_graph)
        if self.room_graph.number_of_nodes() == 1 and self.room_graph.nodes[0]["rect"] is None:
            if self.le.line_graph.number_of_nodes() > 0:
                line_ends = [self.le.line_graph.nodes[line]["line"].endpoints for line in self.room_graph.nodes[0]["lines"]] 
                multi_lines = MultiLineString(line_ends)
                rect = multi_lines.bounds
                self.rtree.insert(0, rect)
                self.room_graph.nodes[0]["pos"] = np.array(list(multi_lines.centroid.coords)).flatten()
                self.room_graph.nodes[0]["rect"] = rect
            return
        local_lines = []
        for room in self.local_rooms:
            room_lines = self.room_graph.nodes[room]["lines"]
            local_lines += room_lines
                
        if self.le.isUpdated and len(local_lines) > 2: 
            self.num_local_clusters = len(self.local_rooms)
            i = 0
            while i < len(self.local_rooms):
                room = self.local_rooms[i]
                room_lines = self.room_graph.nodes[room]["lines"]
                if len(room_lines) == 1:
                    neighbors = list(self.room_graph.neighbors(room))
                    if len(neighbors) > 1:
                        neighbor = min(neighbors, key=lambda label:abs(label-room))
                    else:
                        neighbor = neighbors[0]
                    self.room_graph.nodes[neighbor]["lines"].extend(room_lines)
                    self.le.line_graph.nodes[room_lines[0]]["label"] = neighbor
                    self.rtree.delete(room, self.room_graph.nodes[room]["rect"])
                    self.room_graph.remove_node(room)
                    self.local_rooms.remove(room)
                    if self.current_room == room:
                        self.current_room = neighbor
                    i -= 1
                i += 1
            visibility_graph = self.create_visibility_graph(local_lines)     
            isClustered, cluster_labels = self.spectral_clustering(visibility_graph)
            if isClustered:
                relabeling = self.local2global_labeling(cluster_labels, visibility_graph)
            else:
                relabeling = list(nx.get_node_attributes(visibility_graph, "label").values())
        else:
            visibility_graph = self.le.line_graph.subgraph(local_lines)
            relabeling = list(nx.get_node_attributes(visibility_graph, "label").values())
        self.room_changes = self.update_rooms(relabeling, visibility_graph)

        room = self.identify_room(self.robot_pos)
        if room != self.current_room and not self.room_graph.has_edge(room, self.current_room):
            self.room_graph.add_edge(room, self.current_room)
        self.current_room = room
        
    
    def spectral_clustering(self, visibility_graph):
        """ Updates local room count based on current local room count 
        and performs spectral clustering on the visibility graph to divide 
        the line segments into rooms.

        Args:
            visibility_graph (networkx graph): visibility graph

        Returns:
            bool: true if the rooms were updated, false otherwise
            dict: local room labels
        """
        num_local_clusters = len(self.local_rooms)
        global_labels = np.array(list(nx.get_node_attributes(visibility_graph, "label").values())).astype(int)
        unique_labels = np.unique(global_labels).astype(int)
        local_labels = np.zeros(len(global_labels)).astype(int)
        isUpdated = False
        if num_local_clusters > 1:
            current_local_label = len(unique_labels)-1
            local_labels[global_labels==self.current_room] = current_local_label
            other_local_label = current_local_label-1
            neighbor_labels = list(unique_labels)
            neighbor_labels.remove(self.current_room)
            for global_label in neighbor_labels:
                local_labels[global_labels==global_label] = other_local_label
                other_local_label -= 1
        local_labels_dict = {k:l for k,l in zip(visibility_graph.nodes(),local_labels)}

        first_label = int(max(local_labels_dict.values()))
        current_room_lines = self.room_graph.nodes[self.current_room]["lines"]
        if num_local_clusters > 1:
            current_graph = visibility_graph.subgraph(current_room_lines)
        else:
            current_graph = visibility_graph
        
        current_components = list(nx.connected_components(current_graph))
        clusters = defaultdict(list)
        foundNew = False

        isNew = True
        if num_local_clusters > 1:
            L = nx.normalized_laplacian_matrix(visibility_graph)

            # Compute the eigenvalues and eigenvectors of the Laplacian matrix
            eigenvalues, eigenvectors = np.linalg.eig(L.toarray())
            
            # Sort the eigenvalues and corresponding eigenvectors in ascending order
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:,idx]
            
            # Eigengap heuristic
            eigengap = np.diff(eigenvalues)
            n_clusters = np.argmax(eigengap) + 1
            if n_clusters < num_local_clusters+1:
                isNew = False
        isUpdated = False
        if len(current_components) == 1 and isNew:
            # Compute adjacency and Laplacian matrix
            L_current = nx.normalized_laplacian_matrix(current_graph)

            # Compute the eigenvalues and eigenvectors of the Laplacian matrix
            current_eigenvalues, current_eigenvectors = np.linalg.eig(L_current.toarray())

            # Sort the eigenvalues and corresponding eigenvectors in ascending order
            idx = current_eigenvalues.argsort()
            current_eigenvalues = current_eigenvalues[idx]
            current_eigenvectors = current_eigenvectors[:,idx]
            
            if current_eigenvalues[1] <= self.fiedler_thresh:
                foundNew = True
                maps = current_eigenvectors[:,:2]
                labels = list(cluster_qr(maps))
                
                for node, label in zip(current_graph.nodes(), labels):
                    local_labels_dict[node] = first_label + label
                    clusters[first_label + label].append(node)
                    
                # Examine split rooms for remerging
                inter_cluster_edges = nx.edge_boundary(current_graph, clusters[first_label], clusters[first_label + 1], data=True)
                num_visibility_edges = 0
                visited_nodes = []
                sum_weights = 0
                for u,v,d in list(inter_cluster_edges):
                    if d['weight'] > 1e-3 and not (u in visited_nodes or v in visited_nodes):
                        num_visibility_edges += 1
                        sum_weights += d['weight']
                        visited_nodes += [u,v]

                if len(list(inter_cluster_edges)) > 0:
                    inter_similarity = num_visibility_edges/min(len(clusters[first_label]), len(clusters[first_label+1]))
                    if inter_similarity >= self.remerge_thresh:
                        for node in clusters[first_label+1]:
                            local_labels_dict[node] = first_label
                    else:
                        isUpdated = True
                else:
                    isUpdated = True
                
        if len(self.le.new_lines) > 0 and not foundNew and len(self.local_rooms) > 1:
            # Either merge rooms or retain the current room count
            merge_gap = eigengap[num_local_clusters-2]
            current_gap = eigengap[num_local_clusters-1]
            isUpdated = True
            if n_clusters <= 1 and (num_local_clusters == 1 or (num_local_clusters == 2 and abs(merge_gap-current_gap) >= 1)):
                local_labels_dict = {k:0 for k in visibility_graph.nodes()}
            else: 
                if current_gap < merge_gap and abs(merge_gap-current_gap) >= 0.1:
                    n_clusters = num_local_clusters-1
                else:
                    n_clusters = num_local_clusters

                maps = eigenvectors[:,:n_clusters]
                local_labels = cluster_qr(maps)
                local_labels_dict = {k:l for k,l in zip(visibility_graph.nodes(), local_labels)}

        return isUpdated, local_labels_dict
    
    def find_missing_label(self, labels):
        """Computes the room labels that are missing in the order

        Args:
            labels (list): list of existing room labels

        Returns:
            list: missing room labels
        """
        label_set = set(labels)
        max_label = max(labels)
        full_set = set(range(max_label + 1))
        
        # use set difference to get the missing element
        return list(full_set - label_set)
            
    def local2global_labeling(self, local_labels_dict, visibility_graph):
        """Translates the updated local room labels to the global room labels

        Args:
            local_labels_dict (dict): new local room labels
            visibility_graph (networkx graph): visibility graph

        Returns:
            list: translated global room labels
        """
        local_labels = list(local_labels_dict.values())
        local_line_nodes = list(local_labels_dict)
        global_labels = list(nx.get_node_attributes(visibility_graph, "label").values())
        unique_local_labels = np.unique(local_labels)
        unique_global_labels = np.unique(global_labels)
        relabeling = copy.copy(local_labels)

        if (len(local_labels) == len(self.previous_clusters) and np.all(local_labels == self.previous_clusters)): 
            return self.previous_relabeling
        elif len(unique_local_labels) == 1: 
            relabeling = list(min(unique_global_labels)*np.ones(len(local_labels)).astype(int))
        elif np.all(local_labels == global_labels) and not 1 in list(Counter(global_labels).values()):
            relabeling = global_labels
        else:
            self.next_room_label = max(list(self.room_graph.nodes()))+1
            contingency = contingency_matrix(global_labels, local_labels).astype('double')
            for col in range(contingency.shape[1]):
                for row in range(contingency.shape[0]):
                    if contingency[row,col] > 0:
                        list1 = np.where(np.array(local_labels) == col)[0]
                        list2 = np.where(np.array(global_labels) == unique_global_labels[row])[0]
                        indices = list(set(list1).intersection(list2))
                        if len(indices) > 0:
                            tot_weight = 0
                            for i in indices:
                                tot_weight += self.le.next_line - list(local_line_nodes)[i]
                            contingency[row,col] = tot_weight/len(indices)
            cost = -contingency
            # use the linear sum assignment algorithm to find the optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            mapping = {}
            
            remaining_clusters = list(unique_local_labels)

            # loop through each pair of indices
            mapping_global2local = dict()
            for i, j in zip(row_ind, col_ind):
                # map the label in the first set to the label in the second set
                mapping_global2local[unique_global_labels[i]] = j
                mapping[j] = unique_global_labels[i]
                remaining_clusters.remove(j)

            if len(remaining_clusters) > 0:
                label = remaining_clusters[0]
                is_new_removed = False
                if local_labels.count(label) == 1:
                    mapping[label] = self.current_room 
                    is_new_removed = True
                else:
                    new_room_lines = [local_line_nodes[i] for i in range(len(local_labels)) if local_labels[i]==label]
                    old_line_labels = np.unique(list(nx.get_node_attributes(visibility_graph.subgraph(new_room_lines), "label").values()))
                    if not np.any(old_line_labels == self.current_room):
                        is_new_removed = True
                        mapping[label] = np.unique(old_line_labels)[0]

                if not is_new_removed:
                
                    missing = self.find_missing_label(self.room_graph.nodes())
                    if len(missing) > 0:
                        value = missing[0]
                    else:
                        value = self.next_room_label
                        self.next_room_label += 1
                    mapping[label] = value

            counter = Counter(local_labels)
            for (label, count) in counter.items():
                if count == 1:
                    index = local_labels.index(label)
                    mapping[label] =  mapping[mapping_global2local[global_labels[index]]]
            
            relabeling = np.array(global_labels)
            for i, x in enumerate(local_labels):
                if relabeling[i] != mapping[x] and relabeling[i] == self.current_room:
                    relabeling[i] = mapping[x]
            relabeling = list(relabeling)
            for l in list(set(relabeling)):
                if relabeling.count(l) == 1:
                    idx = relabeling.index(l)
                    max_label = max(relabeling)
                    if l < max_label:
                        max_indices = np.where(np.array(relabeling) == max_label)[0]
                        for max_idx in max_indices:
                            relabeling[max_idx] = l
                    relabeling[idx] = self.le.line_graph.nodes[list(local_line_nodes)[idx]]["label"]
            self.previous_clusters = local_labels
            self.previous_relabeling = relabeling
        self.num_clusters = len(set(relabeling))
        return relabeling
    
    def update_rooms(self, relabeling, visibility_graph):
        """Updates the rooms based on the result of the graph clustering

        Args:
            relabeling (list): updated global room labels
            visibility_graph (networkx graph): visibility graph

        Returns:
            dict: dictionary of room changes 
        """
        visibility_graph_nodes = list(visibility_graph.nodes())
        updated_rooms = []  
        room_changes = dict()
        for (label, line) in zip(relabeling, visibility_graph_nodes):
            line_label = self.le.line_graph.nodes[line]["label"]
            if line_label != label:
                if (line_label, label) not in room_changes:
                    updated_rooms.append(label)
                    if not self.room_graph.has_node(label):
                        self.new_room = label
                        self.room_graph.add_node(label, lines=[], pos=None, rect=None, GP=None, visibility_graph=None)
                        new_lines = [visibility_graph_nodes[i] for i in range(len(relabeling)) if relabeling[i] == label]
                        self.room_graph.add_edge(self.current_room, label)
                        room_changes[(self.current_room, label)] = "new_room"
                        for edge in visibility_graph.edges(new_lines):
                            neighbor_label = visibility_graph.nodes[edge[1]]["label"]
                            if edge[1] not in new_lines and neighbor_label != self.current_room: # and (new_line_global_label neighbor_label != new_line_global_label):
                                self.room_graph.add_edge(neighbor_label, label)
                                if self.room_graph.has_edge(neighbor_label, self.current_room):
                                    self.room_graph.remove_edge(neighbor_label, self.current_room)
                    elif self.room_graph.has_node(line_label) and line_label not in relabeling:
                        room_changes[(line_label, label)] = "merged_rooms"
                        self.local_rooms.remove(line_label)
                        if self.room_graph.nodes[line_label]["rect"] is not None:
                            self.rtree.delete(line_label, self.room_graph.nodes[line_label]["rect"])
                        for neighbor_label in self.room_graph.neighbors(line_label):
                            if neighbor_label != label:
                                self.room_graph.add_edge(neighbor_label, label)
                        if self.room_graph.nodes[line_label]["GP"] is None: 
                            self.room_graph.remove_node(line_label)   
                    else:
                        updated_rooms.append(line_label)
                        room_changes[(line_label, label)] = [line] 
                elif type(room_changes[(line_label, label)]) is list:
                    room_changes[(line_label, label)].append(line)
                if self.room_graph.has_node(line_label):
                    self.room_graph.nodes[line_label]["lines"].remove(line)
                self.le.line_graph.nodes[line]["label"] = label
                self.room_graph.nodes[label]["lines"].append(line)
           
        other_rooms = [room for room in self.local_rooms if room not in updated_rooms and self.room_graph.has_node(room)]
        for label in updated_rooms:
            if self.room_graph.nodes[label]["rect"] is not None:
                self.rtree.delete(label, self.room_graph.nodes[label]["rect"])
            
            line_ends = [self.le.line_graph.nodes[line]["line"].endpoints for line in self.room_graph.nodes[label]["lines"]] 
            multi_lines = MultiLineString(line_ends)
            rect = multi_lines.bounds
            self.rtree.insert(label, rect)
            self.room_graph.nodes[label]["pos"] = np.array(list(multi_lines.centroid.coords)).flatten()
            self.room_graph.nodes[label]["rect"] = rect
        
        if other_rooms:
            for label in other_rooms:
                rect1 = self.room_graph.nodes[label]["rect"]
                line_ends = [self.le.line_graph.nodes[line]["line"].endpoints for line in self.room_graph.nodes[label]["lines"]]
                multi_lines = MultiLineString(line_ends)
                rect2 = multi_lines.bounds
                if sum(abs(a - b) for a, b in zip(rect1, rect2)) > 0.3:
                    self.rtree.delete(label, self.room_graph.nodes[label]["rect"])
                    self.rtree.insert(label, rect2)
                    self.room_graph.nodes[label]["rect"] = rect2
                self.room_graph.nodes[label]["pos"] = np.array(list(multi_lines.centroid.coords)).flatten()
        
        for room in other_rooms+updated_rooms:
            self.room_graph.nodes[room]["visibility_graph"] = visibility_graph.subgraph(self.room_graph.nodes[room]["lines"])
        return room_changes

    def identify_room(self, point, visualize_GP=False):
        """Identifies the room in which a point is located

        Args:
            point (array): any point
            visualize_GP (bool, optional): whether the room identification will be used 
                                            for visualization. Defaults to False.

        Returns:
            int: identified room
        """
        room_label = 0
        if self.room_graph.number_of_nodes() > 1:
            query = np.tile(point, 2)
            potential_rooms = list(self.rtree.intersection(query))
            if len(potential_rooms) > 0:
                if len(potential_rooms) > 1:
                    min_positive_dist_per_room = dict()
                    min_dist_per_room = dict()
                    for label in potential_rooms:
                        endpoints = np.array([self.le.line_graph.nodes[line]["line"].endpoints for line in self.room_graph.nodes[label]["lines"]])
                        min_index, min_dist = self.nearest_lineseg(point, endpoints)
                        min_line_key = self.room_graph.nodes[label]["lines"][min_index]
                        min_line = self.le.line_graph.nodes[min_line_key]["line"]
                        min_dist_per_room[label] = min_dist
                        if self.le.signed_dist(point, min_line.endpoints[0], min_line.normal) > 0:
                            min_positive_dist_per_room[label] = min_dist
                    if len(min_positive_dist_per_room) > 1:
                        room_label = min(min_positive_dist_per_room, key=min_positive_dist_per_room.get)
                    elif len(min_positive_dist_per_room) == 1:
                        room_label = next(iter(min_positive_dist_per_room.keys()))
                    else:
                        room_label = min(min_dist_per_room, key=min_dist_per_room.get)
                else:
                    room_label = potential_rooms[0]
            else:
                if visualize_GP:
                    room_label = list(self.rtree.nearest(query,3))
                else:
                    potential_rooms = list(self.rtree.nearest(query,2))
                    if len(potential_rooms) > 1:
                        min_positive_dist_per_room = dict()
                        min_dist_per_room = dict()
                        for label in potential_rooms:
                            endpoints = np.array([self.le.line_graph.nodes[line]["line"].endpoints for line in self.room_graph.nodes[label]["lines"]])
                            min_index, min_dist = self.nearest_lineseg(point, endpoints)
                            min_line_key = self.room_graph.nodes[label]["lines"][min_index]
                            min_line = self.le.line_graph.nodes[min_line_key]["line"]
                            min_dist_per_room[label] = min_dist
                            if self.le.signed_dist(point, min_line.endpoints[0], min_line.normal) > 0:
                                min_positive_dist_per_room[label] = min_dist
                        if len(min_positive_dist_per_room) > 1:
                            room_label = min(min_positive_dist_per_room, key=min_positive_dist_per_room.get)
                        elif len(min_positive_dist_per_room) == 1:
                            room_label = next(iter(min_positive_dist_per_room.keys()))
                        else:
                            room_label = min(min_dist_per_room, key=min_dist_per_room.get)
                    else:
                        room_label = potential_rooms[0]  
        return room_label
    
    def nearest_lineseg(self, point, endpoints):
        """Determines the nearest line segment to a point

        Args:
            point (array): any point
            endpoints (list): endpoints of a line segment

        Returns:
            int: the index of the nearest line segment
            float: shortest distance to the nearest line segment
        """
        a = endpoints[:, 0, :]
        b = endpoints[:, 1, :] 
        ab = b - a
        ab2 = np.sum(ab**2, axis=-1)
        ax = point - a 
        abax = np.sum(ab * ax, axis=-1)
        t = np.clip(abax / ab2, 0.0, 1.0)
        xt = ax - np.expand_dims(t, axis=-1) * ab 
        xt2 = np.sum(xt**2, axis=-1)
        min_index = np.argmin(xt2)
        min_distance = np.sqrt(xt2[min_index])
        return min_index, min_distance
    
    def linesegs_within_range(self, lines, range):
        """Determines the line segments within range to the robot

        Args:
            lines (list): keys of line segments
            range (float): range from the robot

        Returns:
            list: line segments within range
        """
        endpoints = np.array([self.le.line_graph.nodes[line]["line"].endpoints for line in lines])
        a = endpoints[:, 0, :] 
        b = endpoints[:, 1, :] 
        ab = b - a  
        ab2 = np.sum(ab**2, axis=-1) 
        ap = self.robot_pos - a  
        abap = np.sum(ab * ap, axis=-1)  
        t = np.clip(abap / ab2, 0.0, 1.0) 
        pt = ap - t[:, np.newaxis] * ab  
        pt2 = np.sum(pt**2, axis=-1)
        within_range = list(np.array(lines)[pt2 <= range**2])
        return within_range
        
    def compute_weight(self, edge, distance=None, max_length=None, includeDist=True, 
                       includeRobotPos=True, includeLength=True):
        """Computes the edge weights of the visibility graph

        Args:
            edge (tuple): edge between two line segments
            distance (float, optional): shortest distance between two line segments. Defaults to None.
            max_length (float, optional): length of the longest line segment in the visibility graph. Defaults to None.
            includeDist (bool, optional): whether to include the distance-based weight. Defaults to True.
            includeRobotPos (bool, optional): whether to include the robot position-based weight. Defaults to True.
            includeLength (bool, optional): whether to include the length-based weight. Defaults to True.

        Returns:
            float: the combined edge weight
        """
        line1 = self.le.line_graph.nodes[edge[0]]["line"]
        line2 = self.le.line_graph.nodes[edge[1]]["line"]
        
        # Robot position weight
        if includeRobotPos:
            rob_dist = np.linalg.norm(line1.robot_pos - line2.robot_pos)
            robot_pos_weight = np.exp(-self.gamma_rob * rob_dist ** 2)
        else:
            robot_pos_weight = 1
        
        # Shortest distance weight
        if includeDist:
            dist_weight = np.exp(-self.gamma_dist * distance ** 2)
        else:
            dist_weight = 1
            
        # Length based weight
        if includeLength:
            length1 = np.linalg.norm(np.diff(line1.endpoints, axis=0))
            length2 = np.linalg.norm(np.diff(line2.endpoints, axis=0))
            length_weight = (length1 + length2) / (2 * max_length)
        else:
            length_weight = 1 

        return length_weight * dist_weight * robot_pos_weight

    def get_line_data(self, p, line_data, visibility_graph):
        """Stores line data needed for the visibility graph construction to be reused.

        Args:
            p (int): key of a line segment
            line_data (dict): data of the line segments in the visibility graph
            visibility_graph (networkx graph): visibility graph

        Returns:
            list: line data for a single line segment
        """
        if p not in line_data:
            p1, p2 = visibility_graph.nodes[p]["line"].endpoints
            p_center = np.array([(p1[0] + p2[0])/2, (p1[1]+p2[1])/2]) 
            p1_quarter = (p_center + p1)/2
            p2_quarter = (p_center + p2)/2
            linestring = LineString([p1,p2])
            p_normal = visibility_graph.nodes[p]["line"].normal 
            line_data[p] = [p_center, [p1_quarter, p2_quarter], linestring, p_normal, [p1,p2]]
        return line_data[p]
    
    def create_visibility_graph(self, local_lines):
        """Create the visibility graph for the local line segments

        Args:
            local_lines (list): local line segments

        Returns:
            networkx graph: visibility graph
        """
        visibility_graph = self.le.line_graph.subgraph(local_lines).to_undirected()
        max_length = max([np.linalg.norm(np.diff(visibility_graph.nodes[line]["line"].endpoints, axis=0)) for line in visibility_graph.nodes()])
        for edge in visibility_graph.edges():
            visibility_graph.edges[edge[0],edge[1]]["weight"] = self.compute_weight(edge, max_length=max_length, includeLength=True, includeDist=False)
        line_data = dict()
        old_lines = local_lines
        #viewed_set = set(self.le.viewed_lines)
        #current_lines = self.room_graph.nodes[self.current_room]["lines"]
        #old_current_lines = list(set(current_lines)-viewed_set)
        #current_visibility_graph = self.room_graph.nodes[self.current_room]["visibility_graph"]

        # viewed_pairs = list(combinations(self.le.viewed_lines, 2))
        # for pair in viewed_pairs:
        #     if not visibility_graph.has_edge(*pair):
        #         line1, line2 = pair
        #         data1 = self.get_line_data(line1, line_data, visibility_graph)
        #         data2 = self.get_line_data(line2, line_data, visibility_graph)
        #         dist = data1[2].distance(data2[2])
        #         weight = self.compute_weight((line1, line2), dist, max_length, includeRobotPos=False)
        #         visibility_graph.add_edge(line1, line2, weight=weight)
        
        # if len(old_current_lines) > 0 and current_visibility_graph is not None:
        #     old_current_subgraph = current_visibility_graph.subgraph(old_current_lines)
        #     old_current_edges = old_current_subgraph.edges(data=True)
        #     for edge in old_current_edges:
        #         if not visibility_graph.has_edge(edge[0],edge[1]):
        #             visibility_graph.add_edges_from([edge])
        # if len(self.local_rooms) > 1:
        #     for room in self.local_rooms[1:]:
        #         other_visibility_graph = self.room_graph.nodes[room]["visibility_graph"]
        #         if other_visibility_graph is not None:
        #             room_lines = self.room_graph.nodes[room]["lines"]
        #             old_lines = list(set(room_lines)-viewed_set)
        #             if len(old_lines) > 0:
        #                 old_subgraph = other_visibility_graph.subgraph(old_lines)
        #                 visibility_graph.add_edges_from(old_subgraph.edges(data=True))            
        # if len(old_lines) == 0:
        #     return visibility_graph

        for i in range(visibility_graph.number_of_nodes()):
            p = local_lines[i] 
            p_center, [p1_quarter, p2_quarter], linestring1, p_normal, [p1,p2] = self.get_line_data(p, line_data, visibility_graph)
            edge_candidates = dict()
            intersection_candidates = dict()
            for j in range(len(old_lines)):
                q = old_lines[j] 
                if p != q:
                    q_center, [q1_quarter, q2_quarter], linestring2, q_normal, [q1,q2] = self.get_line_data(q, line_data, visibility_graph)
                    dist = linestring1.distance(linestring2)
                    if dist < self.vis_dist_thresh:
                        if dist <= self.le.door_width_interval[1] and not visibility_graph.has_edge(p, q) and np.dot(q_normal, p_normal) >= 0:
                            line_dist = self.le.point2line_dist(p_center, visibility_graph.nodes[q]["line"].r, visibility_graph.nodes[q]["line"].theta)
                            angle_d = np.fabs(visibility_graph.nodes[p]["line"].theta - visibility_graph.nodes[q]["line"].theta)
                            if (angle_d < np.pi/4 or angle_d > (3/4*np.pi)) and line_dist < 0.5:
                                pair = None
                                if self.le.signed_dist(q1, p2, visibility_graph.nodes[p]["line"].dir) > 0:
                                    pair = (p,q)
                                elif self.le.signed_dist(p1, q2, visibility_graph.nodes[q]["line"].dir) > 0:
                                    pair = (q,p)
                                if pair is not None:
                                    isFree = True
                                    if len(self.le.line_graph.out_edges(pair[0])) > 0:
                                        other_line1 = list(self.le.line_graph.out_edges(pair[0]))[0][1]
                                        d1 = self.le.signed_dist(self.le.line_graph.nodes[other_line1]["line"].endpoints[1], visibility_graph.nodes[pair[0]]["line"].endpoints[1], visibility_graph.nodes[pair[0]]["line"].normal)
                                        if d1 < 0:
                                            isFree = True
                                        else:
                                            isFree = False
                                    if len(self.le.line_graph.in_edges(pair[1])) > 0:
                                        other_line2 = list(self.le.line_graph.in_edges(pair[1]))[0][0]
                                        d2 = self.le.signed_dist(self.le.line_graph.nodes[other_line2]["line"].endpoints[0], visibility_graph.nodes[pair[1]]["line"].endpoints[0], visibility_graph.nodes[pair[1]]["line"].normal)
                                        if d2 < 0:
                                            isFree = True
                                        else:
                                            isFree = False
                                    if isFree:
                                        visibility_graph.add_edge(p, q, weight=1)
                                    continue
                        d1 = self.le.signed_dist(q1, p_center, p_normal)
                        d2 = self.le.signed_dist(q2, p_center, p_normal)
                        if d1 >= 0 or d2 >= 0:
                            intersection_candidates[q] = dist
                            if not visibility_graph.has_edge(p, q):
                                d1 = self.le.signed_dist(q_center, p_center, p_normal)
                                d2 = self.le.signed_dist(p_center, q_center, q_normal)
                                if d1 >= 0 and d2 >= 0:
                                    edge_candidates[q] = dist
            if len(edge_candidates) > 0:
                for c in edge_candidates:
                    c_data = line_data[c]
                    center = c_data[0]
                    quarter1, quarter2 = c_data[1]
                    dist = edge_candidates[c]
                    closer_candidates = {key: val for key, val in intersection_candidates.items() if val < dist}
                    lines_between = [LineString([p_center, center]), LineString([quarter1, p2_quarter]), LineString([quarter2, p1_quarter])]
                    if not np.any([np.any([line_between.intersects(line_data[key][2]) for line_between in lines_between]) for key in closer_candidates]):
                        weight = self.compute_weight((p, c), dist, max_length)
                        visibility_graph.add_edge(p, c, weight=weight) 
        if not nx.is_connected(visibility_graph):
            self.connect_components(visibility_graph)
        return visibility_graph
    
    def connect_components(self, visibility_graph):
        """Connect disconnected components in the visibility graph

        Args:
            visibility_graph (networkx graph): visibility graph
        """
        components = list(map(list, nx.connected_components(visibility_graph)))
        min_numbers = [min(line for line in component) 
                    for component in components]
        # Start with the component that has the smallest min detection number
        while len(components) > 1:
            # Get component with smallest min detection number
            i = min_numbers.index(min(min_numbers))
            component_i = components[i]
            min_number_i = min_numbers[i]

            # Find component with closest larger detection number
            closest_j = None
            closest_number = float('inf')
            for j in range(len(components)):
                if j == i: 
                    continue
                min_number_j = min(line for line in components[j])
                if min_number_i < min_number_j < closest_number:
                    closest_number = min_number_j
                    closest_j = j

            # Add edge between components i and j
            line_i = min((line for line in component_i), key=lambda line: abs(line - closest_number))
            line_j = min((line for line in components[closest_j]), key=lambda line: abs(line - line_i))
            visibility_graph.add_edge(line_i, line_j, weight=1e-10)

            # Merge components i and j
            components[i] += components[closest_j]
            components.pop(closest_j)
            min_numbers[i] = min(min_numbers[i], min_numbers[closest_j])
            min_numbers.pop(closest_j)
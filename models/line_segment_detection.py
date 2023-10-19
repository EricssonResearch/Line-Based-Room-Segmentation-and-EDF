import math
import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import LineString
import copy
import collections
import networkx as nx
import scipy


def jacobian(p, x, y):
    """Jacobian of the polar line equation
    """
    df2 = y*np.cos(p[1]) - x*np.sin(p[1])
    df1 = -np.ones(df2.shape)
    jac = np.column_stack((df1, df2))
    return jac

def polar_func(p, x, y):
    """ p = [r, theta] 
    """
    d_sq = np.cos(p[1])*x + np.sin(p[1])*y - p[0]
    return d_sq

class LineSeg:
    """Class for attributes of each line segment"""
    def __init__(self):
        self.r = 0.0
        self.theta = 0.0
        self.normal = None
        self.dir = None
        self.end_keys = [None, None]
        self.endpoints = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
        self.point_keys = [] 
        self.isUpdated = True
        self.weight = 1
        self.robot_pos = None
        self.isShort = False
        self.cluster_number = None
        
class Least():
    """Class for least squares result"""
    def __init__(self):
        self.r = 0.0
        self.theta = 0.0
        self.cost = 0.0
        
class LineSegParams:
    """Class for line segment parameters"""
    def __init__(self, least_thresh=0.03, min_line_length=0.6, 
                 min_line_points=6, line_seed_points=3):
        self.least_thresh = least_thresh
        self.min_line_length = min_line_length
        self.min_line_points = min_line_points
        self.line_seed_points = line_seed_points
        

class LineSegDetection:
    """Class for the line segment detection module"""
    def __init__(self, k=7, least_thresh=0.05, min_line_length=0.5, 
                 min_line_points=25, line_seed_points=25, corner_thresh=0.5,
                 door_width_interval=[1.0, 3.5], gap_thresh=0.5, delta_d=0.2):
        self.params_ = LineSegParams(least_thresh, min_line_length, 
                                     min_line_points, line_seed_points)
        
        self.door_width_interval = door_width_interval  # Interval of potential doorway widths
        self.corner_thresh = corner_thresh  # Distance threshold to form a corner between two line segments
        self.gap_thresh = gap_thresh  # Distance threshold of maximum gap width to join two collinear line segments
        self.delta_d = delta_d  # Maximum orthogonal distance to mergetwo line segments
        self.k = k  # Magnification factor for adaptive breakpoint detection
        
        self.points = dict()
        self.angles = dict()
        self.point_num_ = 0
        self.m_least = Least()
        self.next_line = 0
        self.old_lines = []
        self.new_lines = []

        self.next_point = 0
        self.new_remaining_points = dict()
        self.robot_pos = None
        self.num_lines = 0 
        self.current_room = 0
        self.room_graph = None
        self.found_new_lines = False
        self.line_graph = nx.DiGraph()
        
        self.lineseg_clusters = []
        self.point_clusters = []
        self.remaining_cluster_points = collections.defaultdict(dict)
        self.short_lines = []
        self.isUpdated = False
        self.line_processing = False
        self.viewed_lines = []
                
    def change_endpoint(self, line, end_index, point):
        """Change one of the endpoints of a line segment.

        Args:
            line (int): key of line segment
            end_index (int): index of one of the line segment's endpoints 
            point (array): new point to replace the endpoint at end_index
        """
        self.line_graph.nodes[line]["line"].endpoints[end_index] = point
        
    def remove_line(self, line):
        """Remove a previously detected line segment

        Args:
            line (int): key of line segment
        """
        if line in self.short_lines:
            self.short_lines.remove(line)
        if line in self.new_lines:
            self.new_lines.remove(line)
        else:
            self.old_lines.remove(line)
        if line in self.viewed_lines:
            self.viewed_lines.remove(line)
        self.room_graph.nodes[self.line_graph.nodes[line]["label"]]["lines"].remove(line)
        self.line_graph.remove_node(line)
        if line in self.local_lines:
            self.local_lines.remove(line)
        
    def add_line(self, line, room_label):
        """Add a new line segment

        Args:
            line (int): key of line segment
            room_label (int): room label of the line segment
        """
        self.new_lines.append(self.next_line)
        self.line_graph.add_node(self.next_line, line=line, label=room_label)
        self.room_graph.nodes[room_label]["lines"].append(self.next_line)
        self.viewed_lines.append(self.next_line)
        self.next_line += 1
    
    def leastsquare(self, seed_point_keys, guess):
        """Perform least squares to extract line parameters

        Args:
            seed_point_keys (list): the keys of the seed points
            guess (list): initial guess for line parameters

        Returns:
            Least: Least object
        """
        least_params = Least()
        X = np.zeros(len(seed_point_keys))
        Y = np.zeros(len(seed_point_keys))
        for i, key in enumerate(seed_point_keys):
            X[i] = self.points[key][0]
            Y[i] = self.points[key][1]
        polar_lsq = least_squares(polar_func, guess, jac=jacobian, args=[X, Y], method='dogbox', loss='linear', bounds=[(-math.inf, 0), (math.inf, np.pi)]) 
        least_params.r = polar_lsq.x[0]
        least_params.theta = polar_lsq.x[1]
        least_params.cost = polar_lsq.cost
        return least_params
    
    def create_normal(self, point, endpoints):
        """Creates the normal vector of a line segment

        Args:
            point (array): point to which the normal should point
            endpoints (list): endpoints of a line segment

        Returns:
            array: normal vector
        """
        p1 = endpoints[0]
        p2 = endpoints[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if (point[0]-p1[0])*dy - (point[1]-p1[1])*dx > 0:
            normal = np.array([dy, -dx])
        else:
            normal = np.array([-dy, dx])
        return normal/np.linalg.norm(normal)
    
    def seg2seg_dist(self, endpoints1, endpoints2):
        """Computes shortest distance between two line segments

        Args:
            endpoints1 (array): endpoints of line segment 1
            endpoints2 (array): endpoints of line segment 2

        Returns:
            float: distance
        """
        line1 = LineString(endpoints1)
        line2 = LineString(endpoints2)
        return line1.distance(line2)
    
    def point2seg_dist(self, point, endpoints):
        """Computes shortest distance between a point and a line segment

        Args:
            point (array): any point
            endpoints (list): endpoints of the line segment

        Returns:
            float: distance
        """
        d = np.divide(endpoints[1] - endpoints[0], np.linalg.norm(endpoints[1] - endpoints[0]))
        s = np.dot(endpoints[0] - point, d)
        t = np.dot(point - endpoints[1], d)
        h = np.maximum.reduce([s, t, 0])
        c = np.cross(point - endpoints[0], d)
        return abs(np.hypot(h, c))
    
    def point2line_dist(self, point, r, theta):
        """Computes orthogonal distance from a point to a line

        Args:
            point (array): any point
            r (float): orthogonal distance from line to origin
            theta (float): angle of the normal of the line

        Returns:
            float: distance
        """
        a = np.cos(theta)
        b = np.sin(theta)
        c = -r
        return np.abs(a*point[0] + b*point[1] + c)
    
    def signed_dist(self, point, line_point, vector):
        """Computes the signed distance between a line and a point

        Args:
            point (array): any point
            line_point (array): any point on the line
            vector (array): vecor determining the sign

        Returns:
            float: signed distance
        """
        return (point[0]-line_point[0])*vector[0] + (point[1]-line_point[1])*vector[1]
    
    def isLine(self, seed_point_keys):
        """Checks if the line fitted to the seed points represent them well

        Args:
            seed_point_keys (list): seed point keys

        Returns:
            bool: true if fitted line represent seed points well, false otherwise
        """
        flag = False
        for k in seed_point_keys:
            error = self.point2line_dist(self.points[k], self.m_least.r, self.m_least.theta)
            if error > self.params_.least_thresh:
                flag = True
                break
        return not flag
    
    def grow_line_seed(self, start_index, end_index, cluster):
        """Extends the line seed with one point at a time

        Args:
            start_index (int): index of first point of line seed
            end_index (int): index of last point of line seed
            cluster (list): clusters of successive points
        """
        new_line = LineSeg()

        flag = True
        theta = self.m_least.theta
        r = self.m_least.r

        i = end_index+1
        if end_index == len(cluster)-1:
            end = cluster[end_index]
            flag = False
        
        point_keys = cluster[start_index:i]
        while(flag):
            end = cluster[i]
            point = self.points[end]
            dist = self.point2line_dist(point, r, theta)
            if dist < self.params_.least_thresh: 
                point_keys.append(end)
                if end == cluster[-1]:
                    flag = False
                else:
                    i += 1
            else:
                guess = [r, theta] 
                self.m_least = self.leastsquare(point_keys, guess)
                theta = self.m_least.theta
                r = self.m_least.r
                dist = self.point2line_dist(point, r, theta)
                if dist < self.params_.least_thresh: 
                    point_keys.append(end)
                    if end == cluster[-1]:
                        flag = False
                    else:
                        i += 1
                else:
                    end = cluster[i-1]
                    flag = False
        isDetected = False
        start = cluster[start_index]
        line_length = np.linalg.norm(self.points[end] - self.points[start])
        isShort = False
        line_include = i-1
        if line_length >= 0.4 and len(point_keys) >= self.params_.min_line_points: 
                isDetected = True
                new_line.end_keys = [start, end]
                guess = [r, theta] 
                new_line.point_keys = point_keys
                m_result = self.leastsquare(new_line.point_keys, guess)
                new_line.theta = m_result.theta
                new_line.r = m_result.r
                new_line.endpoints[0] = self.project(self.points[start], [m_result.r, m_result.theta])
                new_line.endpoints[1] = self.project(self.points[end], [m_result.r, m_result.theta])
                new_line.normal = self.create_normal(self.robot_pos, new_line.endpoints)
                new_line.dir = new_line.endpoints[1] - new_line.endpoints[0]
                new_line.dir /= np.linalg.norm(new_line.dir)
                new_line.robot_pos = self.robot_pos
                if line_length < self.params_.min_line_length:
                    self.short_lines.append(self.next_line)
                    new_line.isShort = True
                    isShort = True
                self.add_line(new_line, self.current_room)
                  
        return line_include, isDetected, isShort, line_length, point_keys
    
    def merge_graph_nodes(self, node, new_nodes):
        """Transfers edges of in and out of new_nodes to node

        Args:
            node (int): key of old line segment
            new_nodes (list): list of keys of new line segments
        """
        if len(self.line_graph.in_edges(node)) == 0:
            new_in_edges = list(self.line_graph.in_edges(new_nodes))
            if len(new_in_edges) > 0:
                node_end1 = self.line_graph.nodes[node]["line"].endpoints[0] 
                end_distances = [np.linalg.norm(node_end1-self.line_graph.nodes[edge[0]]["line"].endpoints[1]) for edge in new_in_edges] 
                min_dist_index = np.argmin(end_distances)
                if end_distances[min_dist_index] < 0.3:
                    self.line_graph.add_edge(new_in_edges[min_dist_index][0], node, type=self.line_graph.edges[new_in_edges[min_dist_index]]["type"])
        
        if len(self.line_graph.out_edges(node)) == 0:
            new_out_edges = list(self.line_graph.out_edges(new_nodes))
            if len(new_out_edges) > 0:
                node_end2 = self.line_graph.nodes[node]["line"].endpoints[1] 
                end_distances = [np.linalg.norm(node_end2-self.line_graph.nodes[edge[1]]["line"].endpoints[0]) for edge in new_out_edges] 
                min_dist_index = np.argmin(end_distances)
                if end_distances[min_dist_index] < 0.3:
                    self.line_graph.add_edge(node, new_out_edges[min_dist_index][1], type=self.line_graph.edges[new_out_edges[min_dist_index]]["type"])
            
    def identify_overlap(self, p, q):
        """Identifies overlaps between line segments

        Args:
            p (int): key of a line segment
            q (int): key of a line segment

        Returns:
            bool: true if overlap, false otherwise
        """
        isOverlap = False
        q_line = self.line_graph.nodes[q]["line"]
        p_line = self.line_graph.nodes[p]["line"]
        q_endpoints = q_line.endpoints
        p_endpoints = p_line.endpoints
        dist = self.seg2seg_dist(p_endpoints, q_endpoints)
        angle_d = np.fabs(p_line.theta - q_line.theta)
        if (angle_d < np.pi/5 or angle_d > (4/5*np.pi)) and np.dot(p_line.normal, q_line.normal) > 0 and dist < self.delta_d:
            d_dir_p1q1 = self.signed_dist(q_endpoints[0], p_endpoints[0], p_line.dir)
            d_dir_p2q2 = self.signed_dist(q_endpoints[1], p_endpoints[1], p_line.dir)
            if d_dir_p1q1 > 0:
                d_dir_21 = self.signed_dist(q_endpoints[0], p_endpoints[1], p_line.dir)
                d_dir_12 = self.signed_dist(q_endpoints[1], p_endpoints[0], p_line.dir)
                d_dir_22 = d_dir_p2q2
            else:
                d_dir_21 = self.signed_dist(p_endpoints[0], q_endpoints[1], q_line.dir)
                d_dir_12 = self.signed_dist(p_endpoints[1], q_endpoints[0], q_line.dir)
                d_dir_22 = -d_dir_p2q2
            if (d_dir_21 < 0 or (d_dir_12 > 0 and d_dir_22 < 0)):
                isOverlap = True
                self.line_graph.nodes[q]["line"].robot_pos = self.robot_pos
        return isOverlap
    
    def merge_lines(self, overlapping_lines):
        """Merges line segments in overlapp_lines

        Args:
            overlapping_lines (list): list of line segments
        """
        endpoints = np.array([self.line_graph.nodes[p]["line"].endpoints for p in overlapping_lines])
        line_lengths = scipy.spatial.distance.cdist(endpoints[:,0], endpoints[:,1])
        max_index = np.unravel_index(np.argmax(line_lengths), line_lengths.shape)
        _left_line = overlapping_lines[max_index[0]]
        _right_line = overlapping_lines[max_index[1]]
        main_line = min(overlapping_lines)
        
        if len(overlapping_lines) > 2:
            old_lines = [line for line in overlapping_lines if line < self.new_lines[0]]
            if len(old_lines) > 1: 
                num_old_edges = sum(len(self.line_graph.in_edges(line))+len(self.line_graph.out_edges(line)) for line in old_lines)
                old_endpoints = np.array([self.line_graph.nodes[line]["line"].endpoints for line in old_lines])
                old_line_lengths = scipy.spatial.distance.cdist(old_endpoints[:,0], old_endpoints[:,1])
                old_max_index = np.unravel_index(np.argmax(old_line_lengths), old_line_lengths.shape)
                left_old = old_lines[old_max_index[0]]
                right_old = old_lines[old_max_index[1]]
                if _left_line > self.new_lines[0] or _right_line > self.new_lines[0]:
                    if  _left_line > self.new_lines[0] and len(self.line_graph.in_edges(left_old)) == 0:
                        left_old_line = self.line_graph.nodes[left_old]["line"]
                        left_old_line.endpoints[0] = self.project(self.line_graph.nodes[_left_line]["line"].endpoints[0], [left_old_line.r, left_old_line.theta])
                    if _right_line > self.new_lines[0] and len(self.line_graph.out_edges(right_old)) == 0:
                        right_old_line = self.line_graph.nodes[right_old]["line"]
                        right_old_line.endpoints[1] = self.project(self.line_graph.nodes[_right_line]["line"].endpoints[1], [right_old_line.r, right_old_line.theta])
                    self.remove_line(overlapping_lines[0])
                    return main_line, False, overlapping_lines
                if (num_old_edges > 0 and _left_line == _right_line) or(len(self.line_graph.out_edges(left_old)) > 0) or (len(self.line_graph.out_edges(right_old)) > 0):
                    self.remove_line(overlapping_lines[0])
                    return main_line, False, overlapping_lines
        weights = []
        sum_x = 0
        sum_y = 0
        for p in overlapping_lines:
            theta = self.line_graph.nodes[p]["line"].theta
            r = self.line_graph.nodes[p]["line"].r
            x_p = r*np.cos(theta)
            y_p = r*np.sin(theta)
            weight = self.line_graph.nodes[p]["line"].weight
            sum_x += weight*x_p
            sum_y += weight*y_p
            weights.append(weight)
        mean_x = sum_x/sum(weights)
        mean_y = sum_y/sum(weights)
        
        new_theta = np.arctan2(mean_y, mean_x)
        new_r = np.sqrt(mean_x**2+mean_y**2)
        if new_theta < 0:
            new_r = -new_r
            new_theta += np.pi
        elif new_theta > np.pi:
            new_r = -new_r
            new_theta -= np.pi
        
        isOldUpdated = False
        highest_weight = max(weights)
        if main_line != _left_line or main_line != _right_line:
            self.isUpdated = True
            isOldUpdated = True
            self.line_graph.nodes[main_line]["line"].isUpdated = True
        
        overlapping_lines.remove(main_line)
        for l in overlapping_lines:
            self.line_graph.nodes[main_line]["line"].point_keys += self.line_graph.nodes[l]["line"].point_keys
        
        self.line_graph.nodes[main_line]["line"].r = new_r
        self.line_graph.nodes[main_line]["line"].theta = new_theta
        self.line_graph.nodes[main_line]["line"].weight = highest_weight + 1

        new_p1 = self.project(self.line_graph.nodes[_left_line]["line"].endpoints[0], [new_r, new_theta])
        self.change_endpoint(main_line, 0, new_p1)
        new_p2 = self.project(self.line_graph.nodes[_right_line]["line"].endpoints[1], [new_r, new_theta])
        self.change_endpoint(main_line, 1, new_p2)
       
        self.merge_graph_nodes(main_line, overlapping_lines)
        for p in overlapping_lines:
            self.remove_line(p)
        return main_line, isOldUpdated, overlapping_lines

    def clean_line_segments(self):
        """Compares new line segments with old lines segments and 
        merges those that meet the conditions.
        """
        num_new = len(self.new_lines)      
        updated_old_lines = dict() 
        if num_new > 0 and self.num_lines > 0: 
            i = 0
            while i < len(self.lineseg_clusters):
                cluster = self.lineseg_clusters[i]
                j = 0
                while j < len(cluster):
                    p = cluster[j]
                    overlapping_lines = [p]
                    new_overlap = False
                    for q in self.local_lines:
                        isOverlap = self.identify_overlap(p, q)
                        if isOverlap:
                            new_overlap = True
                            overlapping_lines.append(q)
                    if new_overlap:
                        merged_line, isOldUpdated, removed_lines = self.merge_lines(overlapping_lines)
                        if (len(self.new_lines) == 0 or merged_line < self.new_lines[0]):
                            if isOldUpdated and merged_line not in updated_old_lines:
                                self.viewed_lines.append(merged_line) 
                                self.lineseg_clusters[i][j] = merged_line
                                updated_old_lines[merged_line] = i
                                for line in removed_lines:
                                    if line in updated_old_lines:
                                        self.lineseg_clusters[updated_old_lines[line]].remove(line)
                                        j -= 1
                            else:
                                del self.lineseg_clusters[i][j]
                                j -= 1
                    j += 1
                i += 1
        self.lineseg_clusters = list(filter(None, self.lineseg_clusters))
        if len(self.new_lines) > 0:
            self.isUpdated = True
    
    def join_corner(self, p, q):
        """Joins two line segments, p and q, at their point of intersection
        to form a corner and creates an edges between them in the line_graph. 

        Args:
            p (int): key of line segment
            q (int): key of line segment

        Returns:
            bool: true if line segments were joined, otherwise false
        """
        p_endpoints = self.line_graph.nodes[p]["line"].endpoints
        q_endpoints = self.line_graph.nodes[q]["line"].endpoints 
        if np.linalg.norm(p_endpoints[1] - q_endpoints[0]) > self.corner_thresh:
            return False

        r_p = self.line_graph.nodes[p]["line"].r
        theta_p = self.line_graph.nodes[p]["line"].theta
        r_q = self.line_graph.nodes[q]["line"].r
        theta_q = self.line_graph.nodes[q]["line"].theta
        intersection = self.intersection([r_p, theta_p], [r_q, theta_q])
        if np.linalg.norm(intersection - q_endpoints[0]) < 0.5 or np.linalg.norm(intersection - q_endpoints[1]) < 0.5:
            return False

        d1 = self.signed_dist(q_endpoints[1], intersection, self.line_graph.nodes[p]["line"].normal)
        d2 = self.signed_dist(p_endpoints[0], intersection, self.line_graph.nodes[q]["line"].normal) 
        if d1*d2 < 0:
            return False
        
        out_edge = list(self.line_graph.out_edges(p))
        in_edge = list(self.line_graph.in_edges(q))
        
        if (len(in_edge) != 0 or len(out_edge) != 0):
            if d1 < 0 or d2 < 0:
                return False
        
            if len(out_edge) != 0 and self.line_graph.edges[out_edge[0]]["type"] == "corner":
                d_out = self.signed_dist(self.line_graph.nodes[out_edge[0][1]]["line"].endpoints[1], intersection, self.line_graph.nodes[p]["line"].normal)
                if d_out > 0:
                    return False
            if len(in_edge) != 0 and self.line_graph.edges[in_edge[0]]["type"] == "corner":
                d_in = self.signed_dist(self.line_graph.nodes[in_edge[0][0]]["line"].endpoints[0], intersection, self.line_graph.nodes[q]["line"].normal)
                if d_in > 0:
                    return False
        
        self.change_endpoint(p, 1, intersection)
        self.change_endpoint(q, 0, intersection)
        if not self.line_graph.has_edge(p,q):
            self.line_graph.add_edge(p,q,type='corner')
        self.line_graph.remove_edges_from(in_edge + out_edge)
        return True

    def intersection(self, line_params1, line_params2):
        """Computes the point of intersection between two lines.

        Args:
            line_params1 (list): parameters for line 1
            line_params2 (list): parameters for line 2

        Returns:
            array: point of intersection
        """
        r1, theta1 = line_params1
        r2, theta2 = line_params2
        intersection = np.zeros(2)
        intersection[0] = (r1*np.sin(theta2)-r2*np.sin(theta1))/np.sin(theta2-theta1)
        intersection[1] = (r1*np.cos(theta2)-r2*np.cos(theta1))/np.sin(theta1-theta2)
        return intersection
    
    def project(self, point, line_params):
        """projects a point onto a line

        Args:
            point (array): any point
            line_params (list): parameters of a line

        Returns:
            array: projection of point
        """
        r, theta = line_params
        p_proj = np.empty(2)
        p_proj[0]  = np.sin(theta)*(np.sin(theta)*point[0] - np.cos(theta)*point[1]) + np.cos(theta)*r
        p_proj[1]  = np.cos(theta)*(-np.sin(theta)*point[0] + np.cos(theta)*point[1]) + np.sin(theta)*r
        return p_proj
    
    def split_and_join(self, p, q):
        """Splits a line segment into two at the point of intersection 
        with a non-parallel line segmen to form a corner.

        Args:
            p (int): key of a line segment
            q (int): key of a line segment

        Returns:
            bool: true if line segment was split, otherwise false
        """
        line_p = self.line_graph.nodes[p]["line"]
        line_q = self.line_graph.nodes[q]["line"]
        
        join = False
        p_endpoints = line_p.endpoints 
        q_endpoints = line_q.endpoints 
        
        intersection = self.intersection([line_p.r, line_p.theta], [line_q.r, line_q.theta])
        min_index = np.argmin([np.linalg.norm(intersection - point) for point in p_endpoints + q_endpoints])
        if min_index < 2:
            p_free = [len(self.line_graph.in_edges(p)) == 0, len(self.line_graph.out_edges(p)) == 0] 
            if p_free[min_index]:
                d1_dir = self.signed_dist(intersection, q_endpoints[0], line_q.dir) 
                d2_dir = self.signed_dist(intersection, q_endpoints[1], line_q.dir) 
                if d1_dir > 0 and d2_dir < 0:
                    dist_q1 = np.linalg.norm(intersection - q_endpoints[0]) 
                    dist_q2 = np.linalg.norm(intersection - q_endpoints[1])
                    if dist_q1 >= 0.5 and dist_q2 >= 0.5:
                        other = p_endpoints[int(not min_index)]
                        d_split = self.signed_dist(other, intersection, line_q.normal)
                        if d_split > 0:
                            key_join = p
                            key_split = q
                            join_index = min_index
                            join = True
        else:
            min_index -= 2
            q_free = [len(self.line_graph.in_edges(q)) == 0, len(self.line_graph.out_edges(q)) == 0] 
            if q_free[min_index]:
                d1_dir = self.signed_dist(intersection, p_endpoints[0], line_p.dir) 
                d2_dir = self.signed_dist(intersection, p_endpoints[1], line_p.dir) 
                if d1_dir > 0 and d2_dir < 0:
                    dist_p1 = np.linalg.norm(intersection - p_endpoints[0])
                    dist_p2 = np.linalg.norm(intersection - p_endpoints[1])
                    if dist_p1 >= 0.5 and dist_p2 >= 0.5:
                        other = q_endpoints[int(not min_index)]
                        d_split = self.signed_dist(other, intersection, line_p.normal)
                        if d_split > 0:
                            key_join = q
                            key_split = p
                            join_index = min_index
                            join = True

        if join: 
            self.change_endpoint(key_join, join_index, intersection)
            split_line = self.line_graph.nodes[key_split]["line"]
            split_index = np.argmin([np.linalg.norm(split_line.robot_pos - endpoint) for endpoint in split_line.endpoints])
            new_line = LineSeg()
            new_line.endpoints[split_index] = intersection
            new_line.endpoints[int(not split_index)] = split_line.endpoints[int(not split_index)]
            new_line.r = split_line.r
            new_line.theta = split_line.theta
            new_line.normal = split_line.normal
            new_line.isUpdated = split_line.isUpdated
            
            new_line.dir = split_line.dir
            new_line.robot_pos = split_line.robot_pos

            self.change_endpoint(key_split, int(not split_index), intersection)
        
            if split_index == 0:
                split_edges = list(self.line_graph.out_edges(key_split))
            else:
                split_edges = list(self.line_graph.in_edges(key_split))
            for edge in split_edges:
                type = self.line_graph.edges[edge[0], edge[1]]["type"]
                self.line_graph.remove_edge(*edge)
                list(edge)[int(not split_index)] = self.next_line
                self.line_graph.add_edge(*edge, type=type)

            d_join = self.signed_dist(self.line_graph.nodes[key_split]["line"].endpoints[0], intersection, self.line_graph.nodes[key_join]["line"].normal)
            if d_join > 0:
                if join_index == 1:
                    self.line_graph.add_edge(key_join, key_split, type='corner')
                else:
                    self.line_graph.add_edge(key_split, key_join, type='corner')
            else:
                if join_index == 1:
                    self.line_graph.add_edge(key_join, self.next_line, type='corner')
                else:
                    self.line_graph.add_edge(self.next_line, key_join, type='corner')
 
            self.add_line(new_line, self.line_graph.nodes[key_split]["label"]) 
        return join
    
    def detect_doors(self, p, q):
        """Splits a line segment into two at the point of intersection 
        with a non-parallel line segment where a potential doorway might be located.

        Args:
            p (int): key of a line segment
            q (int): key of a line segment

        Returns:
            bool: true if line segment was split, otherwise false
        """
        isDoor = False
        key_split = p
        p_endpoints = self.line_graph.nodes[p]["line"].endpoints 
        q_endpoints = self.line_graph.nodes[q]["line"].endpoints 
        line_q = self.line_graph.nodes[q]["line"]
        d1 = self.signed_dist(p_endpoints[0], q_endpoints[0], line_q.normal) 
        d2 = self.signed_dist(p_endpoints[1], q_endpoints[0], line_q.normal) 
        if d1 > 0 and d2 > 0:
            p_center = (p_endpoints[0] + p_endpoints[1])/2
            d1_dir = self.signed_dist(p_center, q_endpoints[0], line_q.dir) 
            d2_dir = self.signed_dist(p_center, q_endpoints[1], line_q.dir) 
            if d1_dir > 0 and d2_dir < 0:
                projection = self.project(p_center, [line_q.r, line_q.theta])
                dist_q1 = np.linalg.norm(projection - q_endpoints[0])
                dist_q2 = np.linalg.norm(projection - q_endpoints[1])
                if dist_q1 >= 0.7 and dist_q2 >= 0.7:
                    isDoor = True
                    key_split = q
        else:  
            line_p = self.line_graph.nodes[p]["line"]
            d1 = self.signed_dist(q_endpoints[0], p_endpoints[0], line_p.normal) 
            d2 = self.signed_dist(q_endpoints[1], p_endpoints[0], line_p.normal) 
            if d1 > 0 and d2 > 0:
                q_center = (q_endpoints[0] + q_endpoints[1])/2
                d1_dir = self.signed_dist(q_center, p_endpoints[0], line_p.dir) 
                d2_dir = self.signed_dist(q_center, p_endpoints[1], line_p.dir) 
                if d1_dir > 0 and d2_dir < 0:
                    projection = self.project(q_center, [line_p.r, line_p.theta])
                    dist_p1 = np.linalg.norm(projection - p_endpoints[0])
                    dist_p2 = np.linalg.norm(projection - p_endpoints[1])
                    if dist_p1 >= 0.7 and dist_p2 >= 0.7:
                        isDoor = True
                        key_split = p      
            
            if isDoor:
                split_line = self.line_graph.nodes[key_split]["line"]
                split_index = np.argmin([np.linalg.norm(split_line.robot_pos - endpoint) for endpoint in split_line.endpoints])
                    
                new_line = LineSeg()
                new_line.endpoints[split_index] = projection
                new_line.endpoints[int(not split_index)] = split_line.endpoints[int(not split_index)]
                new_line.r = split_line.r
                new_line.theta = split_line.theta
                new_line.normal = split_line.normal
                new_line.isUpdated = split_line.isUpdated
                new_line.dir = split_line.dir
                new_line.robot_pos = split_line.robot_pos
            
                self.change_endpoint(key_split, int(not split_index), projection)

                if split_index == 0:
                    split_edges = list(self.line_graph.out_edges(key_split))
                else:
                    split_edges = list(self.line_graph.in_edges(key_split))
                for edge in split_edges:
                    type = self.line_graph.edges[edge[0], edge[1]]["type"]
                    self.line_graph.remove_edge(*edge)
                    list(edge)[int(not split_index)] = self.next_line
                    self.line_graph.add_edge(*edge, type=type)
                    
                new_edge = (key_split, self.next_line) if split_index == 0 else (self.next_line, key_split)
                self.line_graph.add_edge(*new_edge, type="parallel")    
                self.add_line(new_line, self.line_graph.nodes[key_split]["label"])  
        return isDoor, key_split

    def close_gap(self, p, q):
        """Merges two collinear line segments separated by a small gap.

        Args:
            p (int): key of a line segment
            q (int): key of a line segment
        """
        if len(self.line_graph.out_edges(p)) > 0 or len(self.line_graph.in_edges(q)) > 0:
            return False, p
        p_endpoints = self.line_graph.nodes[p]["line"].endpoints 
        q_endpoints = self.line_graph.nodes[q]["line"].endpoints 
        r_p = self.line_graph.nodes[p]["line"].r
        r_q = self.line_graph.nodes[q]["line"].r
        theta_p = self.line_graph.nodes[p]["line"].theta
        theta_q = self.line_graph.nodes[q]["line"].theta

        d_dir_p2q1 = self.signed_dist(q_endpoints[0], p_endpoints[1], self.line_graph.nodes[p]["line"].dir)
        if d_dir_p2q1 < 0 or np.linalg.norm(p_endpoints[1] - q_endpoints[0]) > self.gap_thresh or self.point2line_dist(p_endpoints[1], r_q, theta_q) > self.delta_d:
            return False, p

        weights = [self.line_graph.nodes[p]["line"].weight, self.line_graph.nodes[q]["line"].weight]
        mean_x = (weights[0]*r_p*np.cos(theta_p)+weights[1]*r_q*np.cos(theta_q))/sum(weights)
        mean_y = (weights[0]*r_p*np.sin(theta_p)+weights[1]*r_q*np.sin(theta_q))/sum(weights)
        
        new_theta = np.arctan2(mean_y, mean_x)
        new_r = np.sqrt(mean_x**2+mean_y**2)
        if new_theta < 0:
            new_r = -new_r
            new_theta += np.pi
        elif new_theta > np.pi:
            new_r = -new_r
            new_theta -= np.pi
        
        highest_weight = max(weights)
        
        keys = (p,q)
        index = np.argmin(keys)
        main_key = keys[index]
        other_key = keys[int(not index)]
        if self.line_graph.nodes[main_key]["line"].isShort:
            temp_key = main_key
            main_key = other_key
            other_key = temp_key
        
        self.line_graph.nodes[main_key]["line"].point_keys += self.line_graph.nodes[other_key]["line"].point_keys
        self.line_graph.nodes[main_key]["line"].r = new_r
        self.line_graph.nodes[main_key]["line"].theta = new_theta
        self.line_graph.nodes[main_key]["line"].endpoints[int(not index)] = self.line_graph.nodes[other_key]["line"].endpoints[int(not index)]
        self.line_graph.nodes[main_key]["line"].isUpdated = True
        self.line_graph.nodes[main_key]["line"].weight = highest_weight + 1
        new_p1 = self.project(self.line_graph.nodes[main_key]["line"].endpoints[0], [new_r, new_theta])
        self.change_endpoint(main_key, 0, new_p1)
        new_p2 = self.project(self.line_graph.nodes[main_key]["line"].endpoints[1], [new_r, new_theta])
        self.change_endpoint(main_key, 1, new_p2)

        self.merge_graph_nodes(main_key, [other_key])
        self.remove_line(other_key)     
     
    def process_line_segments(self): 
        """Process line segments to ensure meaningful rooms.
        """
        close_gap_new = []
        close_gap_old = []
        pre = -1
        new_set = set()
        for i, cluster in enumerate(self.lineseg_clusters):
            proceed_processing = False
            isJoined1 = False
            if len(self.lineseg_clusters) > 1:
                p_lines = [self.lineseg_clusters[pre][-1]] 
                if len(self.lineseg_clusters[pre]) > 1 and self.line_graph.nodes[p_lines[0]]["line"].isShort:
                    p_lines.append(self.lineseg_clusters[pre][-2])
                q_lines = [cluster[0]] 
                if len(cluster) > 1 and self.line_graph.nodes[q_lines[0]]["line"].isShort:
                    q_lines.append(cluster[1])
                for p in p_lines:
                    for q in q_lines:
                        line_p = self.line_graph.nodes[p]["line"]
                        line_q = self.line_graph.nodes[q]["line"]
                        angle_d = np.fabs(line_p.theta - line_q.theta)
                        if self.line_processing and np.pi/3 < angle_d < 2/3*np.pi and not (line_p.isShort or line_q.isShort):
                            isJoined1 = self.join_corner(p, q)
                            if isJoined1:
                                break
                        elif (angle_d < np.pi/5 or angle_d > (4/5*np.pi)) and not (line_p.isShort and line_q.isShort):
                            normal_d = np.dot(line_p.normal, line_q.normal)
                            if normal_d > 0:
                                close_gap_new.append((p,q))
                                proceed_processing = True
                                break
                    if proceed_processing or isJoined1:
                        break
                new_lines = p_lines + q_lines
            else:
                new_lines = cluster
            
            process_lines = [] 
            for line in new_lines:
                if line not in new_set:
                    process_lines.append(line)
                
            if not isJoined1 and process_lines:
                new_set.update(process_lines)
                for r in process_lines:
                    for s in self.local_lines:
                        if s != r and s not in process_lines:
                            pair = (r,s) if self.signed_dist(self.line_graph.nodes[s]["line"].endpoints[0], self.line_graph.nodes[r]["line"].endpoints[0], self.line_graph.nodes[r]["line"].dir) > 0 else (s,r)
                            line1 = self.line_graph.nodes[pair[0]]["line"]
                            line2 = self.line_graph.nodes[pair[1]]["line"]
                            out_edge =list(self.line_graph.out_edges(pair[0]))
                            in_edge = list(self.line_graph.in_edges(pair[1]))
                            angle_d = np.fabs(line1.theta - line2.theta)
                            if self.line_processing and (angle_d > np.pi/3) and (angle_d < 2/3*np.pi) and not self.line_graph.nodes[r]["line"].isShort:
                                isJoined2 = self.join_corner(*pair)
                                if isJoined2 and proceed_processing and close_gap_new:
                                    close_gap_new.pop()
                            elif len(out_edge) == 0 and len(in_edge) == 0 \
                                and (angle_d < np.pi/5 or angle_d > (4/5*np.pi)):
                                normal_d = np.dot(line1.normal, line2.normal)
                                if normal_d > 0 and pair not in close_gap_new:
                                    close_gap_old.append(pair)
            pre = i

        close_gap_list = close_gap_new + close_gap_old 
        if len(close_gap_list) > 0:
            for pair in close_gap_list:
                if pair[0] in self.line_graph.nodes() and pair[1] in self.line_graph.nodes():
                    self.close_gap(*pair)
        lines = self.local_lines + self.new_lines
        if self.line_processing:
            for i, p in enumerate(lines[:-1]):
                line_p = self.line_graph.nodes[p]["line"]
                if line_p.isShort:
                    continue
                for q in lines[i+1:]:
                    line_q = self.line_graph.nodes[q]["line"]
                    if line_q.isShort or q in self.line_graph.neighbors(p):
                        continue
                    if (line_q.isUpdated or line_p.isUpdated):
                        angle_d = np.fabs(line_p.theta - line_q.theta)
                        if (angle_d > np.pi/3) and (angle_d < 2/3*np.pi):
                            p_endpoints = self.line_graph.nodes[p]["line"].endpoints 
                            q_endpoints = self.line_graph.nodes[q]["line"].endpoints 
                            seg2seg_dist = self.seg2seg_dist(p_endpoints, q_endpoints)
                            if self.door_width_interval[0] < seg2seg_dist < self.door_width_interval[1]:
                                self.detect_doors(p, q)
                            elif seg2seg_dist < self.corner_thresh and ((len(self.line_graph.in_edges(p)) == 0 or len(self.line_graph.out_edges(p)) == 0) \
                                or (len(self.line_graph.in_edges(q)) == 0 or len(self.line_graph.out_edges(q)) == 0)):
                                self.split_and_join(p, q)
                line_p.isUpdated = False
            self.line_graph.nodes[lines[-1]]["line"].isUpdated = False
    
    def generate_guess(self, start, end):
        """Computes a guess of line parameters for the least sqaures fitting.

        Args:
            start (int): start point of seed
            end (int): end point of seed

        Returns:
            list: line parameter guess
        """
        start = self.points[start]
        end = self.points[end]
        theta = np.pi/2 - np.arctan2(end[1] - start[1], end[0] - start[0])
        r = start[0]*np.cos(theta) + start[1]*np.sin(theta)
        if theta < 0:
            r = -r
            theta += np.pi
        elif theta > np.pi:
            r = -r
            theta -= np.pi
        return [r, theta]

    def detect_breakpoints(self, points, ranges, delta_theta):
        """Divides sensor measurements into separate clusters by identifying
        breakpoints at which two clusters should be separated.

        Args:
            points (list): list of measurement points
            ranges (list): list of ranges to the points
            delta_theta (float): angular resolution of measurements

        Returns:
            list: clusters of measurements
        """
        points_array = np.array(list(points.values()))
        distances = list(np.sqrt(np.sum(np.diff(points_array, axis=0)**2,1))) 
        distances.append(np.linalg.norm(points_array[0] - points_array[-1]))
        
        keys = list(points)
        clusters = [] # List of clusters
        cluster = [keys[0]] # Current cluster
        dist_cluster = []
        first_dist_cluster = []
        next_cluster = 0
        N = len(keys)
        for i in range(len(distances)):
            next = (i + 1) % N
            
            # Calculate the adaptive threshold for each point
            D = self.k * ranges[i] * delta_theta
            if distances[i] < D:
                dist_cluster.append(distances[i])
                if i == len(distances)-1:
                    clusters[0] = cluster + clusters[0]
                    first_dist_cluster = dist_cluster + first_dist_cluster
                else:
                    cluster.append(keys[next])
                
            else:
                # Breakpoint detected, end the current cluster and start a new one
                if not first_dist_cluster:
                    first_dist_cluster = dist_cluster
                
                if len(cluster) == 0:
                    continue
                elif len(cluster) <= 2 and len(clusters) > 0:
                    self.remaining_cluster_points[next_cluster] = {key: self.points[key] for key in cluster}
                    next_cluster += 1
                elif len(clusters) > 0 and np.mean(dist_cluster) > 0.11: 
                    for point_key in cluster:
                        self.new_remaining_points.pop(point_key)
                else:
                    clusters.append(cluster)
                cluster = [keys[next]]
                dist_cluster = []
                
        # Add the last cluster to the first cluster
        if len(clusters) > 0:
            if len(clusters[0]) <= 2:
                self.remaining_cluster_points[next_cluster] = {key: self.points[key] for key in clusters[0]}
                del clusters[0]
                next_cluster += 1
            elif np.mean(first_dist_cluster) > 0.11: 
                for point_key in clusters[0]:
                    self.new_remaining_points.pop(point_key)
                del clusters[0]
           
            if len(clusters) > 1:
                # Start with merging clusters
                i = 0
                while i < len(clusters):
                    next_i = (i + 1) % len(clusters) 

                    # Merge current and next cluster if they're close enough
                    if np.linalg.norm(self.points[clusters[i][-1]] - self.points[clusters[next_i][0]]) < 0.2:
                        clusters[i] += clusters[next_i]
                        del clusters[next_i]
                        # If the next cluster is before the current cluster, decrease i
                        if next_i < i:
                            i -= 1
                    else:
                        i += 1  # Move to the next cluster if no merge was done
        else:
            clusters.append(cluster)
         # After the entire merging process, go through clusters and remove those below the threshold
        i = 0
        while i < len(clusters):
            if len(clusters[i]) < self.params_.line_seed_points:
                self.remaining_cluster_points[next_cluster] = {key: self.points[key] for key in clusters[i]}
                next_cluster += 1
                del clusters[i]  # Remove the cluster without moving to the next one
            else:
                i += 1  # Move to the next cluster only if the current one is not removed
        return clusters
    
    def update(self, points, angles, ranges, robot_pos, current_room, local_lines, delta_theta, room_graph, line_processing=True):
        """Runs the line segment detection process for each arriving sensor reading

        Args:
            points (list): list of measurement points
            angles (list): list of angles of the measurements
            ranges (list): list of ranges to the points
            robot_pos (array): current position of the robot
            current_room (int): current room the robot is located
            local_lines (list): Local lines within sensor range
            delta_theta (float): angular resolution of the sensor
            room_graph (networkx graph): graph of room configuration
            line_processing (bool, optional): set to true if line segments should be processed, false otherwise.
        """
        self.viewed_lines = []
        self.found_new_lines = False
        self.robot_pos = robot_pos
        self.new_lines = []
        self.next_point += self.point_num_
        self.current_room = current_room
        self.local_lines = local_lines
        self.short_lines = [] 
        self.remaining_cluster_points = collections.defaultdict(dict)
        self.room_graph = room_graph
        self.line_processing = line_processing
        self.isUpdated = False
        self.new_remaining_points = dict()
        if len(points) > 0:
            new_points = {k + self.next_point: p for k, p in enumerate(points)}
            new_angles = {k + self.next_point: a for k, a in enumerate(angles)}
            self.points.update(new_points)
            self.angles.update(new_angles)
            self.new_remaining_points = copy.copy(new_points)
            if len(new_points) >= self.params_.min_line_points:
                self.num_lines = self.line_graph.number_of_nodes()
                if self.num_lines > 0:
                    self.old_lines = list(self.line_graph.nodes())
                self.point_num_ = len(new_points)
                self.point_clusters = self.detect_breakpoints(new_points, ranges, delta_theta)
                self.extract_line_segments()
            else:
                self.remaining_cluster_points = {0:new_points}
            if len(self.remaining_cluster_points) > 0 and (len(self.local_lines) > 0 or len(self.new_lines) > 0):
                self.filter_points()
    
    def filter_points(self):
        """Filters the measurements that remains after line segment detection.
        """
        lines = self.local_lines + self.new_lines
        lines = np.array([self.line_graph.nodes[line]["line"].endpoints for line in lines])[np.newaxis, ...]
        remaining_clusters = list(self.remaining_cluster_points.values())
        centroids = np.array([np.mean(list(cluster.values()), axis=0) for cluster in remaining_clusters])

        p1, p2 = lines[...,0,:], lines[...,1,:]
        line_vecs = p2 - p1
        point_vecs = centroids[:, np.newaxis, :] - p1

        t = np.sum(point_vecs * line_vecs, axis=-1) / (np.sum(line_vecs**2, axis=-1)+1e-10)
        t = np.clip(t, 0.0, 1.0)

        nearests = t[..., np.newaxis] * line_vecs + p1
        dists = np.linalg.norm(centroids[:, np.newaxis, :] - nearests, axis=-1)
        mask = np.any(dists <= 0.15, axis=-1)

        self.remaining_cluster_points = [cluster for cluster, m in zip(remaining_clusters, mask) if not m]
        self.new_remaining_points = dict(collections.ChainMap(*self.remaining_cluster_points))
        
    def get_farthest_point(self, seed_keys):
        """Computes the farthest point in seed_keys from the line 
        that connects the start and end of seed_keys.

        Args:
            seed_keys (list): keys of seed points

        Returns:
            int: key of farthest point
        """
        threshold = 0.10
        # Get the endpoints
        start, end = self.points[seed_keys[0]], self.points[seed_keys[-1]]

        # Compute the distances from all points to the line segment
        x1, y1 = start
        x2, y2 = end

        # Get all points except the endpoints
        inner_points = np.array([self.points[key] for key in seed_keys[1:-1]] )
        px, py = inner_points[:, 0], inner_points[:, 1]

        num = np.abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
        den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        distances = num / den

        # Compute the average distance
        avg_distance = np.mean(distances)

        if avg_distance > threshold:
            # If the average distance is above the threshold,
            # return the index (in the original points array) of the point with the maximum distance
            farthest_point_index = np.argmax(distances) + 1 
            return farthest_point_index
        else:
            return None    
    

    def extract_line_segments(self):
        """Performs the actual detection of new line segments 
        and updates of existing line segments.
        """
        line_include = 0
        if self.point_num_ < self.params_.min_line_points:
            return
        self.lineseg_clusters = []
        next_cluster = len(self.remaining_cluster_points)
        for cluster in self.point_clusters:
            line_include = -1
            lineseg_cluster = []
            i = 0
            pre = None
            is_last_short = False
            first_is_found = False
            cluster_points = {key: self.points[key] for key in cluster}
            while i <= len(cluster)-self.params_.line_seed_points:
                end = i + self.params_.line_seed_points-1
                seed_keys = cluster[i:end+1]
                
                isCorner = True
                while isCorner:
                    corner_index = self.get_farthest_point(seed_keys)
                    if corner_index is None:
                        isCorner = False
                    else:
                        i += corner_index
                        end = i + self.params_.line_seed_points-1
                        if end-i < 10 or end >= len(cluster):
                            break
                        seed_keys = cluster[i:end+1]
                if end-i < 10 or end >= len(cluster):
                    continue
                guess = self.generate_guess(cluster[i], cluster[end])
                self.m_least = self.leastsquare(seed_keys, guess)
                if self.isLine(seed_keys):
                    if i - line_include < 10:
                        for k in range(line_include+1, i):
                            key = cluster[k]
                            self.new_remaining_points.pop(key)
                            del cluster_points[key]
                    line_include, isDetected, isShort, line_length, point_keys = self.grow_line_seed(i, end, cluster)
                    if line_length >= 0.4 or len(point_keys) < 10:
                        for key in point_keys:
                            self.new_remaining_points.pop(key)  
                            del cluster_points[key]
                    i = line_include
                    if isDetected:
                        self.found_new_lines = True
                        self.line_graph.nodes[self.next_line-1]["line"].cluster_number = next_cluster
                        if not isShort:
                            if not first_is_found:
                                first_is_found = True
                        if pre is not None:
                            if not isShort:
                                if not is_last_short:
                                    self.line_graph.add_edge(pre, self.next_line-1, type='corner')
                        is_last_short = isShort
                            
                        lineseg_cluster.append(self.next_line-1)
                        pre = self.next_line-1
                i += 1 
            if len(cluster)-1 - line_include < 10 and len(cluster)-1 - line_include > 0:
                for k in range(line_include+1, len(cluster)):
                    key = cluster[k]
                    if key in self.new_remaining_points:
                        self.new_remaining_points.pop(key)
                        del cluster_points[key]
            if len(lineseg_cluster) > 0:
                self.lineseg_clusters.append(lineseg_cluster)
            if len(cluster_points) > 0:
                self.remaining_cluster_points[next_cluster] = cluster_points
            next_cluster += 1

        self.clean_line_segments()

        if self.isUpdated:
            self.process_line_segments()
        
        while len(self.short_lines) > 0:
            line = self.short_lines[0]
            if len(self.line_graph.in_edges(line)) == 0 or len(self.line_graph.out_edges(line)) == 0:
                short_line = self.line_graph.nodes[line]["line"]
                points_dict = {key: self.points[key] for key in short_line.point_keys}
                self.new_remaining_points.update(points_dict)
                self.remaining_cluster_points[short_line.cluster_number].update(points_dict)
                self.remove_line(line)

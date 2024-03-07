import os
import cv2
import glob
import math
import collections
from copy import deepcopy
from render import render
import json
import  numpy as np
import random
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--background_path",
        type=str,
    )
    parser.add_argument(
        "--ROOT",
        type=str,
    )
    parser.add_argument(
        "--FPS",
        type=int,
    )
    parser = parser.parse_args()
    return parser

DEBUG = True
RENDER_ONLY = False
DIVISION = 50
W_CONSTANT = 1.17
FPS = 25
INF = 2**32 - 1
WEIGHT_COLLISIONS = 0.6
WEIGHT_DENSITY = 0.05
WEIGHT_TIME = 0.3

THRESHOLD_OVERLAP = 0.25
# background_path = 'background_paper.jpg'
# ROOT = "/home1/data/congvu/deepstream-test1/synopsis_paper"

def calculate_angle(start_point, end_point):
    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def check_direction(object_1, object_2):
    start_point1, end_point1 = object_1
    start_point2, end_point2 = object_2
    angle1 = calculate_angle(start_point1, end_point1)
    angle2 = calculate_angle(start_point2, end_point2)
    angle_diff = abs(angle1 - angle2)
    return angle_diff < 90

def check_overlap_bounding_boxes(box1, box2):
    """
    Check if two bounding boxes overlap or not.

    Parameters:
    box1 (tuple): Coordinates of the first bounding box in the format (x1, y1, x2, y2).
    box2 (tuple): Coordinates of the second bounding box in the format (x1, y1, x2, y2).

    Returns:
    bool: True if the bounding boxes overlap, False otherwise.
    """
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Calculate the overlapping area
    overlap_width = min(x2_box1, x2_box2) - max(x1_box1, x1_box2)
    overlap_height = min(y2_box1, y2_box2) - max(y1_box1, y1_box2)

    # Check if there is any overlap
    if overlap_width <= 0 or overlap_height <= 0:
        return False

    # Calculate the overlap ratio
    overlap_area = overlap_width * overlap_height
    box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    overlap_ratio = overlap_area / min(box1_area, box2_area)

    # Check if the overlap ratio is larger than 0.3
    if overlap_ratio > THRESHOLD_OVERLAP:
        return True

    return False


def check_overlap_segment(segment_a, segment_b):
    """
        Check that 2 segment are overlapped
    """
    if segment_a[0] > segment_b[1] or segment_a[1] < segment_b[0]:
        return False
    return True

def cal_threshold_V(duration_list = []):
    # return math.exp(-W_CONSTANT*sum(duration_list)/len(duration_list)*FPS/30)
    return W_CONSTANT*sum(duration_list)/len(duration_list)*FPS

def euclid_distance_bottom_edge(box_a, box_b):
    ctx_a = (box_a[0] + box_a[2])/2
    cty_a = max(box_a[1],box_a[3])
    ctx_b = (box_b[0] + box_b[2])/2
    cty_b = max(box_b[1],box_b[3])
    return math.sqrt((ctx_a - ctx_b)**2 + (cty_a - cty_b)**2)

def R_interactive_conditions(bboxes_a, bboxes_b, N_tube_1_and_tube_2):
    box1,prev_box1, box2, prev_box2 = bboxes_a[1],bboxes_a[0],bboxes_b[1], bboxes_b[0]
    pre_center_1 = [(prev_box1[0] + prev_box1[2]) / 2, (prev_box1[1] + prev_box1[3]) / 2]
    pre_center_2 = [(prev_box2[0] + prev_box2[2]) / 2, (prev_box2[1] + prev_box2[3]) / 2]
    center_1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center_2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    # Compare the signs of the displacements to check if both objects are moving in the same direction
    same_direction = check_direction([pre_center_1,center_1],[pre_center_2, center_2])

    if same_direction:
        FX_TUBE_1_AND_TUBE_2 = 1
    else:
        FX_TUBE_1_AND_TUBE_2 = 0

    if FX_TUBE_1_AND_TUBE_2 == 1:
        R_TUBE_1_AND_TUBE_2 = 1
    else:
        if FPS<N_tube_1_and_tube_2:
            R_TUBE_1_AND_TUBE_2 = 1
        else:
            R_TUBE_1_AND_TUBE_2 = 0
    return R_TUBE_1_AND_TUBE_2

def calculate_energy(lst_grouptube, matrix):
    shifted_list_grouptube = []
    list_energy = []
    for row in matrix:
        for i,grouptube in enumerate(lst_grouptube):
            shifted_grouptube = deepcopy(grouptube)
            shifted_grouptube.shift_group_tube(abs(row[i]))
            shifted_list_grouptube.append(shifted_grouptube)
        tmp_energy = 0
        for i,grouptube_1 in enumerate(shifted_list_grouptube[:-1]):
            tmp_energy += energy_TSMM(shifted_list_grouptube[i+1:],grouptube_1)
        list_energy.append(tmp_energy)
    return list_energy.index(min(list_energy)),list_energy.index(min(list_energy)), list_energy



def calculate_spatio_distance_group(grouptube_a, grouptube_b, mode, reduce_type, THRESHOLD_V):
    '''
        Calculate calculate_spatio_distance for E_t
        Target: Maximize if mode = syn
        Return
            - Distance
    '''
    assert mode in ["ori", "syn"]
    assert reduce_type in ["avg", "min"]
    lst_distance = []
    g_interaction_global = []
    R_inter_lst = []
    for ia in range(grouptube_a.total_tubes):
        for ib in range(grouptube_b.total_tubes):
            intersection = []
            g_interaction = [0]
            if mode == "ori":
                time_a = grouptube_a.group_ori_time_segment[ia]
                time_b = grouptube_b.group_ori_time_segment[ib]
            else:
                time_a = grouptube_a.group_syn_time_segment[ia]
                time_b = grouptube_b.group_syn_time_segment[ib]

            if not check_overlap_segment(time_a, time_b):
                continue # mark as INFINITY, so skip
            else:
                range_a      = list(range(time_a[0], time_a[1] + 1))
                range_b      = list(range(time_b[0], time_b[1] + 1))
                # Get list of intersection frame ID
                intersection = list(set(range_a) & set(range_b))
                min_distance = deepcopy(INF)
                for frame_id in intersection:
                    box_a = grouptube_a.get_box_by_tube_id_frame_id(tube_id = ia, frame_id = frame_id,
                                                                                        mode = mode)
                    box_b = grouptube_b.get_box_by_tube_id_frame_id(tube_id = ib, frame_id = frame_id,mode = mode)
                    distance = euclid_distance_bottom_edge(box_a,box_b)
                    height_a, height_b =  (box_a[3] - box_a[1]), (box_b[3] - box_b[1])
                    if distance/((height_a+height_b)/2)<W_CONSTANT:
                        g_interaction.append(1)
                    lst_distance.append(distance)
            if len(intersection) <= 1:
                continue
            boxes_tube_1 = [grouptube_a.get_box_by_tube_id_frame_id(tube_id = ia, frame_id = intersection[0],mode = mode),
                            grouptube_a.get_box_by_tube_id_frame_id(tube_id = ia, frame_id = intersection[-1],mode = mode)]
            boxes_tube_2 = [grouptube_b.get_box_by_tube_id_frame_id(tube_id = ib, frame_id = intersection[0],mode = mode),
                            grouptube_b.get_box_by_tube_id_frame_id(tube_id = ib, frame_id = intersection[-1],mode = mode)]
            avg_dis = sum(lst_distance)/len(lst_distance)
            N_tube_1_and_tube_2 =  sum(g_interaction)
            R_inter = R_interactive_conditions(boxes_tube_1,boxes_tube_2,N_tube_1_and_tube_2)
            distance_Tube_1_and_tube_2 = avg_dis+np.log(N_tube_1_and_tube_2)
            if distance_Tube_1_and_tube_2<THRESHOLD_V and R_inter==1 and N_tube_1_and_tube_2 != 0:
                return True,grouptube_a.group_object_id[ia]
            else: 
                pass

    return False,None

def density_cost(Q_tubes, shifted_grouptube,mode="syn", threshold = 30):
    print("[INFO] Calculate DENSITY COST")
    cost = 0
    density_dict = {}
    for tube_T_Si in Q_tubes:
        for ia in range(tube_T_Si.total_tubes):
            if mode == "ori":
                        time_a = tube_T_Si.group_ori_time_segment[ia]
            else:
                        time_a = tube_T_Si.group_syn_time_segment[ia]
            for frame_id in range(time_a[0],time_a[1]+1):
                boxes = tube_T_Si.get_box_by_tube_id_frame_id(tube_id = ia,frame_id = frame_id,mode = mode)
                if frame_id not in density_dict.keys():
                    density_dict[frame_id] = len(boxes)
                else:
                    density_dict[frame_id] += len(boxes)
    for ia in range(shifted_grouptube.total_tubes):
            if mode == "ori":
                        time_a = shifted_grouptube.group_ori_time_segment[ia]
            else:
                        time_a = shifted_grouptube.group_syn_time_segment[ia]
            for frame_id in range(time_a[0],time_a[1]+1):
                boxes = shifted_grouptube.get_box_by_tube_id_frame_id(tube_id = ia,frame_id = frame_id,mode = mode)
                if frame_id not in density_dict.keys():
                    density_dict[frame_id] = len(boxes)
                else:
                    density_dict[frame_id] += len(boxes)
    for k,v in density_dict.items():
        if v >= threshold:
            cost+=1
    print(f"[INFO] DENSITY COST IS {cost}")
    return cost
def collision_cost(Q_tubes, shifted_grouptube, mode='syn'):
    print("[INFO] Calculate COLLISION COST")
    cost = 0
    tube_T_Sj = shifted_grouptube
    for tube_T_Si in Q_tubes:
        for ia in range(tube_T_Si.total_tubes):
            for ib in range(tube_T_Sj.total_tubes):
                    intersection = []
                    if mode == "ori":
                        time_a = tube_T_Si.group_ori_time_segment[ia]
                        time_b = tube_T_Sj.group_ori_time_segment[ib]
                    else:
                        time_a = tube_T_Si.group_syn_time_segment[ia]
                        time_b = tube_T_Sj.group_syn_time_segment[ib]

                    if not check_overlap_segment(time_a, time_b):
                        continue # mark as INFINITY, so skip
                    else:
                        range_a      = list(range(time_a[0], time_a[1] + 1))
                        range_b      = list(range(time_b[0], time_b[1] + 1))
                        # Get list of intersection frame ID
                        intersection = list(set(range_a) & set(range_b))
                        for frame_id in intersection:
                            box_a = tube_T_Si.get_box_by_tube_id_frame_id(tube_id = ia, frame_id = frame_id,mode = mode)
                            box_b = tube_T_Sj.get_box_by_tube_id_frame_id(tube_id = ib, frame_id = frame_id,mode = mode)
                            is_overlap = check_overlap_bounding_boxes(box_a,box_b)
                            if is_overlap:
                                cost+=1
    print(f"[INFO] COLLISION COST IS {cost}")
    return cost

def Chronological_cost(Q_tubes, shifted_grouptube,mode = 'syn'):
    print("[INFO] Calculate Chronological")
    cost = 0
    tube_T_Sj = shifted_grouptube
    list_ori = []
    list_syn = []
    for tube_T_Si in Q_tubes:
        for ia in range(tube_T_Si.total_tubes):
                for ib in range(tube_T_Sj.total_tubes):
                    time_a = tube_T_Si.group_ori_time_segment[ia]
                    time_b = tube_T_Sj.group_ori_time_segment[ib]
                    time_a_syn = tube_T_Si.group_syn_time_segment[ia]
                    time_b_syn = tube_T_Sj.group_syn_time_segment[ib]
                    if time_a[0] <= time_b[0] :
                        if time_a_syn[0] <= time_b_syn[0]:
                            continue
                        else:
                            cost+=1
                    elif time_a[0] > time_b[0] :
                        if time_a_syn[0] > time_b_syn[0]:
                            continue
                        else:
                            cost+=1
    print(f"[INFO] Calculate Chronological cost {cost}")
    return cost


def energy_TSMM(Q_tubes, shifted_grouptube):
    '''
        Order distance define by myself
    '''
    cost_colli = collision_cost(Q_tubes,shifted_grouptube)
    cost_density = density_cost(Q_tubes, shifted_grouptube)
    cost_chronological = Chronological_cost(Q_tubes,shifted_grouptube)
    energy = WEIGHT_COLLISIONS*cost_colli + WEIGHT_DENSITY*cost_density + WEIGHT_TIME*cost_chronological
    return energy
def optimize_shift_grouptube(lst_grouptube):
    print("[INFO] TIME TO OPTIMIZE")
    total_grouptubes = len(lst_grouptube)
    lst_grouptube_result = []
    # Get first tubes & shift all remain tubes
    first_grouptube = lst_grouptube.pop(0)
    n_frames_to_shift = first_grouptube.wrap_group_ori_time_segment[0]
    first_grouptube.shift_group_tube(n_frames_to_shift)
    # lst_tubes  = shift_all_tube_by_n_frames(lst_tubes, n_frames_to_shift)
    assert first_grouptube.wrap_group_syn_time_segment[0] == 0
    for i in range(first_grouptube.total_tubes):
        print(first_grouptube.group_syn_time_segment[i])
    # assert False
    if DEBUG:
        print('='*25, " GroupTube 1 ", '='*25)
        first_grouptube.print_out()
    lst_grouptube_result.append(first_grouptube)
    # Check all tubes
    while len(lst_grouptube) > 0:
        grouptube = lst_grouptube.pop(0)
        lower, higher = get_range_of_grid(lst_grouptube_result)
        stride = max(1, int((higher - lower)/DIVISION)) # Minimum stride = 1
        best_energy = 1e20 # Get max distance as possible
        best_shift_value = -1
        print('='*25, " GroupTube {} ".format(len(lst_grouptube_result) + 1), '='*25)
        if DEBUG:
            print('- Lower/Higher: {} - {}'.format(lower, higher))
            print('- Stride: ', stride)
        for i in range(DIVISION):
            
            if lower > higher:
                break
            print('-> Iter {}: Try from {} - {}'.format(i, lower, higher))
            n_frames_shifted = grouptube.wrap_group_syn_time_segment[0] - lower
            shifted_grouptube = deepcopy(grouptube)
            shifted_grouptube.shift_group_tube(n_frames_shifted)
            energy = energy_TSMM(lst_grouptube_result,shifted_grouptube)

            if energy < best_energy:
                best_energy = energy
                best_shift_value = n_frames_shifted
                if DEBUG:
                    print('--> Found best distance {} from frame {} with shift value {}'.format(energy, lower, best_shift_value))
            lower += stride

        if best_energy == 1e20:
            print('Best shift not found, append to the end')
            # Append to the end
            n_frames_shifted = grouptube.wrap_group_syn_time_segment[0] - higher - 1
            shifted_grouptube = deepcopy(grouptube)
            shifted_grouptube.shift_group_tube(n_frames_shifted)
        else:
            shifted_grouptube = deepcopy(grouptube)
            shifted_grouptube.shift_group_tube(best_shift_value)
        if DEBUG:
            print('-'*10, ' Finally ', '-'*10)
            shifted_grouptube.print_out()
        lst_grouptube_result.append(shifted_grouptube)
    print('-'*10, ' Finally ', '-'*10)
    # for grouptube in lst_grouptube_result:
    #     grouptube.print_out()
    assert len(lst_grouptube_result) == total_grouptubes
    return lst_grouptube_result
class GroupTube(object):
    def __init__(self, group_object_id, group_ori_time_segment, group_lst_box, group_lst_path = None):
        """
            Class for grouping of tube
        """
        assert len(group_object_id) == len(group_ori_time_segment)
        assert len(group_object_id) == len(group_lst_box)
        if group_lst_path is not None:
            assert len(group_object_id) == len(group_lst_path)
        # Store for all tubes
        self.group_object_id = group_object_id
        self.group_ori_time_segment = group_ori_time_segment
        self.group_lst_box = group_lst_box
        self.group_lst_path = group_lst_path

        # Init solution
        self.group_syn_time_segment = deepcopy(self.group_ori_time_segment)

        # Calculate for wrapping-meta
        self.rebuild_wrap_meta()

    def rebuild_wrap_meta(self):
        """
            Re-calculate wrap meta, usually do after merging or initing group
        """
        self.total_tubes = len(self.group_ori_time_segment)
        self.wrap_group_ori_time_segment = [None, None]
        self.wrap_group_ori_time_segment[0] = min([self.group_ori_time_segment[i][0] for i in range(self.total_tubes)])
        self.wrap_group_ori_time_segment[1] = max([self.group_ori_time_segment[i][1] for i in range(self.total_tubes)])

        self.wrap_group_syn_time_segment = [None, None]
        self.wrap_group_syn_time_segment[0] = min([self.group_syn_time_segment[i][0] for i in range(self.total_tubes)])
        self.wrap_group_syn_time_segment[1] = max([self.group_syn_time_segment[i][1] for i in range(self.total_tubes)])
        self.total_frames = self.wrap_group_ori_time_segment[1] - self.wrap_group_ori_time_segment[0] + 1

    def get_box_by_tube_id_frame_id(self, tube_id, frame_id, mode):
        """
            Get object box by TubeID and FrameID
        """
        assert mode in ['ori', 'syn']
        if mode == 'ori':
            offset = frame_id - self.group_ori_time_segment[tube_id][0]
            assert offset >= 0 and offset < self.total_frames
            return self.group_lst_box[tube_id][offset]
        else:
            offset = frame_id - self.group_syn_time_segment[tube_id][0]
            assert offset >= 0 , f"OFFSET IS LESS THAN 0: {offset}"
            assert offset < self.total_frames, f"OFFSET IS GREATER THAN total_frames: {offset} < {self.total_frames}"
            return self.group_lst_box[tube_id][offset]

    def get_path_by_tube_id_frame_id(self, tube_id, frame_id, mode):
        """
            Get object box by TubeID and FrameID
        """
        assert mode in ['ori', 'syn']
        if mode == 'ori':
            offset = frame_id - self.group_ori_time_segment[tube_id][0]
            assert offset >= 0 and offset < self.total_frames
            return self.group_lst_path[tube_id][offset]
        else:
            offset = frame_id - self.group_syn_time_segment[tube_id][0]
            assert offset >= 0 and offset < self.total_frames
            return self.group_lst_path[tube_id][offset]

    def sort_group_by_syn_time_segment(self):
        """
            Rebuild group by sorting synopsis time segment (not original time segment)
        """
        sorted_group_object_id = [x for x, y in sorted(zip(self.group_object_id, self.group_syn_time_segment), key = lambda x: x[1][0])]
        sorted_group_ori_time_segment = []
        sorted_group_syn_time_segment = []
        sorted_group_lst_box = []
        if self.group_lst_path is not None:
            sorted_group_lst_path = []
        for i in range(self.total_tubes):
            sorted_group_ori_time_segment.append(self.group_ori_time_segment[self.group_object_id.index(sorted_group_object_id[i])])
            sorted_group_syn_time_segment.append(self.group_syn_time_segment[self.group_object_id.index(sorted_group_object_id[i])])
            sorted_group_lst_box.append(self.group_lst_box[self.group_object_id.index(sorted_group_object_id[i])])
            if self.group_lst_path is not None:
                sorted_group_lst_path.append(self.group_lst_path[self.group_object_id.index(sorted_group_object_id[i])])
        self.group_object_id        = sorted_group_object_id
        self.group_ori_time_segment = sorted_group_ori_time_segment
        self.group_syn_time_segment = sorted_group_syn_time_segment
        self.group_lst_box          = sorted_group_lst_box
        if self.group_lst_path is not None:
            self.group_lst_path         = sorted_group_lst_path

    def print_out(self):
        """
            Print information of group
        """
        print('-'*20)
        for i in range(self.total_tubes):
            print('Tube ', i)
            print('   Object ID: ', self.group_object_id[i])
            print('   Original frame range: ', self.group_ori_time_segment[i])
            print('   Synopsis frame range: ', self.group_syn_time_segment[i])

    def merge_with_group_tube(self, group_tube):
        """
            Merge with another group_tube
        """
        self.group_object_id.extend(group_tube.group_object_id)
        assert len(self.group_object_id) == len(list(set(self.group_object_id))), "Object ID overlapped: {}".format(self.group_object_id) # Make sure no overlap object ID
        self.group_ori_time_segment.extend(group_tube.group_ori_time_segment)
        self.group_syn_time_segment.extend(group_tube.group_syn_time_segment)
        self.group_lst_box.extend(group_tube.group_lst_box)
        if self.group_lst_path is not None:
            self.group_lst_path.extend(group_tube.group_lst_path)

        self.rebuild_wrap_meta()
        self.sort_group_by_syn_time_segment()
        

    def shift_group_tube(self, shift_value):
        """
            Shift all tube in group by amount of frames
        """
        for tube in self.group_syn_time_segment:
            tube[0] -= shift_value
            tube[1] -= shift_value
        # Rebuild wrap meta only for synopsis
        self.wrap_group_syn_time_segment[0] -= shift_value
        self.wrap_group_syn_time_segment[1] -= shift_value

def parse_label(label_path):
    tmp = open(label_path).read().strip().split()
    return [float(x) for x in tmp]


def sort_grouptube_by_time(dict_grouptube):
    """
        Sort all tube by time
    """
    lst_object_id = list(dict_grouptube.keys())
    lst_grouptube     = list(dict_grouptube.values())

    sorted_object_id = sorted(lst_object_id, key = lambda x: dict_grouptube[x].wrap_group_ori_time_segment[0])
    sorted_grouptube     = [dict_grouptube[x] for x in sorted_object_id]
    return sorted_object_id, sorted_grouptube


def get_range_of_grid(lst_grouptube_result):
    # lower  = min([grouptube.wrap_group_syn_time_segment[0] for grouptube in lst_grouptube_result])
    # higher = max([grouptube.wrap_group_syn_time_segment[1] for grouptube in lst_grouptube_result])
    lower  = lst_grouptube_result[-1].wrap_group_syn_time_segment[0]
    higher = lst_grouptube_result[-1].wrap_group_syn_time_segment[1]
    return lower, higher 


def perform_group_tubes(lst_grouptube):
    lst_grouptube_result = []
    lst_duration_times = []
    for grouptube in lst_grouptube:
        for i in range(grouptube.total_tubes):
            time_a = grouptube.group_ori_time_segment[i]
            second = float((time_a[1]-time_a[0]+1)/FPS)
            lst_duration_times.append(second)
    THRESHOLD_V = cal_threshold_V(lst_duration_times)
    # print(lst_duration_times)
    # print(THRESHOLD_V)
    # assert False
    while len(lst_grouptube) > 0:
        grouptube = lst_grouptube.pop(0)
        if len(lst_grouptube_result) == 0:
            lst_grouptube_result.append(grouptube)
        else:
            merged = False
            for grouptube_rs in lst_grouptube_result:
                boolean_merged, tube_choosed = calculate_spatio_distance_group(grouptube, grouptube_rs, mode = 'ori', reduce_type = 'avg',THRESHOLD_V=THRESHOLD_V)
                if boolean_merged:
                    if DEBUG:
                        print('--> Merge group {} vs {} by {}'.format(grouptube.group_object_id, grouptube_rs.group_object_id,tube_choosed))
                    grouptube_rs.merge_with_group_tube(grouptube)
                    merged = True
                    break
            if not merged:
                lst_grouptube_result.append(grouptube)
    return lst_grouptube_result


def get_object_frame_and_box(args, skip_if_less_than=10):
    """
        Processing: Handle missing frames when tracking objs
        Return: dictionaries contain object_id (key) and bounding boxes +
            time object appaers + path to image object
    """
    dict_object = dict()
    for object_id in glob.glob(f'{args.ROOT}/*'):
        object_id = object_id.replace('\\', '/')
        lst_cropped = glob.glob(object_id + '/*.jpg')
        object_id = int(object_id.split('/')[-1])
        tmp_frame_box = dict()
        for cropped in lst_cropped:
            cropped = cropped.replace('\\', '/')
            frame_id = int(cropped.split('_')[-1].split('.')[0])
            box      = parse_label(cropped.replace('.jpg', '.txt'))
            tmp_frame_box[frame_id] = {"box": box, "path": cropped}

        # Sort dict
        sorted_frame_box = collections.OrderedDict(sorted(tmp_frame_box.items()))
        # Regress missing frame
        tmp_frame_id = list(sorted_frame_box.keys())
        tmp_box_path = list(sorted_frame_box.values())
        if len(tmp_frame_id) == 0:
            continue

        total_frames = max(tmp_frame_id) - min(tmp_frame_id) + 1
        if total_frames < skip_if_less_than:
            continue
        assert total_frames >= len(tmp_frame_id)
        if total_frames > len(tmp_frame_id):
            # Regress missing box
            # assert False
            lower  = min(tmp_frame_id)
            higher = max(tmp_frame_id)
            lst_box_path = []
            last_frame_box_path  = None
            total_regressed = int((len(tmp_frame_id) - total_frames)*100/total_frames)
            for frame_id in range(lower, higher + 1):
                if frame_id not in tmp_frame_id:
                    # Append last frame
                    lst_box_path.append(last_frame_box_path)
                else:
                    # Get from lst_box
                    last_frame_box_path = tmp_box_path[tmp_frame_id.index(frame_id)]
                    lst_box_path.append(last_frame_box_path)
            assert len(lst_box_path) == total_frames
        else:
            if DEBUG:
                print('[INFO] Object {} no need to regress'.format(object_id))
                print('    - Total frame: ', total_frames)
                print('    - Frame range: ', min(tmp_frame_id), max(tmp_frame_id))
            lst_box_path      = tmp_box_path
        ori_time_segment = [min(tmp_frame_id), max(tmp_frame_id)]
        dict_object[object_id] = {"ori_time_segment": ori_time_segment, "lst_box": [x["box"] for x in lst_box_path], "lst_path": [x["path"] for x in lst_box_path]}
    return dict_object

if __name__ == '__main__':
    args = parse_arguments()
    dict_object = get_object_frame_and_box(args)
    dict_grouptube   = dict()
    for object_id in dict_object:
        dict_grouptube[object_id] = GroupTube(group_object_id = [object_id],
                                    group_ori_time_segment = [dict_object[object_id]["ori_time_segment"]],
                                    group_lst_box = [dict_object[object_id]["lst_box"]],
                                    group_lst_path = [dict_object[object_id]["lst_path"]])
    _, lst_grouptube = sort_grouptube_by_time(dict_grouptube)
    lst_grouptube = perform_group_tubes(lst_grouptube)
    print("[INFO] Number of Tube Groups are ",len(lst_grouptube))
    # HSAJAYA_optimization(lst_grouptube)
    lst_grouptube_synopsis = optimize_shift_grouptube(lst_grouptube)
    render(lst_grouptube_synopsis, dict_object,lst_grouptube_synopsis[-1].wrap_group_syn_time_segment[1],
        args)
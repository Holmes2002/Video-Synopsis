import os
import cv2
import time
from progressbar import progressbar
from tqdm import tqdm
import glob
import numpy as np
ori_time = {}
# FPS = 15
TEXT_SIZE = 0.25

def find_start_time_per_obj(path):
    global ori_time
    for obj_id_folder in os.listdir(path):
        for file in glob.glob(f'{path}/{obj_id_folder}/**.jpg',recursive=True):
            if int(obj_id_folder) not in list(ori_time.keys()):
                frame_id = file.split('_')[-1].split('.')[0]
                ori_time[int(obj_id_folder)] = [int(frame_id)]
            else:
                frame_id = int(file.split('_')[-1].split('.')[0])
                ori_time[int(obj_id_folder)].append(int(frame_id))



def get_frame_info(frame_id, lst_grouptube_synopsis, dict_object):
    """
        Get box and image coordinate for each object
    """
    frame_info = []
    for grouptube in lst_grouptube_synopsis:
        if frame_id > grouptube.wrap_group_syn_time_segment[1] or frame_id < grouptube.wrap_group_syn_time_segment[0]:
            continue
        for tube_id in range(grouptube.total_tubes):
            if frame_id > grouptube.group_syn_time_segment[tube_id][1] or frame_id < grouptube.group_syn_time_segment[tube_id][0]:
                continue

            frame_info.append({"box": grouptube.get_box_by_tube_id_frame_id(tube_id = tube_id,
                                                                        frame_id = frame_id,
                                                                        mode = 'syn'),
                            "path": grouptube.get_path_by_tube_id_frame_id(tube_id = tube_id,
                                                                        frame_id = frame_id,
                                                                        mode = 'syn'),
                            "object_id": grouptube.group_object_id[tube_id],
                            "frame_id": frame_id - grouptube.group_syn_time_segment[tube_id][0] + grouptube.group_ori_time_segment[tube_id][0]})
    return frame_info

def frame_index_to_time(frame_index, fps=30):
    """
    Convert frame index to time in hours, minutes, and seconds.

    Parameters:
    frame_index (int): Frame index.
    fps (int): Frames per second (default is 30).

    Returns:
    tuple: (hours, minutes, seconds).
    """
    total_seconds = frame_index / fps
    hours = int(total_seconds / 3600)
    minutes = int((total_seconds % 3600) / 60)
    seconds = int(total_seconds % 60)
    return f"{hours}:{minutes}:{seconds}"


def get_frame_image(frame_info,background_image):
    global  ori_time
    total_object = len(frame_info)
    lst_box = []
    lst_image = []
    list_objs_id = []
    list_frame_id = []
    for i in range(total_object):
        x1, y1, x2, y2 = frame_info[i]['box']
        frame_id = int(frame_info[i]['frame_id'])
        list_frame_id.append(frame_id)
        list_objs_id.append(frame_info[i]['object_id'])
        x1 = int(x1) 
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        lst_box.append((x1, y1, x2, y2))
        img = cv2.imread(frame_info[i]['path'])
        assert y1< background_image.shape[0] and y2<= background_image.shape[0], f"Wrong height: {y1}< {background_image.shape[0]} vs {y2}=< {background_image.shape[0]} ,path {frame_info[i]['path']}"
        assert x1< background_image.shape[1] and x2<= background_image.shape[1], f"Wrong width: {x1}< {background_image.shape[1]} vs {x2}=< {background_image.shape[1]} ,path {frame_info[i]['path']}"
        lst_image.append(img)
    assert len(lst_box) == len(lst_image)
    return lst_box, lst_image, list_objs_id, list_frame_id

def reconstruct_image_from_frame_info_new(img, lst_box, lst_image, list_objs_id =[],list_frame_id=[],args = ''):
    """
        Reconstruct image from frame_info
    """
    global TEXT_SIZE, ori_time
    total_object = len(lst_box)
    for ix in range(total_object):
        obj_img = lst_image[ix]
        idx_obj = int(list_objs_id[ix])

        x1, y1, x2, y2 = lst_box[ix]
        time = f"{idx_obj} {frame_index_to_time(list_frame_id[ix], args.FPS)}"
        height, width, _ = obj_img.shape
        assert x2 - x1 == width, "Wrong width: {} vs {}".format(x2 - x1, width)
        assert y2 - y1 == height, "Wrong height: {} vs {}".format(y2 - y1, height)
        assert y1< img.shape[0] and y2<= img.shape[0], f"Wrong width: {y1}< {img.shape[0]} vs {y2}=< {img.shape[0]}"
        assert x1< img.shape[1] and x2<= img.shape[1], f"Wrong width: {x1}< {img.shape[1]} vs {x2}=< {img.shape[1]}"
        
        img[y1: y2, x1: x2,:] = img[y1: y2, x1: x2,:]*0.25 + obj_img*0.75

        # img = cv2.seamlessClone(obj_img, img, np.ones(obj_img.shape, obj_img.dtype)*255, (int((x1+x2)/2), int((y1+y2)/2)), cv2.MIXED_CLONE)

        img = cv2.putText(img, time, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX,  
                   TEXT_SIZE, (255,255,255), 1, cv2.LINE_AA) 

    return img


def render(lst_grouptube_synopsis, dict_object, num_syn_frames = 1000, args = ''):
    """
        Perform render video from mapping
    """
    find_start_time_per_obj(args.ROOT)
    render_range = [None, None]

    render_range[0] = min([grouptube.wrap_group_syn_time_segment[0] for grouptube in lst_grouptube_synopsis])
    render_range[1] = max([grouptube.wrap_group_syn_time_segment[1] for grouptube in lst_grouptube_synopsis])

    background_image = cv2.imread(args.background_path)
    height, width, _ = background_image.shape
    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    tik = time.time()
    # for frame_id in progressbar(range(render_range[0], render_range[1] + 1)):
    
    for frame_id in tqdm(range(num_syn_frames)):
            t0 = time.time()
            frame_info = get_frame_info(frame_id, lst_grouptube_synopsis, dict_object)
            t1 = time.time()
            lst_box, lst_image, list_objs_id, list_frame_id = get_frame_image(frame_info,background_image.copy())
            frame  = reconstruct_image_from_frame_info_new(background_image.copy(), lst_box, lst_image,list_objs_id,list_frame_id, args)
            t2 = time.time()
            writer.write(frame)
            t3 = time.time()
            # print('Time get info: ', t1 - t0)
            # print('Time reconstruct: ', t2 - t1)
            # print('Time write image: ', t3 - t2)
    writer.release()
    tok = time.time()
    print(tok - tik)
    
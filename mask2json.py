import os
import cv2
import numpy as np
import json
from skimage.measure import label, regionprops


def get_points_list(mask):
    # output format: [['1,2','2,3'], ['4,5','4,3']]
    # cal
    label_mask = label(mask, connectivity = mask.ndim, background = 0)
    # props = regionprops(label_img)
    points_list = []
    for label_ in np.unique(label_mask):
        if label_ == 0:
            pass
        else:
            index = np.argwhere(label_mask==label_)
            pos_list = []
            for pos in index:
                pos_list.append(str(pos[0])+','+str(pos[1]))
            points_list.append(pos_list)
    return points_list


def get_json_file(mask_path):
    mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask_>=255/2.)*255
    points_list = get_points_list(mask)

    # reformat information as json
    params_str = []
    points_dict = dict()
    for i, points in enumerate(points_list):
        tmp = 'points'+str(i)
        params_str.append(tmp)
        points_dict[tmp] =  points
    result = {'Height': mask.shape[0],
             'Width': mask.shape[1],
             'name': mask_name,
             'regions': points_dict }
    json_str = json.dumps(result)
    for item in params_str:
        json_str = json_str.replace(item, 'points')
    return json_str

if __name__ == '__main__':
    mask_name = 'mask.png'
    mask_path = os.path.join(data_path, mask_name)
    json_str = get_json_file(mask_path)
    print(json_str)
    
 # ref: https://blog.csdn.net/pursuit_zhangyu/article/details/94209489

from ultralytics import YOLO
from ultralytics import RTDETR
import pandas as pd
import numpy as np
import os
from ensemble_boxes import *

TEST = True
FILTERING = False
AUGMENT = True
RUN = os.environ.get("Z_RUN_1", '')
RUN2 = os.environ.get("Z_RUN_2", '')
RUN3 = 103 #28
RUN4 = 104 #28

sub = pd.read_csv("data/SampleSubmission.csv")
test = pd.read_csv("data/Test.csv")

loc = f'runs/detect/train{RUN}'
loc2 = f'runs/detect/train{RUN2}'


OVERLAP = 0.25
THRESH = 0.1
SZ = (640,640)
SZ2 = (640,640)
"""
Tuning iou_thr:
Lower values may lead to more aggressive merging, which can help when models predict boxes that are slightly shifted but represent the same object.
Higher values ensure stricter grouping but may split detections that are slightly offset.
"""
IOU_THRESH = 0.65 # 0.7

data_dir = os.environ["Z_DATA_DIR"]
image_parent = os.environ.get("Z_IMAGE_PARENT")
test_dir  = f'{image_parent}/images'
val_dir   = f'{data_dir}/val/images/2017'
model_path = f'{data_dir}/{loc}/weights/best.pt'
model_path2 = f'{data_dir}/{loc2}/weights/best.pt'


img_id = ""
#sub.loc[sub['image_id'] == img_id, 'Target'] = 0 # place holder

test_imgs = test.Image_ID.to_list()


model = RTDETR(model_path) #RTDETR
model2 = RTDETR(model_path2)


def get_areas(boxes):
    areas = []
    for b in boxes:
        arr = abs(b[2] - b[0]) * abs(b[3] - b[1])
        areas.append(arr)
    return areas

def get_results(res, threshold=0.5):
    res     = res.cpu().numpy()
    cls     = list(res.boxes.cls)
    conf    = list(res.boxes.conf)
    boxes   = res.boxes.xyxyn.tolist()

    #areas   = get_areas(res.boxes.xyxy.tolist())
    ccls = []
    bbxs = []
    cnfs = []
    #confs = []
    for i in range(len(cls)):
        if conf[i] >= threshold:
            ccls.append(int(cls[i]))
            bbxs.append(boxes[i])
            cnfs.append(conf[i])
    bbxs = np.array(bbxs)
    cnfs = np.array(cnfs)
    ccls = np.array(ccls)
    return ccls, bbxs, cnfs

def overlap_ratio(rect1, rect2):
    # Unpack the coordinates
    xmin1, ymin1, xmax1, ymax1 = rect1
    xmin2, ymin2, xmax2, ymax2 = rect2

    # Calculate the coordinates of the intersection rectangle
    xmin_inter = max(xmin1, xmin2)
    ymin_inter = max(ymin1, ymin2)
    xmax_inter = min(xmax1, xmax2)
    ymax_inter = min(ymax1, ymax2)

    # Compute the area of the intersection
    inter_width = max(0, xmax_inter - xmin_inter)
    inter_height = max(0, ymax_inter - ymin_inter)
    area_inter = inter_width * inter_height

    # Compute the area of the reference rectangle (choose the first rectangle as reference)
    area_rect1 = (xmax1 - xmin1) * (ymax1 - ymin1)

    # If there is no intersection, the overlap ratio is 0
    if area_inter == 0:
        return 0.0

    # Calculate the overlap ratio
    overlap_ratio = area_inter / area_rect1

    return overlap_ratio

def filter_non_overlapping_boxes(boxes, iou_threshold=OVERLAP):
    """Filters a list of bounding boxes, keeping only those that do not overlap.

    Args:
        boxes: A list of numpy arrays, each representing a bounding box in xyxy format.
        iou_threshold: The threshold for determining if two bounding boxes overlap.

    Returns:
        A tuple of two lists:
        - The list of non-overlapping bounding boxes.
        - The corresponding indices of the non-overlapping boxes in the original list.
    """

    filtered_boxes = []
    filtered_indices = []
    for i, box in enumerate(boxes):
        is_overlapping = False
        for filtered_box in filtered_boxes:
            iou = overlap_ratio(box, filtered_box)
            if iou > iou_threshold:
                is_overlapping = True
                break
        if not is_overlapping:
            filtered_boxes.append(box)
            filtered_indices.append(i)

    return filtered_boxes, filtered_indices

def run_nms(bboxes, confs,labels, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None):
    boxes =  [bbox for bbox in bboxes]
    scores = [conf for conf in confs]
    #labels = [np.ones(conf.shape[0]) for conf in confs]
    boxes, scores, labels = nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
    #boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_nmw(bboxes, confs,labels, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None):
    boxes =  [bbox for bbox in bboxes]
    scores = [conf for conf in confs]
    #labels = [np.ones(conf.shape[0]) for conf in confs]
    boxes, scores, labels = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
    #boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_soft_nms(bboxes, confs,labels, image_size, iou_thr=0.50, skip_box_thr=0.0001,sigma=0.1, weights=None):
    boxes =  [bbox for bbox in bboxes]
    scores = [conf for conf in confs]
    #labels = [np.ones(conf.shape[0]) for conf in confs]
    boxes, scores, labels = soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    #boxes = boxes*(image_size-1)
    return boxes, scores, labels

def run_wbf(bboxes, confs,labels, image_size, iou_thr=0.50, skip_box_thr=0.0001, weights=None):
    #boxes =  [bbox/(image_size-1) for bbox in bboxes]
    boxes =  [bbox for bbox in bboxes]
    scores = [conf for conf in confs]
    #labels = [np.ones(conf.shape[0]) for conf in confs]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    #boxes = boxes*(image_size-1)
    return boxes, scores, labels

loc_str = loc.replace("/","_")
names_ = {
        'Corn_Cercospora_Leaf_Spot': 0,
        'Tomato_Septoria': 1,
        'Tomato_Late_Blight': 2,
        'Corn_Streak': 3,
        'Tomato_Healthy': 4,
        'Pepper_Septoria': 5,
        'Pepper_Leaf_Mosaic': 6,
        'Tomato_Early_Blight': 7,
        'Pepper_Bacterial_Spot': 8,
        'Corn_Common_Rust': 9,
        'Corn_Healthy': 10,
        'Pepper_Leaf_Curl': 11,
        'Tomato_Fusarium': 12,
        'Pepper_Healthy': 13,
        'Pepper_Late_Blight': 14,
        'Pepper_Leaf_Blight': 15,
        'Tomato_Bacterial_Spot': 16,
        'Pepper_Cercospora': 17,
        'Pepper_Fusarium': 18,
        'Tomato_Leaf_Curl': 19,
        'Corn_Northern_Leaf_Blight': 20,
        'Tomato_Mosaic': 21,
        'Pepper_Early_Blight': 22,
    }
names = {v: k for k, v in names_.items()}
if TEST:
    print("Prediction Started")
    imgs = []
    classes = []
    confs = []
    ymins = []
    ymaxs = []
    xmins = []
    xmaxs = []

    for test_img in test_imgs:
        t_classes = []
        t_confs = []
        t_ymins = []
        t_ymaxs = []
        t_xmins = []
        t_xmaxs = []
        print(".",end="",flush=True)
        results = model(f"{test_dir}/{test_img}", imgsz = SZ, augment=AUGMENT)
        results2 = model2(f"{test_dir}/{test_img}", imgsz = SZ2, augment=AUGMENT)
        #results3 = model3(f"{test_dir}/{test_img}", imgsz = SZ, augment=AUGMENT)
        #results4 = model4(f"{test_dir}/{test_img}", imgsz = SZ, augment=AUGMENT)
        for res, res2, in zip(results, results2):
            cls, bxs, cnfs = get_results(res, threshold=THRESH)
            cls2, bxs2, cnfs2 = get_results(res2, threshold=THRESH)
            #cls3, bxs3, cnfs3 = get_results(res3, threshold=THRESH)
            #cls4, bxs4, cnfs4 = get_results(res4, threshold=THRESH)

            if not cls.tolist() and not cls2.tolist(): # and not cls3.tolist(): # and not cls4.tolist():
                imgs.append(test_img)
                classes.append("NEG")
                confs.append(1)
                ymins.append(0)
                ymaxs.append(0)
                xmins.append(0)
                xmaxs.append(0)
            else:
                if FILTERING:
                    kept_bxs, idxs = filter_non_overlapping_boxes(bxs, iou_threshold=0.15)
                    kept_cls = [cls[i] for i in idxs]
                    kept_cnfs = [cnfs[i] for i in idxs]
                    kept_bxs2, idxs2 = filter_non_overlapping_boxes(bxs2, iou_threshold=0.15)
                    kept_cls2 = [cls2[i] for i in idxs2]
                    kept_cnfs2 = [cnfs2[i] for i in idxs2]
                else:
                    kept_cls = cls
                    kept_bxs = bxs
                    kept_cnfs = cnfs

                    # Filter out classes 'Pepper_Septoria': 5, and 'Pepper_Late_Blight': 14,
                    #idx = np.where((cls2 != 5) & (cls2 != 14))[0]
                    #cls2 = cls2[idx]
                    #bxs2 = bxs2[idx]
                    #cnfs2 = cnfs2[idx]

                    kept_cls2 = cls2
                    kept_bxs2 = bxs2
                    kept_cnfs2 = cnfs2



                    #kept_cls3 = cls3
                    #kept_bxs3 = bxs3
                    #kept_cnfs3 = cnfs3
                    #kept_cls4 = cls4
                    #kept_bxs4 = bxs4
                    #kept_cnfs4 = cnfs4
                    if test_img == "Xid_14tfmb.jpg":
                        pdb.set_trace()
                    # WBF
                    kept_bxs[kept_bxs > 1] = 0.9999
                    kept_bxs[kept_bxs < 0] = 0.0001
                    kept_bxs2[kept_bxs2 > 1] = 0.9999
                    kept_bxs2[kept_bxs2 < 0] = 0.0001
                    #kept_bxs3[kept_bxs3 > 1] = 0.9999
                    #kept_bxs3[kept_bxs3 < 0] = 0.0001
                    #kept_bxs4[kept_bxs4 > 1] = 0.9999
                    #kept_bxs4[kept_bxs4 < 0] = 0.0001

                kept_bxs, kept_cnfs, kept_cls = run_wbf(
                                                    [kept_bxs, kept_bxs2 ],
                                                    [kept_cnfs, kept_cnfs2],
                                                    [kept_cls, kept_cls2],
                                                    SZ[0], iou_thr=IOU_THRESH
                )

                kept_bxs[:,0] = kept_bxs[:,0] * res.orig_shape[1]
                kept_bxs[:,2] = kept_bxs[:,2] * res.orig_shape[1]
                kept_bxs[:,1] = kept_bxs[:,1] * res.orig_shape[0]
                kept_bxs[:,3] = kept_bxs[:,3] * res.orig_shape[0]

                kept_bxs = kept_bxs.tolist()
                kept_cls = kept_cls.tolist()
                kept_cnfs = kept_cnfs.tolist()
                for kpt_cl,bx,cnf in zip(kept_cls,kept_bxs,kept_cnfs):
                    #if cnf < THRESH:
                    #    t_classes.append(names[int(kpt_cl)])
                    #    t_confs.append(cnf)
                    #    t_ymins.append(int(bx[1]))
                    #    t_ymaxs.append(int(bx[3]))
                    #    t_xmins.append(int(bx[0]))
                    #    t_xmaxs.append(int(bx[2]))
                    #    continue
                    imgs.append(test_img)
                    #pdb.set_trace()
                    classes.append(names[int(kpt_cl)])
                    confs.append(cnf)
                    ymins.append(int(bx[1]))
                    ymaxs.append(int(bx[3]))
                    xmins.append(int(bx[0]))
                    xmaxs.append(int(bx[2]))
        #if test_img == "id_5sfgyq.jpg":
        #    pdb.set_trace()
        #if test_img not in imgs:
        #    print(f"[DEBUG] - Image {test_img} has low confidence boxes")
        #    imgs.append(test_img)
        #    classes.extend(t_classes)
        #    confs.extend(t_confs)
        #    ymins.extend(t_ymins)
        #    ymaxs.extend(t_ymaxs)
        #    xmins.extend(t_xmins)
        #    xmaxs.extend(t_xmaxs)
    print()
sub = pd.DataFrame({
    'Image_ID': imgs,
    'class': classes,
    'confidence': confs,
    'ymin': ymins,
    'xmin': xmins,
    'ymax': ymaxs,
    'xmax': xmaxs
})

def filter_confidence(group):
    # Check if there are rows with confidence >= 0.1
    has_high_confidence = (group['confidence'] >= 0.1).any()
    if has_high_confidence:
        # Keep rows with confidence >= 0.1 or remove rows with confidence < 0.1
        return group[group['confidence'] >= 0.1]
    else:
        # If no rows with confidence >= 0.1, keep all rows
        return group
#sub = sub.groupby('Image_ID', group_keys=False).apply(filter_confidence)
#pdb.set_trace()
sub.to_csv(f"{data_dir}/sub_{loc_str}_{RUN2}_ensemble.csv", index=False)

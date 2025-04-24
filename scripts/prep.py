import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
from PIL import Image
import yaml
import pdb


create_test  = False
create_val   = True
create_labels= True
create_train = True

def is_unique_crop(cls):
    names = [c.split("_")[0] for c in cls]
    return len(set(names))

def get_unique_crop(cls):
    names = [c.split("_")[0] for c in list(cls)]
    return set(names)

def list0(x):
    return '-'.join(list(x))

def get_crop_from_set(cls):
    return list(cls)[0]

def prep_folds():
    df = pd.read_csv("data/Train.csv")
    df_new = df.groupby('Image_ID').agg({'class': list, 'confidence': list, 'ymin': list, 'xmin': list, 'ymax': list, 'xmax': list}).reset_index()
    df_new["class_unique"] = df_new["class"].apply(lambda x: set(x))
    df_new["class_count"] = df_new["class_unique"].apply(lambda x: len(x))
    df_new["class_unique_crop"] = df_new["class_unique"].apply(is_unique_crop)
    df_new["unique_crop_in_pic"] = df_new["class_unique"].apply(get_unique_crop)
    df_new["len_unique_crop_in_pic"] = df_new["unique_crop_in_pic"].apply(len)

    for i in range(len(df_new)):
        if df_new["len_unique_crop_in_pic"][i] == 2:
            crop = list(df_new.at[i, "unique_crop_in_pic"])[-1]
            df_new.at[i, "unique_crop_in_pic"] = {crop}
            ind = []
            ccls = []
            cls = df_new.at[i, "class"]
            for j in range(len(cls)):
                if crop in cls[j]:
                    ind.append(j)
                    ccls.append(cls[j])
            df_new.at[i, "class"] = ccls
            df_new.at[i, "confidence"] = [df_new.at[i, "confidence"][j] for j in ind]
            df_new.at[i, "ymin"] = [df_new.at[i, "ymin"][j] for j in ind]
            df_new.at[i, "xmin"] = [df_new.at[i, "xmin"][j] for j in ind]
            df_new.at[i, "ymax"] = [df_new.at[i, "ymax"][j] for j in ind]
            df_new.at[i, "xmax"] = [df_new.at[i, "xmax"][j] for j in ind]
            df_new.at[i, "class_unique"] = {ccls[0]}
            df_new.at[i, "class_count"] = 1

    df_new["unique_crop_in_pic"] = df_new["unique_crop_in_pic"].apply(get_crop_from_set)
    df_new['class_unique_list'] = df_new.class_unique.apply(list0)

    df_new = df_new.reset_index(drop=True)
    # Define the number of folds
    n_splits = int(os.environ.get("Z_N_SPLITS", 24))

    # Get the target variable (class column)
    target = df_new["unique_crop_in_pic"]
    df_new["fold"] = -1
    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits)

    # Perform stratified k-fold cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(df_new, target)):
        df_new.loc[test_index,"fold"] = fold

    return df_new

def get_bb(df, img):
    sdf = df[df.Image_ID == img]
    boxes = list(sdf.bbox)
    return boxes, sdf.category_id.unique()[0]

def bb_len(box):
    try:
        l= len(box)
    except:
        l = 0
    return int(l)

def create_labels_func(df, label_dir, image_dir):
    labels = []
    count = 0
    _len = len(df)
    cat_dict = {
        "anthracnose": 0,
        "cssvd": 1,
        "healthy": 2,
    }

    for i in range(len(df)):
        print(".",end="",flush=True)
        img = df.Image_ID.iloc[i]
        cats = df["class"].iloc[i]
        #cat = cat_dict[df["class_s"].iloc[i]]
        label_path = f'{label_dir}/{img.replace(".jpg","")}.txt'
        image_path = f'{image_dir}/{img}'

        ymin = df.iloc[i].ymin
        xmin = df.iloc[i].xmin
        ymax = df.iloc[i].ymax
        xmax = df.iloc[i].xmax

        #pdb.set_trace()
        boxes = []
        for i in range(len(ymin)):
            box = [xmin[i], ymin[i], xmax[i], ymax[i]]
            boxes.append(box)
        image = Image.open(image_path)
        imgx, imgy = image.size
        lbl = ''
        #pdb.set_trace()
        for i,bb in enumerate(boxes):
            #print(cats[i])
            cat = int(cat_dict[cats[i]])

            #bb = eval(b)
            xcenter = ((bb[0] +  bb[2]) / 2) / imgx
            ycenter = ((bb[1] +  bb[3]) / 2) / imgy
            width = (bb[2] - bb[0]) / imgx
            height = (bb[3] - bb[1]) / imgy
            lbl += f'{cat} {xcenter:.4} {ycenter:.4} {width:.4} {height:.4}\n'
        if lbl:
            with open(label_path, 'w') as F:
                F.write(lbl)
        else:
            pdb.set_trace()
            a = None

        count += 1
        if count % 250 == 0:
            print(f"\n[INFO] - {count} of {_len} done.\n", flush=True)

data_dir = os.environ["Z_DATA_DIR"]
image_parent = os.environ.get("Z_IMAGE_PARENT")
fold = 0

image_dir = f'{image_parent}/images'
train_dir = f'{data_dir}/train-fold-new{fold}/images/2017'
tlabel_dir= f'{data_dir}/train-fold-new{fold}/labels/2017'
val_dir   = f'{data_dir}/val-fold-new{fold}/images/2017'
vlabel_dir= f'{data_dir}/val-fold-new{fold}/labels/2017'
test_dir  = f'{data_dir}/test'

train = prep_folds()
#train["sc"] = train["class"].apply(lambda x: list(set(eval(x)))[0])
test = pd.read_csv("data/Test.csv")

#from sklearn.model_selection import train_test_split
#train_df, valid_df = train_test_split(train, test_size=0.1, random_state=42, stratify=train["sc"])
train_df = train[train.fold != fold]
valid_df = train[train.fold == fold]
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

if create_test:
    print("\n\nCreating test set")
    os.makedirs(test_dir, exist_ok=True)
    test_images = list(test.Image_ID.unique())
    _len = len(test_images)
    count = 0
    for im in test_images:
        print(".",end="",flush=True)
        cmd = f"cp {image_dir}/{im} {test_dir}"
        os.system(cmd)
        count += 1
        if count % 250 == 0:
            print(f"\n[INFO] - {count} of {_len} moved.", flush=True)

print()

val_fold = 0

if create_train:
    print("\n\nCreating train set")
    os.makedirs(train_dir, exist_ok=True)
    train_images = list(train_df.Image_ID.unique())
    _len = len(train_images)
    count = 0
    for im in train_images:
        print(".",end="",flush=True)
        cmd = f"cp {image_dir}/{im} {train_dir}"
        os.system(cmd)
        count += 1
        if count % 250 == 0:
            print(f"\n[INFO] - {count} of {_len} moved.", flush=True)

if create_val:
    print("\n\nCreating val set")
    os.makedirs(val_dir, exist_ok=True)
    val_images = list(valid_df.Image_ID.unique())
    _len = len(val_images)
    count = 0
    for im in val_images:
        print(".",end="",flush=True)
        cmd = f"cp {image_dir}/{im} {val_dir}"
        os.system(cmd)
        count += 1
        if count % 250 == 0:
            print(f"\n[INFO] - {count} of {_len} moved.", flush=True)

if create_labels:
    print("\n\nCreating Train labels")
    df = train_df
    os.makedirs(tlabel_dir, exist_ok=True)
    create_labels_func(df, tlabel_dir, train_dir)
    del df
    print("\n\nCreating Val labels")
    df = valid_df
    os.makedirs(vlabel_dir, exist_ok=True)
    create_labels_func(df, vlabel_dir, val_dir)


config = {
    'names': [
        "anthracnose": 0,
        "cssvd": 1,
        "healthy": 2,
    ],
    'nc': 3,
    'train': f'{train_dir}',
    'val': f'{val_dir}'
}

with open(f'data{fold}.yaml', 'w') as f:
    yaml.dump(config, f)

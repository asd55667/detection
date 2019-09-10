import pandas as pd 
import csv, random
import numpy as np

ROOT_PATH =  '/media/wcw/TOSHIBA_X/OID/'
INPUT_PATH = ROOT_PATH + 'input/'
OUTPUT_PATH = ROOT_PATH + 'output/'
MODELS_PATH = ROOT_PATH + 'models/'
SUB_PATH = ROOT_PATH + 'sub/'
level_1_dir = OUTPUT_PATH + '/level_1_files/'



def get_image_sizes(subset):
    sizes = pd.read_csv(OUTPUT_PATH + subset + '_image_params.csv')
    ret = {}
    ids = sizes['id'].values
    ws = sizes['width'].values
    ht = sizes['height'].values
    for i in range(len(ids)):
        ret[ids[i]] = (int(ws[i]), int(ht[i]))
    return ret


def get_labels(metadata_dir):
    csv_file = 'class-descriptions-boxable-level-1.csv'

    boxable_classes_descriptions = os.path.join(metadata_dir, csv_file)
    id_to_labels = {}
    cls_index    = {}

    i = 0
    with open(boxable_classes_descriptions) as f:
        for row in csv.reader(f):
            # make sure the csv row is not empty (usually the last one)
            if len(row):
                label       = row[0]
                description = row[1].replace("\"", "").replace("'", "").replace('`', '')
                id_to_labels[i]  = description
                cls_index[label] = i
                i += 1
    return id_to_labels, cls_index

def generate_images_annotations_json(subset, sampling=1):
    level_1_csv = level_1_dir + subset + '-id_annotations-bbox-level-1.csv'

    fieldnames = ['ImageID', 'Source', 'LabelName', 'Confidence',
                'XMin', 'XMax', 'YMin', 'YMax',
                'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
    
    if subset=='train' and sampling:
        cls2imgs = pickle.load(open(OUTPUT_PATH+'cls2imgs.pkl', 'rb'))
        n = random.randint(500, 1000)
        train = set()
        for imgs in cls2imgs.values():
            if len(imgs) > n:
                idxs = np.random.randint(len(imgs), size=(n))
                for idx in idxs:
                    train.add(imgs[idx])
            else:
                train.update(imgs)            
        train = list(train)


    with open(level_1_csv, 'r') as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        id_annotations = {}
        next(reader)

        _, cls_index = get_labels(level_1_dir)
        img2size = get_image_sizes(subset)

        for i, row in enumerate(reader):
            img_id = row['ImageID']

            x1 = float(row['XMin'])
            x2 = float(row['XMax'])
            y1 = float(row['YMin'])
            y2 = float(row['YMax'])

            x1_int = int(round(x1 * width))
            x2_int = int(round(x2 * width))
            y1_int = int(round(y1 * height))
            y2_int = int(round(y2 * height))
            
            cls_id = cls_index[class_name]



            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            if y2_int == y1_int:
                warnings.warn('filtering line {}: rounding y2 ({}) and y1 ({}) makes them equal'.format(line, y2, y1))
                continue

            if x2_int == x1_int:
                warnings.warn('filtering line {}: rounding x2 ({}) and x1 ({}) makes them equal'.format(line, x2, x1))
                continue            
            
            annotation = {'cls_id': cls_id, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
            
            if img_id in id_annotations:
                id_annotations = id_annotations[img_id]
                id_annotations['boxes'].append(annotation)
            else:
                id_annotations[img_id] = {'w': width, 'h': height, 'boxes': [annotation]}


            if subset == 'train':
                if sampling and img_id not in train:
                    continue                            
                pths = os.listdir(os.path.join(ROOT_PATH, subset))
                for pth in pths:
                    img_path = os.path.join(os.path.join(ROOT_PATH, subset) ,pth, img_id + '.jpg')
                    if os.path.isfile(img_path):
                        break
            else:
                img_path = os.path.join(ROOT_PATH, subset, img_id + '.jpg')
            if not os.path.isfile(img_path):
                continue

            id_annotations[img_id] = {'w': img2size[img_id][0], 'h': img2size[img_id][1], 'bbox': [annotation], 'p': img_path}
    return id_annotations




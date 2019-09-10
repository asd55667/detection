import pandas as pd
import os, pickle


train_df = pd.read_csv(level1_dir + 'train-annotations-bbox-level-1.csv')
bboxes_per_img = train_df.groupby(['ImageID']).count()[['LabelName']]
zero_bbox_img = bboxes_per_img[bboxes_per_img['LabelName']==0].index
pd.Series(zero_bbox_img).to_csv(OUTPUT_PATH+'0bboxes_img.csv', sep='\n',index=False)

img2id = {}
id2img = {}
for i, img in enumerate(bboxes_per_img.index):
    img2id[img] = i
    id2img[i] = img
pickle.dump(img2id, open(OUTPUT_PATH+'img2id.pkl', 'wb'))
pickle.dump(id2img, open(OUTPUT_PATH+'id2img.pkl', 'wb'))

label_count = train_df.groupby('LabelName').count()
cls2imgs = {}
for label in label_count.index:
    cls2imgs[label] = list(set(train_df[train_df['LabelName'] == label]['ImageID']))
pickle.dump(cls2imgs, open(OUTPUT_PATH+'cls2imgs.pkl', 'wb'))

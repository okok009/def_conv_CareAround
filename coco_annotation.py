import json


coco_path = 'COCODevKit/annotations/instances_val2017.json'

data_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/val2017/'

save_path = 'val2017.txt'

with open(coco_path) as f:
    ann = json.load(f)

image_id = []
information = []
inf = ''

for i in range(len(ann['annotations'])):
    if str(ann['annotations'][i]['image_id']).zfill(12) not in image_id:
        image_id.append(str(ann['annotations'][i]['image_id']).zfill(12))                  #zfill補零
        for j in range(len(ann['annotations'])):
            if ann['annotations'][i]['image_id'] == ann['annotations'][j]['image_id']:
                coco_bbox = list(map(int, ann['annotations'][j]['bbox']))
                category_id = ann['annotations'][j]['category_id']
                inf += str(coco_bbox[0]) + ',' + str(coco_bbox[1]) + ',' + str(coco_bbox[0]+coco_bbox[2]) + ',' + str(coco_bbox[1]+coco_bbox[3]) + ',' + str(category_id) + ' '
        information.append(inf)
        inf = ''

with open(save_path, 'a') as train_f:
    for i in range(len(image_id)):
        train_f.write(data_path + image_id[i] + '.jpg ')
        train_f.write(information[i] + '\n')



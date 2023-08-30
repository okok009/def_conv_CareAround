import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

def getmeanstd(droot):
    num = 0
    for file in os.listdir(droot):
        img = cv2.imread(droot+"/"+file)####bgr
        s_meanstd = [np.mean(img[:,:,0]),np.mean(img[:,:,1]),np.mean(img[:,:,2]),np.std(img[:,:,0]),np.std(img[:,:,1]),np.std(img[:,:,2])]
        s_h,s_w,c = img.shape
        if num == 0:
            num = s_h*s_w
            bgr_meanstd = s_meanstd
            continue
        bgrmmean=[bgr_meanstd[0],bgr_meanstd[1],bgr_meanstd[2]]
        bgr_meanstd[0] = (bgr_meanstd[0]*num+s_meanstd[0]*s_h*s_w)/(num+s_h*s_w)
        bgr_meanstd[1] = (bgr_meanstd[1]*num+s_meanstd[1]*s_h*s_w)/(num+s_h*s_w)
        bgr_meanstd[2] = (bgr_meanstd[2]*num+s_meanstd[2]*s_h*s_w)/(num+s_h*s_w)
        bgr_meanstd[3] = ((num*(bgrmmean[0]**2+bgr_meanstd[3]**2)+s_h*s_w*(s_meanstd[0]**2+s_meanstd[3]**2))/(num+s_h*s_w)-bgr_meanstd[0]**2)**0.5
        bgr_meanstd[4] = ((num*(bgrmmean[1]**2+bgr_meanstd[4]**2)+s_h*s_w*(s_meanstd[1]**2+s_meanstd[4]**2))/(num+s_h*s_w)-bgr_meanstd[1]**2)**0.5
        bgr_meanstd[5] = ((num*(bgrmmean[2]**2+bgr_meanstd[5]**2)+s_h*s_w*(s_meanstd[2]**2+s_meanstd[5]**2))/(num+s_h*s_w)-bgr_meanstd[2]**2)**0.5
        num += s_h*s_w
        
    return bgr_meanstd

if __name__ == "__main__":
    import os
    import cv2
    import numpy as np
    from torch.utils.data import Dataset
    import torch

    # VOC2007_dataset
    mean = [0.406*255,0.456*255,0.485*255]
    std = [0.225*255,0.224*255,0.229*255]

    # coco_dataset
    bgr_meanstd = [103.86630061083278, 113.90733757212305, 119.94535563375436, 73.59719321974762, 69.94483734161967, 71.08511273123122]
    mean = [0.407*255, 0.446*255, 0.47*255]
    std = [0.288*255, 0.274*255, 0.278*255]

    unknowmeanstd = False
    if unknowmeanstd == True:
        bgr_meanstd = getmeanstd("COCODevKit/train2017")
        mean = [bgr_meanstd[0],bgr_meanstd[1],bgr_meanstd[2]]
        std = [bgr_meanstd[3],bgr_meanstd[4],bgr_meanstd[5]]
        print(bgr_meanstd)

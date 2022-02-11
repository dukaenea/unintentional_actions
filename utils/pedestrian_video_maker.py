
# @Author: Enea Duka
# @Date: 8/10/21

import cv2
import os
from tqdm import tqdm

video_path = '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/videos'
images_path = ''

def create_videos(mode):
    store_path = os.path.join(video_path, mode)
    ped1_path = '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/%s' % mode
    ped2_path = '/BS/unintentional_actions/nobackup/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped2/%s' % mode

    ped1_vids = [f[0] for f in os.walk(ped1_path) if not f[0].endswith('_gt')][1:]
    ped2_vids = [f[0] for f in os.walk(ped2_path) if not f[0].endswith('_gt')][1:]

    vids = ped1_vids + ped2_vids

    for video in tqdm(vids):
        img_paths = [f[2] for f in os.walk(video)][0]
        img_paths.sort()
        ped_set = video.split('/')[-3][-1]
        video_name = video.split('/')[-1]+'_'+ped_set+'.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        img = cv2.imread(os.path.join(video, img_paths[0]))
        (w, h) = img.shape[:2]
        out = cv2.VideoWriter(os.path.join(store_path, video_name),
                              fourcc, 10.0, (w, h))

        for image_path in img_paths:
            image = cv2.imread(os.path.join(video, image_path), -1)
            image = cv2.resize(image, (w, h))
            out.write(image)

        out.release()




if __name__ == '__main__':
    create_videos('Train')
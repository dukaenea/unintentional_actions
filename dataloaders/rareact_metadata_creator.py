# @Author: Enea Duka
# @Date: 4/19/21

from csv import DictReader, DictWriter
import ffmpeg
import os
from tqdm import tqdm

def save_split(split_dict, mode, split_path, videos_path):

    video_directory = '/BS/unintentional_actions/nobackup/rareact/data/%s' % mode
    labels = []
    samples_mdata = []
    same_name_videos = {}
    for key in tqdm(split_dict.keys()):
        samples = split_dict[key]
        try:
            label = samples[0]['verb']+'_'+samples[0]['noun']
        except IndexError:
            print(samples)
            print(key)
        # out_path = video_directory+'/'+label
        # if not os.path.exists(out_path):
        #     os.mkdir(out_path)
        labels.append(label)

        for sample in samples:
            if sample['annotation'] == '1' or sample['annotation'] == '0':
                if sample['video_id'] not in list(same_name_videos.keys()):
                    same_name_videos[sample['video_id']] = 0
                else:
                    same_name_videos[sample['video_id']] += 1
                video_name = sample['video_id'] + str(same_name_videos[sample['video_id']])
                in_path_sample = videos_path+'/'+sample['video_id']+'.mp4'
                out_path_sample = video_directory + '/' + video_name + '.mp4'
                start = sample['start']
                end = sample['end']
                command = 'ffmpeg -hide_banner -loglevel error -i '+in_path_sample+' -ss '+start+' -to '+end+' -c copy '+out_path_sample
                if not os.path.isfile(out_path_sample):
                    os.system(command)
                if os.path.isfile(out_path_sample):
                    samples_mdata.append({'filename': video_name, 'label': label})
                else:
                    print('Error on video %s' % video_name)

    mdata_out = split_path+'/'+mode+'.csv'
    classes_out = split_path+'/'+mode+'_classes.txt'

    with open(mdata_out, 'w', newline='') as mf:
        d_writer = DictWriter(mf, samples_mdata[0].keys())
        d_writer.writeheader()
        d_writer.writerows(samples_mdata)

    with open(classes_out, 'w') as cf:
        for idx, c in enumerate(labels):
            cf.write('%d %s\n' % (idx, c))



if __name__ == '__main__':
    rareact_csv_path = '/BS/unintentional_actions/nobackup/rareact/rareact.csv'
    rareact_split_path = '/BS/unintentional_actions/work/data/rareact/splits'
    vid_path = '/BS/unintentional_actions/nobackup/rareact/data'

    with open(rareact_csv_path) as f:
        rareact_csv = DictReader(f)
        class_sample_dict = {}

        for row in rareact_csv:
            class_id = row['class_id']

            if class_id in class_sample_dict.keys():
                class_sample_dict[class_id].append(row)
            else:
                class_sample_dict[class_id] = [row]

        train_dict = {}
        val_dict = {}
        test_dict = {}

        for key in class_sample_dict.keys():
            class_samples = class_sample_dict[key]
            nr_samples = len(class_samples)

            # if nr_samples > 10:
            train_dict[key] = class_samples[:int(nr_samples * 0.8)]
            val_dict[key] = class_samples[int(nr_samples * 0.8):]
                # test_dict[key] = class_samples[int(nr_samples * 0.8):]

        # print(train_dict)
        save_split(train_dict, 'train', rareact_split_path, vid_path)
        save_split(val_dict, 'val', rareact_split_path, vid_path)
        # save_split(test_dict, 'test', rareact_split_path, vid_path)




















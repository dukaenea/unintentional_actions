# @Author: Enea Duka
# @Date: 5/14/21

from csv import DictReader, DictWriter
from tqdm import tqdm
import os.path as ops
from pprint import pprint


def subsample(split, m_samples=250, prev_classes=None):
    csv_file = '/BS/unintentional_actions/work/data/kinetics/splits/%s.csv' % split
    output_dir = '/BS/unintentional_actions/work/data/kinetics/splits/'
    selected_classes = {} if prev_classes is None else prev_classes
    samples = []
    max_classes = 100
    max_samples = m_samples

    with open(csv_file) as f:
        vid_csv = DictReader(f)

        for row in tqdm(vid_csv):
            label = row['label']
            if label in selected_classes.keys():
                if selected_classes[label] < max_samples:
                    selected_classes[label] += 1
                    samples.append(row)
            else:
                if len(selected_classes.keys()) < max_classes:
                    selected_classes[label] = 1
                    samples.append(row)

        if prev_classes is None:
            with open(ops.join(output_dir, 'rep_lrn_classes') + '.txt', 'w') as cf:
                for idx, c in enumerate(selected_classes):
                    cf.write(str(idx) + ' %s\n' % c.replace(' ', '_'))
        with open(ops.join(output_dir, '%s_rep_lrn' % split) + '.csv', 'w', newline='') as mf:
            d_writer = DictWriter(mf, samples[0].keys())
            d_writer.writeheader()
            d_writer.writerows(samples)

    return selected_classes


def zero_out_classes(classes):
    for key in classes.keys():
        classes[key] = 0
    return classes


if __name__ == '__main__':
    classes = subsample('train')
    classes = zero_out_classes(classes)
    subsample('val', 300, classes)
    subsample('test', 300, classes)

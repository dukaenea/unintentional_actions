# @Author: Enea Duka
# @Date: 6/12/21

from csv import DictReader, DictWriter
from random import shuffle

def create_rok_mdata(mode):
    materials = {
        # 'rareact': '/BS/unintentional_actions/work/data/rareact/splits/%s.csv' % mode,
        # 'oops': '/BS/unintentional_actions/work/data/oops/splits/%s_all.csv' % mode,
        'kinetics': '/BS/unintentional_actions/work/data/kinetics/splits/%s_rep_lrn.csv' % mode
    }

    rok_mdata_path = '/BS/unintentional_actions/work/data/rok/data_splits/%s_rep_lrn_kin.csv' % mode
    rok_mdata = []

    for key, material_path in materials.items():
        with open(material_path) as f:
            material_dict = DictReader(f)
            for row in material_dict:
                rok_mdata.append({'filename': row['filename'], 'dataset': key})

    shuffle(rok_mdata)
    with open(rok_mdata_path, 'w', newline='') as mf:
        d_writer = DictWriter(mf, rok_mdata[0].keys())
        d_writer.writeheader()
        d_writer.writerows(rok_mdata)



if __name__ == '__main__':
    create_rok_mdata('train')

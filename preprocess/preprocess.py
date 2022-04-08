import os
import sys
import glob
import argparse

import cv2
import scipy.io
import pandas as pd

import numpy as np
from scipy.interpolate import interp1d

from tqdm import tqdm

def postpone(next_im, passed):
    offset = next_im.shape[0]
    sub_record = []

    passed += 1

    col = next_im[:,0]
    col_values = np.where(col == 255)
    low = np.max(col_values)
    high = np.min(col_values)

    if next_im.shape[1] == 1:
        keep = (low + high) // 2
        sub_record.append(offset - keep)
        return sub_record, keep, passed

    if passed >= 500:
        keep = (low + high) // 2
        sub_record.append(offset - keep)
        return sub_record, keep, passed

    if low == high:
        keep = high
        sub_record.append(offset - keep)
        return sub_record, keep, passed
    else:
        next_record, selected, passed = postpone(next_im[:,1:], passed)
        if low <= selected:
            sub_record.append(offset - low)
            sub_record.extend(next_record)
            return sub_record, low, passed
        elif high >= selected:
            sub_record.append(offset - high)
            sub_record.extend(next_record)
            return sub_record, high, passed
        else:
            #XXX Heuristic (Current): should be modified
            keep = (low + high) // 2
            sub_record.append(offset - keep)
            sub_record.extend(next_record)
            return sub_record, keep, passed 

def extract(im):
    passed = 0
    offset = im.shape[0]
    record = []

    col = im[:, 0]
    col_values = np.where(col == 255)
    keep_low = np.max(col_values)
    keep_high = np.min(col_values)

    sub_record, _, sub_passed = postpone(im[:, passed:], passed)
    record.extend(sub_record)
    passed += sub_passed
    keep = offset - sub_record[-1]

    steps = range(passed, im.shape[1])
    steps_iter = iter(steps)

    for i in steps_iter:
        col = im[:,i]
        col_values = np.where(col == 255)

        if col_values[0].size == 0:
            record.append(-1)
            continue

        low = np.max(col_values)
        high = np.min(col_values)

        if keep > low:
            keep = high
            keep_low = low
            keep_high = high
            record.append(offset - keep)
        elif keep < high:
            keep = low
            keep_low = low
            keep_high = high
            record.append(offset - keep)
        elif keep == high:
            keep = low
            keep_low = low
            keep_high = high
            record.append(offset - keep)
        elif keep == low:
            keep = high
            keep_low = low
            keep_high = high
            record.append(offset - keep)
        else:
            high_offset = high - keep_high
            low_offset = keep_low - low
            if high_offset > low_offset:
                keep = high
                record.append(offset - keep)
            elif high_offset < low_offset:
                keep = low
                record.append(offset - keep)
            else:
                sub_record, _, passed = postpone(im[:,i:], 0)
                record.extend(sub_record)
                keep = offset - sub_record[-1]

                col = im[:,i+passed-1]
                col_values = np.where(col == 255)
                keep_low = np.max(col_values)
                keep_high = np.min(col_values)

                for j in range(passed-1):
                    next(steps_iter, None)

    record = np.array(record)
    return record

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--csv-path", default=".", type=str, metavar="DIR",
        help="path to 'ECG_Datasheet_sess.csv' so that we can "
        "link ECG records to their corresponding sessions"
    )

    return parser

def main(args):
    black_list = ['F61525', 'F97592', 'F41891']
    unit_mV = 2.5

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    
    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "*/")

    sessions = pd.read_csv(args.csv_path)

    fnames = glob.glob(search_path)
    for fname in tqdm(fnames):
        ecg_dir = fname.split('/')[-2]
        sess = sessions[sessions['ECG_folder'] == ecg_dir].sort_values(['ID_sess'])

        if (
            (ecg_dir in black_list)
            or (len(sess) == 0)
        ):
            continue

        sess = sess.iloc[0]
        id_sess, ecg_date_diff, ecg_grid, idh_a, idh_b = (
            sess[['ID_sess', 'ECG_date_diff', 'ECG_grid', 'IDH_a', 'IDH_b']]
        )
        if ecg_date_diff > 7 or ecg_grid not in [83.0, 157.0]:
            continue

        vals = []        
        min_size = sys.maxsize
        for i in range(12):
            img = cv2.imread(os.path.join(fname, f'lead_{i}.png'), cv2.IMREAD_GRAYSCALE)
            val = extract(img)
            if len(val) < min_size:
                min_size = len(val)

            ratio = unit_mV / ecg_grid
            baseline = np.bincount(val).argmax()

            val = val.astype(np.float64)
            val -= baseline
            val *= ratio

            vals.append(val)

        vals = np.array([v[:min_size] for v in vals])

        if ecg_grid == 83:
            sample_size = vals.shape[-1] * (157 / ecg_grid)
            x = np.linspace(0, sample_size - 1, vals.shape[-1])
            func = interp1d(x, vals, kind='linear')
            feats = func(list(range(int(sample_size))))
        else:
            feats = vals

        feats = feats.astype(np.float32)

        data = {'feats': feats}
        data['curr_sample_rate'] = int(ecg_grid)
        data['file_id'] = ecg_dir
        data['ID_sess'] = id_sess
        data['sample_size'] = min_size
        data['idh_a'] = int(idh_a)
        data['idh_b'] = int(idh_b)
        data['idh_ab'] = np.array([int(idh_a), int(idh_b)])

        scipy.io.savemat(os.path.join(args.dest, ecg_dir + '.mat'), data)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
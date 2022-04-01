import argparse
import glob
import os
import random
import scipy.io

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing mat files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.1,
        type=float,
        metavar="D",
        help="percentage of data to use as validation and test set (between 0 and 0.5)"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="mat", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest"
    )
    return parser

def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 0.5

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    rand = random.Random(args.seed)

    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f, open(
        os.path.join(args.dest, "valid.tsv"), "w") as valid_f, open(
        os.path.join(args.dest, "test.tsv"), "w"
    ) as test_f:
        print(dir_path, file=train_f)
        print(dir_path, file=valid_f)
        print(dir_path, file=test_f)

        def write(fnames, dest):
            for fname in fnames:
                file_path = os.path.realpath(fname)

                if args.path_must_contain and args.path_must_contain not in file_path:
                    continue
                
                print (
                    "{}".format(os.path.relpath(file_path, dir_path)), file=dest, end="\t"
                )

                if args.ext == "mat":
                    data = scipy.io.loadmat(file_path)
                    length = data['feats'].shape[-1]
                    print(length, file=dest)
        
        fnames = list(glob.iglob(search_path, recursive=True))
        rand.shuffle(fnames)

        valid_len = int(len(fnames) * args.valid_percent)
        test_len = int(len(fnames) * args.valid_percent)
        train_len = len(fnames) - (valid_len + test_len)

        train = fnames[:train_len]
        valid = fnames[train_len:train_len + valid_len]
        test = fnames[train_len + valid_len:]

        write(train, train_f)
        write(valid, valid_f)
        write(test, test_f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
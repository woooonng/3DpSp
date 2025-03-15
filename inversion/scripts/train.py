import json
import os
import pprint
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from inversion.options.train_options import TrainOptions
from inversion.training.coach import Coach
from tabulate import tabulate
from inversion.training import utils


def main():
    opts = TrainOptions().parse()
    os.makedirs(opts.exp_dir, exist_ok=True)

    utils.set_seed(opts.seed)

    opts_dict = vars(opts)
    print("<TRAINING OPTIONS>")
    print(tabulate(opts_dict.items(), headers=['Option', 'Value'], tablefmt='fancy_grid'))
    with open(os.path.join(opts.exp_dir, 'trainig_opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    coach = Coach(opts)
    coach.train()

if __name__ == '__main__':
    main()

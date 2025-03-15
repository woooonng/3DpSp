from argparse import ArgumentParser


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()

        self.parser.add_argument('--exp_dir', type=str, required=True, help='Path to experiment output directory')
        self.parser.add_argument('--camera', type=bool, default=True, help='Use camera parameter in encoder')
        self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--lpips_lambda', default=0.5, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--lpips_lambda_mirror', default=0.5, type=float,
                                 help='LPIPS loss multiplier factor for mirror loss')
        self.parser.add_argument('--id_lambda', default=0.4, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--id_lambda_mirror', default=0.4, type=float,
                                 help='ID loss multiplier factor for mirror loss')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--l2_lambda_mirror', default=1.0, type=float,
                                 help='L2 loss multiplier factor for mirror loss')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to model checkpoint to continue training')
        self.parser.add_argument('--seed', default=42, type=int, help='Seed number')
        self.parser.add_argument('--train_val_ratio', default=0.1, type=float, help='The ratio of the train data and the validation data')
        self.parser.add_argument('--pretrained_path', default="pretrained_models", type=str, help='Restyle iteration per batch')
        self.parser.add_argument('--n_iters_per_batch', default=5, type=int, help='Restyle iteration per batch')
        self.parser.add_argument('--max_steps', default=100, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=10, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=20, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=20, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=20, type=int, help='Model checkpoint interval')
        self.parser.add_argument('--preprocessed_dataset_path', type=str, required=True,
                                 help='Path to preprocessed dataset directory')
        self.parser.add_argument('--num_workers', default=4, type=int, help='Number of workers in dataloader')

    def parse(self):
        opts = self.parser.parse_args()
        return opts

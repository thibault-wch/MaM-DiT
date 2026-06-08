from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=20,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--epoch_count', type=int, default=0, help='epoch count')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--down_resolution', action='store_true', help='train with the downsampled resolution')
        parser.add_argument('--pretrained_pth', type=str, default=None,
                            help='the pretrained checkpoint name')

        self.isTrain = True
        return parser

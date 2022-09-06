import pathlib

from utils.args import Args


def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    # GAN ARGS
    parser.add_argument('--num-iters-discriminator', type=int, default=1,
                        help='Number of iterations of the discriminator')
    parser.add_argument('--num-z', type=int, default=2,
                        help='Number of z values to use during training.')
    parser.add_argument('--latent-size', type=int, default=512, help='Size of latent vector for z location 2')

    # LEARNING ARGS
    parser.add_argument('--batch-size', default=40, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0, help='Beta 1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.99, help='Beta 2 for Adam')

    # DATA ARGS
    parser.add_argument('--im-size', default=128, type=int,
                        help='Image resolution')
    parser.add_argument('--R', default=16, type=int,
                        help='Image resolution')
    parser.add_argument('--data-parallel', required=True, action='store_true',
                        help='If set, use multiple GPUs using data parallelism')

    # LOGISTICAL ARGS
    parser.add_argument('--device', type=int, default=0,
                        help='Which device to train on. Use idx of cuda device or -1 for CPU')
    #TODO UPDATE EXPDIR
    parser.add_argument('--exp-dir', type=pathlib.Path,
                        default=pathlib.Path('/home/bendel.8/Git_Repos/celeb-inpaint/trained_models'),
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')

    return parser

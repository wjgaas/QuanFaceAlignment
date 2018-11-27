import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='LAB setup')
    parser.add_argument('--data-dir', default=None, help='')
    parser.add_argument('--img-channels', type=int, default=1, help='The number of input image channels')
    parser.add_argument('--tensorboard-path', type=str, default='/home/ubuntu/TB', help='')
    parser.add_argument('--save-params-path', type=str, default='/home/ubuntu/param,1')
    parser.add_argument('--hourglass-channels', type=int, default=256, help='The number of hourglass image channels')
    parser.add_argument('--landmarks', type=int, default=98, help='The number of landmarks')
    parser.add_argument('--boundary', type=int, default=13, help='The number of boundaries')
    parser.add_argument('--per-batch', type=int, default=10, help='Per card batch size')
    parser.add_argument('--epochs', type=int, default=300, help="The number of epochs")
    parser.add_argument('--lr-epoch', type=int, default=200, help="consine end epoch")
    parser.add_argument('--grad-clip', type=float, default=0.1, help='Avoid Gradient Explosion')
    parser.add_argument('--wd',type=float, default=1e-5, help='Weight Decay')

    parser.add_argument('--coe', type=tuple, default=(1, 2e-3, 5e-5), help="Model loss coefficient")
    parser.add_argument('--lr-base', type=tuple, default=(1e-5, 2e-1, 1e-5), help='Learning Rate')
    parser.add_argument('--lr-target', type=tuple, default=(1e-8, 2e-5, 1e-7), help='Learning Rate')
    parser.add_argument('--lr-mode', type=str, default="cosine", help='Learning Rate')
    parser.add_argument('--beta', type=tuple, default=(0.5, 0.999), help='Adam beta tuple')
    parser.add_argument('--mixup-epoch', type=int, default=0, help='Whether to use miuup')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='')

    parser.add_argument('--norm-type', type=str, default='BN', help=' ')
    parser.add_argument('--num-group', type=int, default=8, help='The number of group norm')

    parser.add_argument('--mpl', type=int, default=0, help='Whether to use MPL')
    parser.add_argument('--pretrained', type=int, default=0, help='')
    parser.add_argument('--loss-type', type=int, default=0, help='')
    args = parser.parse_args()
    return args

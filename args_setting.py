import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('FSL Project', add_help=False)

    # dataset setting
    parser.add_argument('--dataset', default="miniImageNet", type=str)
    parser.add_argument('--data_root', default="/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e/FSL_data/", type=str)

    # model setting
    parser.add_argument('--base_model', default='resnet18', type=str)
    parser.add_argument('--channel', default=512, type=int)
    parser.add_argument("--num_classes", default=64, type=int)

    # FSL setting
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--query', default=15, type=int)

    parser.add_argument('--train_episodes', default=500, type=int)
    parser.add_argument('--val_episodes', default=2000, type=int)
    parser.add_argument('--test_episodes', default=2000, type=int)

    # train setting
    parser.add_argument('--fsl', default=True, type=bool, help='whether fsl model')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--img_size', default=80, help='path for save data')
    parser.add_argument('--aug', default=True, help='whether use augmentation')
    parser.add_argument('--use_slot', default=True, type=bool, help='whether use slot module')
    parser.add_argument('--fix_parameter', default=True, type=bool, help='whether fix parameter for backbone')
    parser.add_argument('--double', default=False, type=bool, help='whether double mode')

    # slot setting
    parser.add_argument('--num_slot', default=7, type=int, help='number of slot')
    parser.add_argument('--interval', default=10, type=int, help='skip applied in category sampling')
    parser.add_argument('--drop_dim', default=False, type=bool, help='drop dim for avg')
    parser.add_argument('--slot_base_train', default=True, type=bool, help='drop dim for avg')
    parser.add_argument('--use_pre', default=True, type=bool, help='whether use pre parameter for backbone')
    parser.add_argument('--loss_status', default=1, type=int, help='positive or negative loss')
    parser.add_argument('--hidden_dim', default=64, type=int, help='dimension of to_k')
    parser.add_argument('--slots_per_class', default=1, type=int, help='number of slot for each class')
    parser.add_argument('--power', default=2, type=float, help='power of the slot loss')
    parser.add_argument('--to_k_layer', default=3, type=int, help='number of layers in to_k')
    parser.add_argument('--lambda_value', default="1.", type=str, help='lambda  of slot loss')
    parser.add_argument('--vis', default=False, type=bool, help='whether save slot visualization')
    parser.add_argument('--vis_id', default=0, type=int, help='choose image to visualization')
    parser.add_argument('--DT', default=True, type=bool, help='DT training')
    parser.add_argument('--random', default=False, type=bool, help='whether random select category')

    # data/machine set
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='saved_model',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser
import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open('./saved_models/{}/log_dataset.txt'.format(opt.exp_name), 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print('Skip {} as it is already initialized'.format(name))
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batch-norm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print('loading pretrained model from {}'.format(opt.saved_model))
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
            # model.load_state_dict(torch.load(opt.saved_model, map_location=torch.device("cpu")), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
            # model.load_state_dict(torch.load(opt.saved_model, map_location=torch.device("cpu")))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        # if opt.baiduCTC:
        #     need to install warpctc. see our guideline.
        #     from warpctc_pytorch import CTCLoss
        #     criterion = CTCLoss()
        # else:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open('./saved_models/{}/opt.txt'.format(opt.exp_name), 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += '{}: {}\n'.format(str(k), str(v))
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print('continue to train, start_iter: {}'.format(start_iter))
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while True:
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            # To see training progress, we also conduct validation when 'iteration == 0'
            elapsed_time = time.time() - start_time
            # for log
            with open('./saved_models/{}/log_train.txt'.format(opt.exp_name), 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, _, _ = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # training loss and validation loss
                loss_log = '[{}/{}] Train loss: {:0.5f}, Valid loss: {:0.5f}, Elapsed_time: {:d} s'.format(
                    iteration + 1, opt.num_iter,
                    loss_avg.val() * 1e5,
                    valid_loss * 1e5,
                    int(elapsed_time)
                )
                loss_avg.reset()

                current_model_log = '{:17s}: {:0.3f}, {:17s}: {:0.2f}'.format(
                    "Current_accuracy", current_accuracy,
                    "Current_norm_ED", current_norm_ED
                )

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), './saved_models/{}/best_accuracy.pth'.format(opt.exp_name))
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), './saved_models/{}/best_norm_ED.pth'.format(opt.exp_name))
                best_model_log = '{:17s}: {:0.3f}, {:17s}: {:0.2f}'.format(
                    "Best_accuracy",
                    best_accuracy,
                    "Best_norm_ED",
                    best_norm_ED
                )

                loss_model_log = '{}\n{}\n{}'.format(loss_log, current_model_log, best_model_log)
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = '{:25s} | {:25s} | Confidence Score & T/F'.format("Ground Truth", "Prediction")
                predicted_result_log = '{}\n{}\n{}\n'.format(dashed_line, head, dashed_line)
                
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += '{:25s} | {:25s} | {:0.4f}\t{}\n'.format(
                        gt + " " * (25 - len(gt.encode("sjis"))),
                        pred + " " * (25 - len(pred.encode("sjis"))),
                        confidence,
                        str(pred == gt)
                    )

                predicted_result_log += '{}'.format(dashed_line)
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(model.state_dict(), './saved_models/{}/iter_{}.pth'.format(opt.exp_name, iteration + 1))

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', default="./lmdb/training", help='path to training dataset')             # Modified
    parser.add_argument('--valid_data', default="./lmdb/validation", help='path to validation dataset')         # Modified
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default="None", help='Transformation stage. None|TPS')              # Modified
    parser.add_argument('--FeatureExtraction', type=str, default="VGG",
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')                                                # Modified
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')   # Modified
    parser.add_argument('--Prediction', type=str, default="CTC", help='Prediction stage. CTC|Attn')                     # Modified
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    cfg = parser.parse_args()

    if not cfg.exp_name:
        cfg.exp_name = '{}-{}-{}-{}'.format(
            cfg.Transformation,
            cfg.FeatureExtraction,
            cfg.SequenceModeling,
            cfg.Prediction
        )
        cfg.exp_name += '-Seed{}'.format(cfg.manualSeed)
        # print(cfg.exp_name)

    os.makedirs('./saved_models/{}'.format(cfg.exp_name), exist_ok=True)

    """ vocab / character number configuration """
    if cfg.sensitive:
        # cfg.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        cfg.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", cfg.manualSeed)
    random.seed(cfg.manualSeed)
    np.random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)
    torch.cuda.manual_seed(cfg.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    cfg.num_gpu = torch.cuda.device_count()
    # print('device count', cfg.num_gpu)
    if cfg.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        cfg.workers = cfg.workers * cfg.num_gpu
        cfg.batch_size = cfg.batch_size * cfg.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', cfg.batch_size)
        cfg.batch_size = cfg.batch_size * cfg.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        cfg.num_iter = int(cfg.num_iter / cfg.num_gpu)
        """
    # -------------------------------------- PM ----------------------------------------- #
    cfg.train_data = "./lmdb/training"
    cfg.valid_data = "./lmdb/validation"
    cfg.Transformation = "TPS"
    cfg.FeatureExtraction = "ResNet"
    cfg.SequenceModeling = "BiLSTM"
    cfg.Prediction = "Attn"
    cfg.character = "0123456789ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖ゙゚゛゜ゝゞゟ゠ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶヷヸヹヺ・ーヽヾヿ･ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝﾞﾟㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ𛀀𛀁"
    # cfg.saved_model = "./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth"
    cfg.num_iter = 10000
    cfg.valInterval = 20
    cfg.FT = True
    # ----------------------------------------------------------------------------------- #

    train(cfg)

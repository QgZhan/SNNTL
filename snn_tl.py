# -*- coding: utf-8 -*-

from __future__ import print_function
import cv2
import CKA

from tqdm import trange

from spiking_model import *
from dataloader import *
from args import get_parser
import torch.nn.functional as F


def weight_init(m):
    """
    用于初始化模型权重
    :param m: 模型
    :return: None
    """
    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.Linear):
        n = m.in_features * m.out_features
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):  # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def train_ddcnet(d_flag, model, source_loader):
    """
    train source and target domain on ddcnet
    :param epoch: current training epoch
    :param model: defined ddcnet
    :param source_loader: source loader
    :param target_loader: target train loader
    :return:
    """
    iter_source = iter(source_loader)
    num_iter = len(source_loader)

    source_correct = 0
    target_correct = 0
    source_total_loss = 0
    target_total_loss = 0
    total_domain_loss = 0

    with trange(num_iter) as t:
        for i in t:
            t.set_description('train')
            source_data, source_label = iter_source.next()
            # 若类别对应取target，则根据source label取同等数量对应类别的target数据（可重复）
            target_data, target_label = target_train_dataset.multi_process_read(source_label.tolist(),
                                                                                num_thread=10, get_type="tensor")
            target_data = torch.stack(target_data)
            target_label = torch.tensor(target_label)

            if cuda:
                source_data, source_label = source_data.cuda(), source_label.cuda()
                target_data, target_label = target_data.cuda(), target_label.cuda()

            # enter training mode
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            source_preds, target_preds, source_feature, target_feature = model(source_data, target_data)

            # CKA计算的是相似度，作为loss的话需要用1减去
            domain_loss = 1 - torch.abs(transfer_criterion(source_feature, target_feature))

            # 将label从(batch_size, 1) 变为 (batch_size, num_classes)即one-hot，便于计算loss
            source_label_ = torch.zeros(len(source_label), args.num_classes).cuda().scatter_(1, source_label.view(-1, 1), 1)
            target_label_ = torch.zeros(len(target_label), args.num_classes).cuda().scatter_(1, target_label.view(-1, 1), 1)
            _, source_predicted = source_preds.max(1)  # 返回的predicated为(batch_size, 1)直接得到值最大的类别的index，便于下面计算正确率
            _, target_predicted = target_preds.max(1)

            source_correct += float(source_predicted.eq(source_label).sum().item())
            target_correct += float(target_predicted.eq(target_label).sum().item())

            source_clf_loss = criterion(source_preds, source_label.squeeze())
            target_clf_loss = criterion(target_preds, target_label.squeeze())

            clf_loss = source_clf_loss + target_clf_loss

            # 在训练source或target达到50%准确率之后再添加迁移损失
            if not d_flag:
                loss = clf_loss
            else:
                loss = clf_loss + args.transfer_ratio * domain_loss

            source_total_loss += source_clf_loss.item()  # 单纯累加所有分类loss的数值
            target_total_loss += target_clf_loss.item()
            total_domain_loss += domain_loss.item()

            loss.backward()
            optimizer.step()

            t.set_postfix(Loss=loss.item(), c_loss=clf_loss.item(), d_loss=domain_loss.item(), use_d_loss=d_flag,
                          S_acc=float(source_correct)/(i+1)/batch_size,
                          T_acc=float(target_correct)/(i+1)/batch_size, )

    source_total_loss /= len(source_loader.dataset)
    target_total_loss /= len(source_loader.dataset)
    total_domain_loss /= len(source_loader)

    source_acc_train = float(source_correct) * 100. / len(source_loader.dataset)
    target_acc_train = float(target_correct) * 100. / len(source_loader.dataset)

    print('{} set: Avg_S_clf_loss: {:.4f}, Avg_T_clf_loss: {:.4f}, S_ACC: {}/{} ({:.2f}%), T_Acc: {}/{} ({:.2f}%)'.format(
        args.SOURCE_NAME, source_total_loss, target_total_loss,
        source_correct, len(source_loader.dataset), source_acc_train,
        target_correct, len(source_loader.dataset), target_acc_train))
    return source_acc_train, target_acc_train, source_total_loss, target_total_loss, total_domain_loss


def test_ddcnet(model, target_loader):
    """
    test target data on fine-tuned alexnet
    :param model: trained alexnet on source data set
    :param target_loader: target dataloader
    :return: correct num
    """
    model.eval()
    test_loss = 0
    correct = 0
    sub_labels = [0 for _ in range(args.num_classes)]
    sub_correct = [0 for _ in range(args.num_classes)]

    iter_target = iter(target_loader)
    num_iter = len(target_loader)

    with trange(num_iter) as t:
        for i in t:
            t.set_description('test')
            test_data, test_label = iter_target.next()

            if cuda:
                test_data, test_label = test_data.cuda(), test_label.cuda()

            test_preds = model(test_data, None)
            _, predicted = test_preds.max(1)

            correct += float(predicted.eq(test_label).sum().item())
            for cls_idx, result in enumerate(predicted):
                sub_labels[test_label[cls_idx]] += 1
                if result == test_label[cls_idx]:
                    sub_correct[test_label[cls_idx]] += 1

            test_label_ = torch.zeros(len(test_label), args.num_classes).cuda().scatter_(1, test_label.view(-1, 1), 1)

            if args.clf_function == "Cross":
                test_loss += criterion(test_preds, test_label.squeeze())
            elif args.clf_function == "MSE":
                test_loss += criterion(test_preds, test_label_)  # 计算分类loss

            t.set_postfix(Loss=test_loss.item()/(i+1)/batch_size, acc=f'{correct}/{(i+1)*batch_size} {correct/(i+1)/batch_size}')

    test_loss /= len(target_loader.dataset)
    print('{} set: Average classification loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        args.TARGET_NAME, test_loss.item(), correct, len(target_loader.dataset),
        100. * correct / len(target_loader.dataset)))
    sub_acc = [sub_correct[idx]/sub_labels[idx] for idx in range(args.num_classes)]

    return 100. * correct / len(target_loader.dataset), test_loss, sub_acc


if __name__ == '__main__':
    # 参数设置
    parser = get_parser()
    args = parser.parse_args()
    project_path = args.project_path
    model_name = args.name
    if not os.path.exists(os.path.join(project_path, 'model', model_name)):
        os.mkdir(os.path.join(project_path, 'model', model_name))

    # GPU设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = torch.cuda.is_available()
    if cuda:  # 使用GPU的情况下，默认多GPU并行
        device_num = torch.cuda.device_count()
        snn = torch.nn.DataParallel(STL(), device_ids=list(range(device_num)))
    else:
        device_num = 1
        snn = STL()
    snn.to(device)
    batch_size = args.base_batch_size * device_num

    # 按类别配对
    source_train_loader = load_all_data(os.path.join(args.project_path, 'transfer_data', args.ROOT_PATH),
                                        args.SOURCE_NAME, batch_size)
    target_test_loader = load_all_data(os.path.join(args.project_path, 'transfer_data', args.ROOT_PATH + '_test_2_3'),
                                       args.TARGET_NAME, batch_size)
    target_train_dataset = ClassMatchDataset(os.path.join(args.project_path, 'transfer_data', args.ROOT_PATH + '_train_1_3'),
                                             args.TARGET_NAME, args.num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    transfer_criterion = CKA.linear_CKA
    optimizer = torch.optim.RMSprop(snn.parameters(), lr=args.learning_rate)

    # 动态学习率调整
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.65)

    best_acc = 0
    best_sub_acc = []
    snn.apply(weight_init)

    model_param = None
    record = np.zeros((args.num_epochs, 8))
    sub_acc_record = np.zeros((args.num_epochs, args.num_classes))

    d_flag = False
    for epoch in range(1, args.num_epochs + 1):
        print('Train Epoch: ', epoch, optimizer.state_dict()['param_groups'][0]['lr'],
              args.ROOT_PATH, f'{args.SOURCE_NAME}->{args.TARGET_NAME}', args.transfer_function)
        source_acc_train, target_acc_train, source_total_loss, target_total_loss, total_domain_loss = train_ddcnet(d_flag,
                                                                                                                   snn,
                                                                                                                   source_train_loader)

        if source_acc_train > 50 or target_acc_train > 50:
            d_flag = True

        with torch.no_grad():
            test_acc, test_loss, sub_acc = test_ddcnet(snn, target_test_loader)
            print('bset test accuracy: ', best_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_sub_acc = sub_acc
                # 仅保存和加载模型参数(推荐使用)
                model_param = snn.state_dict()
        step_lr.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

    # 仅保存和加载模型参数(推荐使用)
    model_save_path = os.path.join(project_path, 'model/', model_name,
                                   'model_{}_{}_{}_{}_{}_{}_{}.pkl'.format(
                                    args.SOURCE_NAME, args.TARGET_NAME,
                                    args.clf_function, args.transfer_function,
                                    str(args.learning_rate), str(args.transfer_ratio), str(best_acc)))
    torch.save(model_param, model_save_path)

    log_save_path = os.path.join(project_path, 'model/', model_name,
                                 'train_loss.log')
    with open(log_save_path, 'a') as f:
        datasets = "SOURCE: " + args.SOURCE_NAME + "    TARGET:" + args.TARGET_NAME + "\n"
        print(datasets)
        f.write(datasets)
        lr = "learning rate: " + str(args.learning_rate) + "\n"
        print(lr)
        f.write(lr)
        timewindows = "time windows: " + str(args.time_window) + "\n"
        print(timewindows)
        f.write(timewindows)
        is_mix = "is mix: " + str(args.is_mix) + "\n"
        print(is_mix)
        f.write(is_mix)
        clf_fuc = "clf loss function: " + args.clf_function + "\n"
        print(clf_fuc)
        f.write(clf_fuc)
        trans_fuc = "transfer loss function: " + args.transfer_function + "\n"
        print(trans_fuc)
        f.write(trans_fuc)
        transfer_ratio = "transfer loss ratio: " + str(args.transfer_ratio) + "\n"
        print(transfer_ratio)
        f.write(transfer_ratio)
        transfer_level = "domain level: " + str(args.transfer_level) + "\n"
        print(transfer_level)
        f.write(transfer_level)
        acc = "best acc: " + str(best_acc) + "\n\n\n\n"
        print(acc)
        f.write(acc)

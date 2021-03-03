import math
import torch
import torch.nn as nn
from args import get_parser

parser = get_parser()
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(args.thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - args.thresh) < args.lens
        return grad_input * temp.float()


act_fun = ActFun.apply


# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * args.decay * (1. - spike) + ops(x)
    spike = act_fun(mem)  # act_fun : approximation firing function
    # x.register_hook(mem_hook_fn)
    return mem, spike


# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
# pool_layer(kernel_size, stride)
cfg_dict = {'big': {'cfg_cnn': [(3, 64, 4, 2, 11),
                               (3, 2),
                               (64, 192, 1, 2, 5),
                               (3, 2),
                               (192, 256, 1, 1, 3),
                               (3, 2)],
                    'cfg_fc': [9216, 2048, 256, args.num_classes]
                    },
            'small': {'cfg_cnn': [(3, 32, 1, 0, 3),
                                  (2, 2),
                                  (32, 64, 1, 0, 3),
                                  (2, 2),
                                  (64, 64, 1, 0, 3),
                                  (2, 2)],
                      'cfg_fc': [256, 64, 512, args.num_classes]
                      },
            'alexnet': {'cfg_cnn': [(3, 64, 4, 2, 11),
                                    (3, 2),
                                    (64, 192, 1, 2, 5),
                                    (3, 2),
                                    (192, 384, 1, 1, 3),
                                    (384, 256, 1, 1, 3),
                                    (256, 256, 1, 1, 3),
                                    (3, 2)],
                        'cfg_fc': [256 * 6 * 6, 4096, 4096, args.num_classes]
                        }
            }
cfg_cnn = cfg_dict[args.model_structs]['cfg_cnn']
cfg_fc = cfg_dict[args.model_structs]['cfg_fc']


class STL(nn.Module):
    def __init__(self):
        super(STL, self).__init__()
        # 定义卷积层和池化层
        self.conv_list = nn.ModuleList()
        for cfg in cfg_cnn:
            if len(cfg) == 5:
                self.conv_list.append(nn.Conv2d(cfg[0], cfg[1], stride=cfg[2], padding=cfg[3], kernel_size=cfg[4]))
            elif len(cfg) == 2:
                self.conv_list.append(nn.MaxPool2d(kernel_size=cfg[0], stride=cfg[1]))

        # 定义全连接层
        self.fc_list = nn.ModuleList()
        for i in range(len(cfg_fc) - 1):
            self.fc_list.append(nn.Linear(cfg_fc[i], cfg_fc[i+1]))

    def forward(self, source, target):
        # =================================================================================================
        # 初始化mem、spike 和 sumspike，
        # mem变量用于记录神经元膜电位，spike记录神经元放电状态， sumspike记录所有时间长度神经元放电累加得到的频率
        # 这三种变量与每一层的输出shape相同，因此初始化变量时需要计算每层的输出shape
        # =================================================================================================
        size = source.shape  # 记录输入的shape
        fm_size = size[2]    # 特征图尺寸，初始化为输入的宽/高，由于输入的宽高相同，因此特征图尺寸只记录一个

        conv_source_mem_list, conv_source_spike_list, conv_source_sumspike_list = [], [], []
        fc_source_mem_list, fc_source_spike_list, fc_source_sumspike_list = [], [], []
        conv_target_mem_list, conv_target_spike_list, conv_target_sumspike_list = [], [], []
        fc_target_mem_list, fc_target_spike_list, fc_target_sumspike_list = [], [], []

        for cfg in cfg_cnn:
            if len(cfg) == 5:
                # cfg有五个参数时代表卷积层
                fm_size = math.floor((fm_size - cfg[4] + 2 * cfg[3]) / cfg[2]) + 1   # 根据参数计算卷积后的特征图大小
                tmp = torch.zeros(size[0], cfg[1], fm_size, fm_size, device=device)  # 创建与特征图相同shape的tensor
                # 将tensor分别添加至列表中，注意，必须用.clone()函数进行tensor的深拷贝
                conv_source_mem_list.append(tmp.clone())
                conv_source_spike_list.append(tmp.clone())
                conv_target_mem_list.append(tmp.clone())
                conv_target_spike_list.append(tmp.clone())
                conv_source_sumspike_list.append(tmp.clone())
                conv_target_sumspike_list.append(tmp.clone())
            elif len(cfg) == 2:
                # cfg有两个参数时代表池化层
                fm_size = math.floor((fm_size - cfg[0]) / cfg[1]) + 1  # 根据参数计算池化后的特征图大小
                # 池化操作不需要记录神经元所有时刻的电位及点火状态，为保持层计数一致，用None填充
                conv_source_mem_list.append(None)
                conv_source_spike_list.append(None)
                conv_target_mem_list.append(None)
                conv_target_spike_list.append(None)
                conv_source_sumspike_list.append(None)
                conv_target_sumspike_list.append(None)

        for cfg in cfg_fc[1:]:
            tmp = torch.zeros(size[0], cfg, device=device)   # 根据参数计算全连接后的特征图大小
            fc_source_mem_list.append(tmp.clone())
            fc_source_spike_list.append(tmp.clone())
            fc_source_sumspike_list.append(tmp.clone())
            fc_target_mem_list.append(tmp.clone())
            fc_target_spike_list.append(tmp.clone())
            fc_target_sumspike_list.append(tmp.clone())

        for step in range(args.time_window):  # simulation time steps
            """
            source feature
            """
            source_x = source > torch.rand(source.size(), device=device)  # prob. firing
            source_x = source_x.float()

            for i in range(len(self.conv_list)):
                if len(cfg_cnn[i]) == 5:
                    # 卷积
                    # print(f"conv {i}", conv_source_mem_list[i].max(), conv_source_mem_list[i].min())
                    conv_source_mem_list[i], conv_source_spike_list[i] = mem_update(self.conv_list[i], source_x,
                                                                                    conv_source_mem_list[i],
                                                                                    conv_source_spike_list[i])
                    conv_source_sumspike_list[i] += conv_source_spike_list[i]
                    source_x = conv_source_spike_list[i]

                elif len(cfg_cnn[i]) == 2:
                    # 池化
                    source_x = self.conv_list[i](source_x)

            # 特征图展平
            source_x = source_x.view(source_x.size(0), 256 * 6 * 6)

            for i in range(len(self.fc_list)):
                source_x = nn.Dropout(0.5)(source_x)
                # 全连接
                fc_source_mem_list[i], fc_source_spike_list[i] = mem_update(self.fc_list[i], source_x,
                                                                            fc_source_mem_list[i],
                                                                            fc_source_spike_list[i])

                fc_source_sumspike_list[i] += fc_source_spike_list[i]
                source_x = fc_source_spike_list[i]

            """
            target feature
            """
            if target is not None:
                target_x = target > torch.rand(source.size(), device=device)  # prob. firing
                target_x = target_x.float()

                for i in range(len(self.conv_list)):
                    if len(cfg_cnn[i]) == 5:
                        # 卷积
                        conv_target_mem_list[i], conv_target_spike_list[i] = mem_update(self.conv_list[i], target_x,
                                                                                        conv_target_mem_list[i],
                                                                                        conv_target_spike_list[i])
                        target_x = conv_target_spike_list[i]
                        conv_target_sumspike_list[i] += conv_target_spike_list[i]

                    elif len(cfg_cnn[i]) == 2:
                        # 池化
                        target_x = self.conv_list[i](target_x)

                # 特征图展平
                target_x = target_x.view(target_x.size(0), 256 * 6 * 6)

                for i in range(len(self.fc_list)):
                    target_x = nn.Dropout(0.5)(target_x)
                    # 全连接
                    fc_target_mem_list[i], fc_target_spike_list[i] = mem_update(self.fc_list[i], target_x,
                                                                                fc_target_mem_list[i],
                                                                                fc_target_spike_list[i])
                    fc_target_sumspike_list[i] += fc_target_spike_list[i]
                    target_x = fc_target_spike_list[i]

        # 计算最终分类结果
        source_result = fc_source_sumspike_list[-1] / args.time_window
        target_result = fc_target_sumspike_list[-1] / args.time_window

        # 若target存在，则返回倒数第transfer_level层的特征，用于计算迁移损失
        if target is not None:
            L = args.transfer_level
            if L < len(fc_source_sumspike_list) - 1:
                return source_result, target_result, fc_source_sumspike_list[-L - 1], fc_target_sumspike_list[-L - 1]
            else:
                L -= len(fc_source_sumspike_list) - 1
                return source_result, target_result, \
                       conv_source_sumspike_list[-L * 2].view(size[0], -1), \
                       conv_target_sumspike_list[-L * 2].view(size[0], -1)
        return source_result

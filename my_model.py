import torch
import torch.nn as nn
from torchvision import models
import pdb
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding


def get_my_model(model_name):


    def get_padded_operation(old_operation):
        # pad weight
        m = nn.ZeroPad2d((0, 1, 0, 1, 0, 0, 0, 0))
        new_weight = torch.nn.Parameter(m(old_operation.weight))

        # define new kernel size
        old_kernel_size = old_operation.kernel_size
        new_kernel_size = (old_kernel_size[0]+1, old_kernel_size[1]+1)

        # define new operation
        new_operation = nn.Conv2d(
            old_operation.in_channels,
            old_operation.out_channels,
            kernel_size=new_kernel_size,
            stride=old_operation.stride,
            padding=old_operation.padding,
            bias=False
        )

        # replace weight
        new_operation.weight = new_weight

        return new_operation


    def get_sliced_operation(old_operation):

        # define new kernel size
        old_kernel_size = old_operation.kernel_size
        new_kernel_size = (old_kernel_size[0]-1, old_kernel_size[1]-1)

        old_padding_size = old_operation.padding
        new_padding_size = (old_padding_size[0]-1, old_padding_size[1]-1)

        new_weight = torch.nn.Parameter(old_operation.weight[:, :, :new_kernel_size[0], :new_kernel_size[1]])

        # define new operation
        new_operation = nn.Conv2d(
            old_operation.in_channels,
            old_operation.out_channels,
            kernel_size=new_kernel_size,
            stride=old_operation.stride,
            padding=new_padding_size,
            bias=False
        )

        # replace weight
        new_operation.weight = new_weight

        return new_operation


    # 나름 로직이 정상인거 체크 했음
    def get_padded_operation_groups(old_operation):
        # pad weight
        m = nn.ZeroPad2d((0, 1, 0, 1, 0, 0, 0, 0))
        new_weight = torch.nn.Parameter(m(old_operation.weight))

        # define new kernel size
        old_kernel_size = old_operation.kernel_size
        new_kernel_size = (old_kernel_size[0]+1, old_kernel_size[1]+1)

        # define new operation
        new_operation = nn.Conv2d(
            old_operation.in_channels,
            old_operation.out_channels,
            kernel_size=new_kernel_size,
            stride=old_operation.stride,
            padding=old_operation.padding,
            groups=old_operation.groups,
            bias=False
        )

        # replace weight
        new_operation.weight = new_weight

        return new_operation


    def get_padded_operation_convstatic(old_operation):
        # pad weight
        m = nn.ZeroPad2d((0, 1, 0, 1, 0, 0, 0, 0))
        new_weight = torch.nn.Parameter(m(old_operation.weight))

        # define new kernel size
        old_kernel_size = old_operation.kernel_size
        new_kernel_size = (old_kernel_size[0]+1, old_kernel_size[1]+1)

        # define new operation
        # new_operation = nn.Conv2d(
        new_operation = Conv2dStaticSamePadding(
            old_operation.in_channels,
            old_operation.out_channels,
            kernel_size=new_kernel_size,
            stride=old_operation.stride,
            groups=old_operation.groups,
            bias=False,
            image_size=224,
        )

        # replace weight
        new_operation.weight = new_weight

        return new_operation


    def get_padded_operation_f(old_operation):

        # define new kernel size
        old_kernel_size = old_operation.kernel_size
        new_kernel_size = (old_kernel_size[0]+1, old_kernel_size[1]+1)

        # define new operation
        new_operation = nn.Conv2d(
            old_operation.in_channels,
            old_operation.out_channels,
            kernel_size=new_kernel_size,
            stride=old_operation.stride,
            padding=old_operation.padding,
            bias=False
        )

        return new_operation


    def get_padded_operation_groups_f(old_operation):

        # define new kernel size
        old_kernel_size = old_operation.kernel_size
        new_kernel_size = (old_kernel_size[0]+1, old_kernel_size[1]+1)

        # define new operation
        new_operation = nn.Conv2d(
            old_operation.in_channels,
            old_operation.out_channels,
            kernel_size=new_kernel_size,
            stride=old_operation.stride,
            padding=old_operation.padding,
            groups=old_operation.groups,
            bias=False
        )

        return new_operation


    def get_sliced_operation_f(old_operation):

        # define new kernel size
        old_kernel_size = old_operation.kernel_size
        new_kernel_size = (old_kernel_size[0]-1, old_kernel_size[1]-1)

        old_padding_size = old_operation.padding
        new_padding_size = (old_padding_size[0]-1, old_padding_size[1]-1)

        # define new operation
        new_operation = nn.Conv2d(
            old_operation.in_channels,
            old_operation.out_channels,
            kernel_size=new_kernel_size,
            stride=old_operation.stride,
            padding=new_padding_size,
            bias=False
        )

        return new_operation


    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=True)
        model_type = "res"

    if model_name == "my_resnet18_pad":
        model_ft = models.resnet18(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_padded_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv1 = get_padded_operation(model_ft.layer2[0].conv1)
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv1 = get_padded_operation(model_ft.layer3[0].conv1)
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv1 = get_padded_operation(model_ft.layer4[0].conv1)
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "my_resnet18_slice":
        model_ft = models.resnet18(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_sliced_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # padding -1
        model_ft.layer2[0].conv1 = get_sliced_operation(model_ft.layer2[0].conv1)
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv1 = get_sliced_operation(model_ft.layer3[0].conv1)
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv1 = get_sliced_operation(model_ft.layer4[0].conv1)
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "my_resnet18_mp":
        model_ft = models.resnet18(pretrained=True)
        model_type = "res"

        model_ft.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # padding -1

    if model_name == "resnet18_f":
        model_ft = models.resnet18(pretrained=False)
        model_type = "res"

    if model_name == "my_resnet18_f":
        model_ft = models.resnet18(pretrained=False)
        model_type = "res"

        # model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3), bias=False)
        # model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        # model_ft.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # model_ft.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
        # model_ft.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # model_ft.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(2, 2), stride=(2, 2), bias=False)
        # model_ft.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # model_ft.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(2, 2), stride=(2, 2), bias=False)

        model_ft.conv1 = get_padded_operation_f(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv1 = get_padded_operation_f(model_ft.layer2[0].conv1)
        model_ft.layer2[0].downsample[0] = get_padded_operation_f(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv1 = get_padded_operation_f(model_ft.layer3[0].conv1)
        model_ft.layer3[0].downsample[0] = get_padded_operation_f(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv1 = get_padded_operation_f(model_ft.layer4[0].conv1)
        model_ft.layer4[0].downsample[0] = get_padded_operation_f(model_ft.layer4[0].downsample[0])


    ######################################


    if model_name == "resnet34":
        model_ft = models.resnet34(pretrained=True)
        model_type = "res"

    if model_name == "my_resnet34_pad":
        model_ft = models.resnet34(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_padded_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv1 = get_padded_operation(model_ft.layer2[0].conv1)
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv1 = get_padded_operation(model_ft.layer3[0].conv1)
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv1 = get_padded_operation(model_ft.layer4[0].conv1)
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "resnet34_f":
        model_ft = models.resnet34(pretrained=False)
        model_type = "res"

    if model_name == "my_resnet34_f":
        model_ft = models.resnet34(pretrained=False)
        model_type = "res"

        model_ft.conv1 = get_padded_operation_f(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv1 = get_padded_operation_f(model_ft.layer2[0].conv1)
        model_ft.layer2[0].downsample[0] = get_padded_operation_f(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv1 = get_padded_operation_f(model_ft.layer3[0].conv1)
        model_ft.layer3[0].downsample[0] = get_padded_operation_f(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv1 = get_padded_operation_f(model_ft.layer4[0].conv1)
        model_ft.layer4[0].downsample[0] = get_padded_operation_f(model_ft.layer4[0].downsample[0])


    ######################################


    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=True)
        model_type = "res"

    if model_name == "my_resnet50_pad":
        model_ft = models.resnet50(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_padded_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation(model_ft.layer2[0].conv2)
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation(model_ft.layer3[0].conv2)
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation(model_ft.layer4[0].conv2)
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "my_resnet50_slice":
        model_ft = models.resnet50(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_sliced_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # padding -1
        model_ft.layer2[0].conv2 = get_sliced_operation(model_ft.layer2[0].conv2)
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_sliced_operation(model_ft.layer3[0].conv2)
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_sliced_operation(model_ft.layer4[0].conv2)
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "resnet50_f":
        model_ft = models.resnet50(pretrained=False)
        model_type = "res"

    if model_name == "my_resnet50_f":
        model_ft = models.resnet50(pretrained=False)
        model_type = "res"

        model_ft.conv1 = get_padded_operation_f(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation_f(model_ft.layer2[0].conv2)
        model_ft.layer2[0].downsample[0] = get_padded_operation_f(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation_f(model_ft.layer3[0].conv2)
        model_ft.layer3[0].downsample[0] = get_padded_operation_f(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation_f(model_ft.layer4[0].conv2)
        model_ft.layer4[0].downsample[0] = get_padded_operation_f(model_ft.layer4[0].downsample[0])


    ######################################


    if model_name == "resnet101":
        model_ft = models.resnet101(pretrained=True)
        model_type = "res"

    if model_name == "my_resnet101_pad":
        model_ft = models.resnet101(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_padded_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation(model_ft.layer2[0].conv2)
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation(model_ft.layer3[0].conv2)
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation(model_ft.layer4[0].conv2)
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "my_resnet101_slice":
        model_ft = models.resnet101(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_sliced_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # padding -1
        model_ft.layer2[0].conv2 = get_sliced_operation(model_ft.layer2[0].conv2)
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_sliced_operation(model_ft.layer3[0].conv2)
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_sliced_operation(model_ft.layer4[0].conv2)
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "resnet101_f":
        model_ft = models.resnet101(pretrained=False)
        model_type = "res"

    if model_name == "my_resnet101_f":
        model_ft = models.resnet101(pretrained=False)
        model_type = "res"

        model_ft.conv1 = get_padded_operation_f(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation_f(model_ft.layer2[0].conv2)
        model_ft.layer2[0].downsample[0] = get_padded_operation_f(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation_f(model_ft.layer3[0].conv2)
        model_ft.layer3[0].downsample[0] = get_padded_operation_f(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation_f(model_ft.layer4[0].conv2)
        model_ft.layer4[0].downsample[0] = get_padded_operation_f(model_ft.layer4[0].downsample[0])


    ######################################


    if model_name == "resnet152":
        model_ft = models.resnet152(pretrained=True)
        model_type = "res"

    if model_name == "my_resnet152_pad":
        model_ft = models.resnet152(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_padded_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "my_resnet152_slice":
        model_ft = models.resnet152(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_sliced_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # padding -1
        model_ft.layer2[0].conv2 = get_sliced_operation(model_ft.layer2[0].conv2)
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_sliced_operation(model_ft.layer3[0].conv2)
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_sliced_operation(model_ft.layer4[0].conv2)
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "my_resnet152_mp":
        model_ft = models.resnet152(pretrained=True)
        model_type = "res"

        model_ft.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # padding -1

    if model_name == "resnet152_f":
        model_ft = models.resnet152(pretrained=False)
        model_type = "res"

    if model_name == "my_resnet152_f":
        model_ft = models.resnet152(pretrained=False)
        model_type = "res"

        # model_ft.conv1 = nn.Conv2d(3, 64, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3), bias=False)
        # model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        # model_ft.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # model_ft.layer2[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(2, 2), stride=(2, 2), bias=False)
        # model_ft.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # model_ft.layer3[0].downsample[0] = nn.Conv2d(512, 1024, kernel_size=(2, 2), stride=(2, 2), bias=False)
        # model_ft.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # model_ft.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(2, 2), stride=(2, 2), bias=False)

        model_ft.conv1 = get_padded_operation_f(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation_f(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation_f(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation_f(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation_f(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation_f(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation_f(model_ft.layer4[0].downsample[0])


    ######################################


    if model_name == "wide_resnet50_2":
        model_ft = models.wide_resnet50_2(pretrained=True)
        model_type = "res"

    if model_name == "my_wide_resnet50_2_pad":
        model_ft = models.wide_resnet50_2(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_padded_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])


    if model_name == "wide_resnet50_2_f":
        model_ft = models.wide_resnet50_2(pretrained=False)
        model_type = "res"

    if model_name == "my_wide_resnet50_2_f":
        model_ft = models.wide_resnet50_2(pretrained=False)
        model_type = "res"

        model_ft.conv1 = get_padded_operation_f(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation_f(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation_f(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation_f(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation_f(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation_f(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation_f(model_ft.layer4[0].downsample[0])


    ######################################


    if model_name == "wide_resnet101_2":
        model_ft = models.wide_resnet101_2(pretrained=True)
        model_type = "res"

    if model_name == "my_wide_resnet101_2_pad":
        model_ft = models.wide_resnet101_2(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_padded_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "wide_resnet101_2_f":
        model_ft = models.wide_resnet101_2(pretrained=False)
        model_type = "res"

    if model_name == "my_wide_resnet101_2_f":
        model_ft = models.wide_resnet101_2(pretrained=False)
        model_type = "res"

        model_ft.conv1 = get_padded_operation_f(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation_f(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation_f(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation_f(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation_f(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation_f(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation_f(model_ft.layer4[0].downsample[0])


    ######################################


    if model_name == "resnext50_32x4d":
        model_ft = models.resnext50_32x4d(pretrained=True)
        model_type = "res"

    if model_name == "my_resnext50_32x4d_pad":
        model_ft = models.resnext50_32x4d(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_padded_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation_groups(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation_groups(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation_groups(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "resnext50_32x4d_f":
        model_ft = models.resnext50_32x4d(pretrained=False)
        model_type = "res"

    if model_name == "my_resnext50_32x4d_f":
        model_ft = models.resnext50_32x4d(pretrained=False)
        model_type = "res"

        model_ft.conv1 = get_padded_operation_f(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation_groups_f(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation_f(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation_groups_f(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation_f(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation_groups_f(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation_f(model_ft.layer4[0].downsample[0])


    ######################################


    if model_name == "resnext101_32x8d":
        model_ft = models.resnext101_32x8d(pretrained=True)
        model_type = "res"

    if model_name == "my_resnext101_32x8d_pad":
        model_ft = models.resnext101_32x8d(pretrained=True)
        model_type = "res"

        model_ft.conv1 = get_padded_operation(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation_groups(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation_groups(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation_groups(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation(model_ft.layer4[0].downsample[0])

    if model_name == "resnext101_32x8d_f":
        model_ft = models.resnext101_32x8d(pretrained=False)
        model_type = "res"

    if model_name == "my_resnext101_32x8d_f":
        model_ft = models.resnext101_32x8d(pretrained=False)
        model_type = "res"

        model_ft.conv1 = get_padded_operation_f(model_ft.conv1)
        model_ft.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        model_ft.layer2[0].conv2 = get_padded_operation_groups_f(model_ft.layer2[0].conv2)  # conv1 아님.
        model_ft.layer2[0].downsample[0] = get_padded_operation_f(model_ft.layer2[0].downsample[0])
        model_ft.layer3[0].conv2 = get_padded_operation_groups_f(model_ft.layer3[0].conv2)  # conv1 아님.
        model_ft.layer3[0].downsample[0] = get_padded_operation_f(model_ft.layer3[0].downsample[0])
        model_ft.layer4[0].conv2 = get_padded_operation_groups_f(model_ft.layer4[0].conv2)  # conv1 아님.
        model_ft.layer4[0].downsample[0] = get_padded_operation_f(model_ft.layer4[0].downsample[0])


    ######################################


    if model_name == "eff_b0":
        model_ft = EfficientNet.from_pretrained('efficientnet-b0')
        model_type = "eff"

    if model_name == "my_eff_b0_pad":
        model_ft = EfficientNet.from_pretrained('efficientnet-b0')
        model_type = "eff"

        model_ft._conv_stem = get_padded_operation_convstatic(model_ft._conv_stem)
        model_ft._blocks[1]._depthwise_conv = get_padded_operation_convstatic(model_ft._blocks[1]._depthwise_conv)
        model_ft._blocks[3]._depthwise_conv = get_padded_operation_convstatic(model_ft._blocks[3]._depthwise_conv)
        model_ft._blocks[5]._depthwise_conv = get_padded_operation_convstatic(model_ft._blocks[5]._depthwise_conv)
        model_ft._blocks[11]._depthwise_conv = get_padded_operation_convstatic(model_ft._blocks[11]._depthwise_conv)


    ######################################


    if model_name == "eff_b7":
        model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        model_type = "eff"


    ######################################


    if model_name == "densenet121":
        model_ft = models.densenet121(pretrained=True)
        model_type = "den"

    if model_name == "my_densenet121_pad":
        model_ft = models.densenet121(pretrained=True)
        model_type = "den"

        model_ft.features.conv0 = get_padded_operation(model_ft.features.conv0)
        model_ft.features.pool0 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)

    if model_name == "densenet121_f":
        model_ft = models.densenet121(pretrained=False)
        model_type = "den"

    if model_name == "my_densenet121_f":
        model_ft = models.densenet121(pretrained=False)
        model_type = "den"

        model_ft.features.conv0 = get_padded_operation_f(model_ft.features.conv0)
        model_ft.features.pool0 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)

    if model_name == "my_densenet121_sliced_f":
        model_ft = models.densenet121(pretrained=False)
        model_type = "den"

        model_ft.features.conv0 = get_sliced_operation_f(model_ft.features.conv0)
        model_ft.features.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # padding -1


    ######################################


    if model_name == "densenet201_f":
        model_ft = models.densenet201(pretrained=False)
        model_type = "den"

    if model_name == "my_densenet201_f":
        model_ft = models.densenet201(pretrained=False)
        model_type = "den"

        model_ft.features.conv0 = get_padded_operation_f(model_ft.features.conv0)
        model_ft.features.pool0 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)


    ######################################


    if model_name == "vgg16":
        model_ft = models.vgg16(pretrained=True)
        model_type = "vgg"

    if model_name == "vgg16_f":
        model_ft = models.vgg16(pretrained=False)
        model_type = "vgg"


    ######################################


    if model_name == "mnasnet1_0":
        model_ft = models.mnasnet1_0(pretrained=True)
        model_type = "mnas"

    return model_ft, model_type

import time
import torch
import torch.utils.data
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from model import Vgg16, Res18, Res50, WRN
from torchvision import transforms, datasets


# 测试模型
def test(model, device, test_loader):
    model.eval()  # 不启用Batch Normalization 和 Dropout
    total = 0
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss = test_loss + loss.item()
            _, predicted = outputs.max(1)
            total = total + targets.size(0)
            correct = correct + predicted.eq(targets).sum().item()

    test_accuracy = correct / total
    return test_accuracy, test_loss


# 训练模型
def train(model, model_name, weight_decay, device, optimizer, scheduler, train_loader, test_loader, max_epoches, checkpoint_epoch):
    start_time = time.time()  # 统计训练时间

    result = pd.DataFrame()  # 记录训练过程数据并存储到csv文件
    train_accuracy_record = [0.0] * max_epoches  # 记录每个epoch训练集的准确率
    train_loss_record = [0.0] * max_epoches  # 记录每个epoch训练集的损失值
    test_accuracy_record = [0.0] * max_epoches  # 记录每个epoch测试集的准确率
    test_loss_record = [0.0] * max_epoches  # 记录每个epoch测试集的损失值

    for epoch in range(0, max_epoches):
        model.train()  # 启用batch normalization和drop out
        total = 0  # 记录样本数
        correct = 0  # 记录总正确数
        train_loss = 0.0  # 记录总损失值

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据加载到gpu
            optimizer.zero_grad()  # 计算梯度
            outputs = model(inputs)  # 前向传播
            loss_ce = F.cross_entropy(outputs, targets)  # 交叉熵损失
            loss_re = 0.0  # 正则化损失
            for name, par in model.named_parameters():
                loss_re = loss_re + weight_decay * 0.5 * torch.sum(torch.pow(par, 2))
            loss = loss_ce + loss_re
            loss.backward()  # 后向传播
            optimizer.step()  # 更新优化器

            # 可视化训练过程
            train_loss = train_loss + loss.item()  # 计算当前损失值
            _, predicted = outputs.max(1)
            total = total + targets.size(0)
            correct = correct + predicted.eq(targets).sum().item()  # 计算当前准确率

        # 展示训练信息
        train_accuracy_record[epoch] = correct / total
        train_loss_record[epoch] = train_loss
        print('epoch: ' + str(epoch + 1) + '  train_loss: ' + str(train_loss_record[epoch]) + ';  train_accuracy: ' + str(train_accuracy_record[epoch] * 100) + '%')
        test_accuracy_record[epoch], test_loss_record[epoch] = test(model, device, test_loader)
        print('epoch: ' + str(epoch + 1) + '  test_loss: ' + str(test_loss_record[epoch]) + ';  test_accuracy: ' + str(test_accuracy_record[epoch] * 100) + '%')

        scheduler.step()  # 余弦退火调整学习率

        if epoch + 1 == checkpoint_epoch:
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, 'model_' + model_name + '_original_parameter_epoch' + str(checkpoint_epoch) + '_ckpt.pth')

    time_now = time.time() - start_time
    print('Finished Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_now // 60, time_now % 60))

    # 将训练过程数据保存到csv文件
    result['Train_Accuracy'] = train_accuracy_record
    result['Train_Loss'] = train_loss_record
    result['Test_Accuracy'] = test_accuracy_record
    result['Test_Loss'] = test_loss_record
    result.to_csv('model_' + model_name + '_train_info' + '.csv')

    torch.save(model.state_dict(), 'model_' + model_name + '_original_parameters.pth')  # 保存模型参数


def get_dataloader(data_name):
    print('...Preparing data...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if data_name == 'cifar10':
        cifar10_train_size, cifar10_test_size, cifar10_classes, cifar10_input_size = 50000, 10000, 10, (3, 32, 32)
        cifar10_train_dataset = datasets.CIFAR10('./cifar10_data', train=True, transform=transform_train, download=False)
        cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=128, shuffle=False)
        cifar10_test_dataset = datasets.CIFAR10('./cifar10_data', train=False, transform=transform_test, download=False)
        cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=128, shuffle=False)
        return cifar10_train_loader, cifar10_test_loader, cifar10_train_size, cifar10_test_size, cifar10_classes, cifar10_input_size


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader, train_size, test_size, num_classes, input_size = get_dataloader('cifar10')  # 构建训练集、测试集
    model_original = Vgg16(num_classes)
    model_original = model_original.to(device)
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight', 
                   'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                   'fc1.weight', 'fc2.weight', 'fc3.weight']
    """


                   
    model_original = Res18(num_classes)
    model_original = model_original.to(device)
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight',
                   'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight',
                   'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight',
                   'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight'] 
                                    
    model_original = Res50(num_classes=10)
    model_original = model_original.to(device)
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight', 'conv8.weight',  'conv9.weight', 'conv10.weight',
                   'conv11.weight', 'conv12.weight', 'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight', 'conv18.weight', 'conv19.weight', 'conv20.weight',
                   'conv21.weight', 'conv22.weight', 'conv23.weight', 'conv24.weight', 'conv25.weight', 'conv26.weight', 'conv27.weight', 'conv28.weight', 'conv29.weight', 'conv30.weight',
                   'conv31.weight', 'conv32.weight', 'conv33.weight', 'conv34.weight', 'conv35.weight', 'conv36.weight', 'conv37.weight', 'conv38.weight', 'conv39.weight', 'conv40.weight',
                   'conv41.weight', 'conv42.weight', 'conv43.weight', 'conv44.weight', 'conv45.weight', 'conv46.weight', 'conv47.weight', 'conv48.weight', 'conv49.weight',
                   'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight', 'shortcut4.weight']

    model_original = WRN(num_classes)
    model_original = model_original.to(device)
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                   'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                   'fc.weight', 'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight']
    """

    if device == 'cuda':
        cudnn.deterministic = True
        cudnn.benchmark = True  # 不改变给定的神经网络结构的情况下，大大提升其训练和预测的速度
    optimizer = optim.SGD(model_original.parameters(), lr=0.1, momentum=0.9, weight_decay=0)  # 创建优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # 动态学习率

    train(model_original, 'Vgg16', 0.001, device, optimizer, scheduler, train_loader, test_loader, 200, 150)  # 训练模型

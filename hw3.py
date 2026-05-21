#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CycleGAN 单文件实现：照片 ↔ 梵高风格迁移
包含自动数据集抽样功能，支持训练和推理。
"""

import argparse
import itertools
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# ================== 工具函数 ==================
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Buffer is empty"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def reduce_images(folder, max_n):
    """随机保留 max_n 张图片，删除其余图片"""
    if not os.path.isdir(folder):
        return
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(files) > max_n:
        keep = set(random.sample(files, max_n))
        for f in files:
            if f not in keep:
                os.remove(os.path.join(folder, f))
        print(f"文件夹 {folder} ：已从 {len(files)} 张图片缩减至 {max_n} 张。")
    else:
        print(f"文件夹 {folder} ：图片数量 {len(files)} ≤ {max_n}，无需缩减。")

# ================== 模型定义 ==================
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)

# ================== 数据集 ==================
class ImageDataset(Dataset):
    def __init__(self, root, unaligned=False, mode='train'):
        self.unaligned = unaligned
        self.files_A = sorted(os.listdir(os.path.join(root, f'{mode}A')))
        self.files_B = sorted(os.listdir(os.path.join(root, f'{mode}B')))
        self.root = root
        self.mode = mode

        self.transform = transforms.Compose([
            transforms.Resize(int(286), Image.BICUBIC),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        A_path = os.path.join(self.root, f'{self.mode}A', self.files_A[index % len(self.files_A)])
        if self.unaligned:
            B_path = os.path.join(self.root, f'{self.mode}B',
                                  self.files_B[torch.randint(0, len(self.files_B), (1,)).item()])
        else:
            B_path = os.path.join(self.root, f'{self.mode}B', self.files_B[index % len(self.files_B)])

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        return {'A': self.transform(A_img), 'B': self.transform(B_img)}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

# ================== 训练函数 ==================
def train(args):
    # 自动抽样：缩减训练集图片数量
    print("正在检查并缩减数据集到指定大小...")
    reduce_images(os.path.join(args.data_root, 'trainA'), max_n=args.max_samples)
    reduce_images(os.path.join(args.data_root, 'trainB'), max_n=args.max_samples)
    print("数据集准备完毕。")

    os.makedirs(args.save_root, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 初始化网络
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # 损失函数
    criterion_GAN = nn.MSELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)

    # 优化器
    optimizer_G = Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                       lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_A = Adam(netD_A.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_B = Adam(netD_B.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # 学习率调度
    def lr_lambda(epoch):
        if epoch < args.decay_epoch:
            return 1.0
        return 1.0 - max(0, epoch - args.decay_epoch) / (args.n_epochs - args.decay_epoch)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    dataset = ImageDataset(args.data_root, unaligned=True, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for epoch in range(args.epoch, args.n_epochs):
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.n_epochs}')
        for batch in pbar:
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # 生成器训练
            set_requires_grad([netD_A, netD_B], False)
            optimizer_G.zero_grad()

            loss_id_A = criterion_identity(netG_B2A(real_A), real_A) * 5.0
            loss_id_B = criterion_identity(netG_A2B(real_B), real_B) * 5.0

            fake_B = netG_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(netD_B(fake_B), torch.ones_like(netD_B(fake_B)))

            fake_A = netG_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(netD_A(fake_A), torch.ones_like(netD_A(fake_A)))

            recov_A = netG_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A) * 10.0

            recov_B = netG_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B) * 10.0

            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()
            optimizer_G.step()

            # 判别器 A 训练
            set_requires_grad([netD_A], True)
            optimizer_D_A.zero_grad()

            loss_D_real = criterion_GAN(netD_A(real_A), torch.ones_like(netD_A(real_A)))
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_D_fake = criterion_GAN(netD_A(fake_A_.detach()), torch.zeros_like(netD_A(fake_A_)))
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # 判别器 B 训练
            set_requires_grad([netD_B], True)
            optimizer_D_B.zero_grad()

            loss_D_real = criterion_GAN(netD_B(real_B), torch.ones_like(netD_B(real_B)))
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_D_fake = criterion_GAN(netD_B(fake_B_.detach()), torch.zeros_like(netD_B(fake_B_)))
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            pbar.set_postfix(G=loss_G.item(), D_A=loss_D_A.item(), D_B=loss_D_B.item())

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        if (epoch+1) % 10 == 0 or epoch+1 == args.n_epochs:
            torch.save(netG_A2B.state_dict(), f"{args.save_root}/netG_A2B_epoch_{epoch+1}.pth")
            torch.save(netG_B2A.state_dict(), f"{args.save_root}/netG_B2A_epoch_{epoch+1}.pth")
            print(f"模型已保存至 epoch {epoch+1}")

    print("训练完成。")

# ================== 推理函数 ==================
def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG_A2B = Generator().to(device)
    netG_A2B.load_state_dict(torch.load(args.model_path, map_location=device))
    netG_A2B.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = Image.open(args.input_image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake = netG_A2B(img_tensor)

    fake = fake.squeeze(0).cpu() * 0.5 + 0.5
    fake_img = transforms.ToPILImage()(fake)
    fake_img.save(args.output_image)
    print(f"风格化图像已保存至 {args.output_image}")

# ================== 主入口 ==================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CycleGAN 单文件：训练 / 推理')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='运行模式')
    # 训练参数
    parser.add_argument('--data_root', type=str, default='data/vangogh2photo', help='数据集目录')
    parser.add_argument('--epoch', type=int, default=0, help='起始 epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='总 epoch 数')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--b1', type=float, default=0.5)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--decay_epoch', type=int, default=25)
    parser.add_argument('--save_root', type=str, default='checkpoints')
    parser.add_argument('--max_samples', type=int, default=100, help='trainA/trainB 中最多保留的图片数（自动抽样）')
    # 推理参数
    parser.add_argument('--input_image', type=str, help='输入图片路径（test 模式）')
    parser.add_argument('--output_image', type=str, default='output.jpg', help='输出图片路径（test 模式）')
    parser.add_argument('--model_path', type=str, help='生成器权重路径（test 模式）')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        if not args.input_image or not args.model_path:
            parser.error("test 模式需要 --input_image 和 --model_path 参数")
        test(args)
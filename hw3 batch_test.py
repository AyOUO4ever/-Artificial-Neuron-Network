import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from hw3 import Generator

def batch_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG_A2B = Generator().to(device)
    netG_A2B.load_state_dict(torch.load('checkpoints/netG_A2B_epoch_30.pth', map_location=device))
    netG_A2B.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    testA_path = 'vangogh2photo/testA'
    output_dir = 'comparison_results'
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(testA_path)[:5]:
        img_path = os.path.join(testA_path, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            fake = netG_A2B(img_tensor)

        fake = fake.squeeze(0).cpu() * 0.5 + 0.5
        fake_img = transforms.ToPILImage()(fake)

        # 拼接原图和结果图
        combined = Image.new('RGB', (512, 256))
        combined.paste(img.resize((256,256), Image.BICUBIC), (0, 0))
        combined.paste(fake_img, (256, 0))
        combined.save(os.path.join(output_dir, f'compare_{fname}'))
        print(f'生成对比图：{fname}')

    print(f'所有对比图保存在 {output_dir} 文件夹')

if __name__ == '__main__':
    batch_test()

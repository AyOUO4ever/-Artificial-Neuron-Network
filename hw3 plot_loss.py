import matplotlib.pyplot as plt

# 从终端日志手动记录的Loss数据（每个epoch最后的显示值）
epochs = list(range(1, 31))
G_loss = [7.98, 8.99, 11, 6.52, 11.1, 9.16, 7.61, 6.19, 6.42, 9.17,
          7.74, 11.2, 7.23, 7.14, 9.26, 10.1, 10.5, 6.3, 6.36, 8.12,
          8.93, 8.72, 7.27, 8.26, 7.9, 6.63, 5.65, 5.19, 7.21, 5.34]
D_A_loss = [0.0624, 0.287, 0.242, 0.146, 0.0921, 0.193, 0.182, 0.325, 0.188, 0.328,
            0.302, 0.105, 0.428, 0.282, 0.0453, 0.0645, 0.0886, 0.212, 0.0843, 0.135,
            0.123, 0.0184, 0.0564, 0.167, 0.106, 0.0138, 0.588, 0.301, 0.0872, 0.0519]
D_B_loss = [0.0701, 0.397, 0.142, 0.108, 0.08, 0.0818, 0.368, 0.105, 0.21, 0.0136,
            0.19, 0.495, 0.0603, 0.112, 0.244, 0.137, 0.151, 0.44, 0.0901, 0.219,
            0.203, 0.365, 0.237, 0.0261, 0.206, 0.207, 0.555, 0.419, 0.00234, 0.351]

plt.figure(figsize=(12, 5))

# 生成器损失
plt.subplot(1, 2, 1)
plt.plot(epochs, G_loss, marker='o', color='red', label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator Training Loss')
plt.grid(True)
plt.legend()

# 判别器损失
plt.subplot(1, 2, 2)
plt.plot(epochs, D_A_loss, marker='s', color='blue', label='Discriminator A Loss')
plt.plot(epochs, D_B_loss, marker='^', color='green', label='Discriminator B Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator Training Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("Loss曲线已保存为 loss_curve.png")
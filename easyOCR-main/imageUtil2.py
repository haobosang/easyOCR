

import os
import cv2
import glob
import imageio
from imgaug import augmenters as iaa
import imgaug as ia

# 添加一个函数，用于将背景图片调整为与输入图片相同的大小并合并
def add_background(image, background_path):
    background = cv2.imread(background_path)  # 读取背景图片
    background = cv2.resize(background, (image.shape[1], image.shape[0]))  # 将背景图片调整为与输入图片相同的大小

    # 检查输入图片是否有透明通道
    if image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0  # 获取输入图片的透明度
        result = cv2.addWeighted(image[:, :, :3], alpha, background, 1 - alpha, 0)
    else:
        # 如果输入图片没有透明通道，直接将背景图片与输入图片叠加
        result = cv2.addWeighted(image, 0.7, background, 0.3, 0)

    return result

# 定义数据增强操作
def add_shadow(image):
    # 创建一个与输入图像大小相同的阴影图像
    shadow = ia.quokka(size=image.shape[:2], extract="square")

    # 将阴影图转换为灰度图像
    shadow = iaa.Grayscale(alpha=1.0)(image=shadow)

    # 调整阴影图像的亮度
    shadow = iaa.LinearContrast(alpha=(0.5, 1.5))(image=shadow)

    # 将阴影图像与输入图像叠加
    result = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2 * 255), per_channel=0.5)(images=[image, shadow])

    return result[0]  # 指定输入和输出文件夹


    # 定义数据增强操作
augmenters = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 1.0)),  # 高斯模糊
])

# 指定输入和输出文件夹

input_folder = "./partImage"  # 设置输入文件夹路径
output_folder = "./partImageNew"  # 设置输出文件夹路径


# 读取所有.png文件
image_files = glob.glob(os.path.join(input_folder, "*.png"))

# 对每个图像进行数据增强并保存到输出文件夹
for image_file in image_files:
    # 读取原始图像
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)  # 保留透明通道
    # 对原始图像进行数据增强
    augmented_image = augmenters(image=image)
    # 添加阴影
    shadow_image = add_shadow(augmented_image)
    # 添加背景
    background_path = "./invoice.png"  # 设置背景图片路径
    final_image = add_background(shadow_image, background_path)
    # 保存增强后的图像到输出文件夹
    output_file = os.path.join(output_folder, os.path.basename(image_file))
    cv2.imwrite(output_file, final_image)

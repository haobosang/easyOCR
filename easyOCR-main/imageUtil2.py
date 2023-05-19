

import os

import glob
import imageio
from imgaug import augmenters as iaa
import imgaug as ia


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
    iaa.PiecewiseAffine(scale=(0.01, 0.02)),  # 扭曲
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 1.0)),  # 高斯模糊
        iaa.MotionBlur(k=5, angle=[-45, 45])  # 运动模糊
    ])
])

# 指定输入和输出文件夹

input_folder = "./partImage"  # 设置输入文件夹路径
output_folder = "./partImageNew"  # 设置输出文件夹路径


# 读取所有.png文件
image_files = glob.glob(os.path.join(input_folder, "*.png"))

# 对每个图像进行数据增强并保存到输出文件夹
for image_file in image_files:
    image = imageio.imread(image_file)
    augmented_image = augmenters(image=image)
    shadow_image = add_shadow(augmented_image)
    output_file = os.path.join(output_folder, os.path.basename(image_file))
    imageio.imwrite(output_file, augmented_image)


# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# def apply_transformations(img):
#     # 随机扭曲图像
#
#     transform = ia.PerspectiveTransform(scale=(0.01, 0.1))
#     img = transform.apply_image(img)
#
#     # 随机旋转图像
#     angle = random.uniform(-30, 30)  # 设置随机旋转角度范围
#     img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
#
#     # 随机应用模糊滤波器
#     if random.random() > 0.5:
#         img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
#
#     return img
#
# for filename in os.listdir(input_folder):
#     if filename.endswith(".png"):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)
#
#         # 读取图像
#         img = Image.open(input_path)
#
#         # 应用数据增强
#         img_transformed = apply_transformations(img)
#
#         # 保存增强后的图像
#         img_transformed.save(output_path)
#
# print("数据增强完成！")
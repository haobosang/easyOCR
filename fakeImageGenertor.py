import csv
import os.path
import glob

from PIL import Image, ImageDraw, ImageFont
import json


def main():
    """
    这个模块用于实现根据 json 文件生成假数据图片。

    作者：Meng Pan
    日期：2023-05-18
    """
    # 获取假数据
    json_dir = glob.glob(".\\fakeLabel\\*.json")
    for json_file in json_dir:
        # 创建背板
        template = Image.open(".\\template_img\\2.png")
        draw = ImageDraw.Draw(template)

        print("Loaded {}".format(json_file))
        j_name = json_file.split('\\')[-1].split('.')[0]
        with open('fakeLabel/{}.json'.format(j_name), 'r', encoding="utf-8") as file:
            fstr = file.read()
            fakeData = json.loads(fstr)
            data = dict(fakeData["data"])
            for fd in data:
                # 添加标题
                font = ImageFont.truetype('meiryo.ttc', size=fd["fontsize"], encoding='Shift_JIS')
                text = str(fd["text"])
                left = fd["left"]
                top = fd["top"]
                width = fd["width"]
                height = fd["height"]
                draw.rectangle(xy=[left, top, left + width, top + height], fill=fd["backcolor"])
                draw.text((left, top), text, font=font, fill=fd["fontcolor"])

            # 保存图片, 这里可以使用 csv 中 image_name 域获取字符串
            img_name = j_name.replace("label", "image")
            print('Generated ./fakeImage/{}.png !\n'.format(img_name))
            template.save('./fakeImage/{}.png'.format(img_name))

            file.close()

        template.close()


if __name__ == '__main__':
    main()

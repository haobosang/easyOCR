import random
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
import json


def main():
    # 创建画布
    img = Image.open('.\\img\\2.png')
    draw = ImageDraw.Draw(img)

    # 获取假数据
    with open('sh.json', 'r', encoding="utf-8") as file:
        fstr = file.read()
        fakeData = json.loads(fstr)
        data = fakeData["data"]
        for fd in data:
            # 添加标题
            font = ImageFont.truetype('meiryo.ttc', size=fd["fontsize"], encoding='Shift_JIS')
            text = fd["text"]
            # text_width, text_height = draw.textsize(text, font=font)
            left = fd["left"]
            top = fd["top"]
            width = fd["width"]
            height = fd["height"]
            draw.rectangle(xy=[left, top, left + width, top + height], fill=fd["backcolor"])
            draw.text((left, top), text, font=font, fill=fd["fontcolor"])

        # 保存图片
        img.save('invoice.png')


if __name__ == '__main__':
    main()

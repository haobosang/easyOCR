import os.path

from PIL import Image, ImageDraw, ImageFont
import json


def main():
    # 获取假数据
    json_dir = os.path.join("./tmp")
    for json_file in os.listdir(json_dir):
        # 创建背板
        img = Image.open("./img/2.png")
        draw = ImageDraw.Draw(img)

        j_name = json_file.split('\\')[-1].split('.')[0]
        with open(f'tmp/{j_name}.json', 'r', encoding="utf-8") as file:
            fstr = file.read()
            fakeData = json.loads(fstr)
            data = fakeData["data"]
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

            # 保存图片
            img.save(f'./fakeImage/{j_name}.png')


if __name__ == '__main__':
    main()

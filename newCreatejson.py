from faker import Faker
import os
import json
import random
import csv
import json
import os
from datetime import datetime
fake = Faker('ja_JP')
#generator = fake.generator
measures = ["個", "台", "箱", "碗", "袋", "款", "セット", "枚", "本", "台", "万", "着", "対", "枚", "包",
                    "部", "袋", "隻", "本", "福", "本", "本", "粒", "冊", "枚", "層", "頭", "頭", "トン", "匹"
        ]
def createjson(textnumber=20):
    for i in range(textnumber):
        data = []
        company = fake.company()
        name = fake.name()
        text = fake.text()
        job = fake.job()
        phone_number = fake.phone_number()
        region = fake.address()
        # generator = fake.generator

        date = fake.date_this_century(before_today=True, after_today=False)
        # Format the date as a string in the desired format
        formatted_date = date.strftime('%Y年%m月%d日')

        block = random.randint(1,10)
        data.append({
            "block": 0,
            "height": 35,
            "left": 1083,
            "line": 1,
            "page": 1,
            "text": "    No. : " + phone_number,
            "top": 160,
            "width": 320,
            "word": 1,
            "fontsize": 22,
            "fontcolor": "#000000",
            "backcolor": "#FFFFFF"
        })
        data.append({
            "block": 0,
            "height": 35,
            "left": 1083,
            "line": 1,
            "page": 1,
            "text": "請求日 : " + formatted_date,
            "top": 195,
            "width": 320,
            "word": 1,
            "fontsize": 22,
            "fontcolor": "#000000",
            "backcolor": "#FFFFFF"
        })
        data.append({
            "block": 0,
            "height": 45,
            "left": 308,
            "line": 1,
            "page": 1,
            "text": company,
            "top": 260,
            "width": 420,
            "word": 1,
            "fontsize": 32,
            "fontcolor": "#000000",
            "backcolor": "#FFFFFF"
        })
        data.append({
            "block": 0,
            "height": 35,
            "left": 244,
            "line": 1,
            "page": 1,
            "text": region,
            "top": 318,
            "width": 530,
            "word": 1,
            "fontsize": 22,
            "fontcolor": "#000000",
            "backcolor": "#FFFFFF"
        })
        data.append({
            "block": 0,
            "height": 35,
            "left": 376,
            "line": 2,
            "page": 1,
            "text": "{} 担当者:{}".format(job, name),
            "top": 362,
            "width": 320,
            "word": 1,
            "fontsize": 24,
            "fontcolor": "#000000",
            "backcolor": "#FFFFFF"
        })
        data.append({
            "block": 0,
            "height": 35,
            "left": 242,
            "line": 1,
            "page": 1,
            "text": "下記",
            "top": 446,
            "width": 50,
            "word": 1,
            "fontsize": 24,
            "fontcolor": "#000000",
            "backcolor": "#FFFFFF"
        })
        data.append({
            "block": 0,
            "height": 35,
            "left": 290,
            "line": 1,
            "page": 1,
            "text": "の通りご請求申し上げます。",
            "top": 446,
            "width": 480,
            "word": 2,
            "fontsize": 24,
            "fontcolor": "#000000",
            "backcolor": "#FFFFFF"
        })

        data.append({
            "block": 0,
            "height": 35,
            "left": 317,
            "line": 1,
            "page": 1,
            "text": "商品名  /  品目",
            "top": 610,
            "width": 200,
            "word": 1,
            "fontsize": 24,
            "fontcolor": "#FFFFFF",
            "backcolor": "#2F5495"
        })
        for i in range(block):
            product_name = fake.word() + ' ' + fake.word()
            product_count = fake.random_int(min=1, max=100)
            measures_number = random.randint(0,len(measures)-1)

            if i%2!=0:
                data.append({
                    "block": i,
                    "height": 30,
                    "left": 298,
                    "line": 1,
                    "page": 1,
                    "text": product_name,
                    "top": 660+(i-1)*45,
                    "width": 470,
                    "word": 1,
                    "fontsize": 22,
                    "fontcolor": "#000000",
                    "backcolor": "#FFFFFF"
                })
                data.append({
                    "block": i,
                    "height": 30,
                    "left": 800,
                    "line": 1,
                    "page": 1,
                    "text": str(product_count)+ measures[measures_number],
                    "top": 660+(i-1)*45,
                    "width": 190,
                    "word": 8,
                    "fontsize": 22,
                    "fontcolor": "#000000",
                    "backcolor": "#FFFFFF"
                })
            else:
                data.append({
                    "block": i,
                    "height": 30,
                    "left": 298,
                    "line": 1,
                    "page": 1,
                    "text": product_name,
                    "top": 660 + (i - 1) * 45,
                    "width": 470,
                    "word": 1,
                    "fontsize": 22,
                    "fontcolor": "#000000",
                    "backcolor": "#D9E2F1"
                })
                data.append({
                    "block": i,
                    "height": 30,
                    "left": 800,
                    "line": 1,
                    "page": 1,
                    "text": str(product_count)+ measures[measures_number],
                    "top": 660 + (i - 1) * 45,
                    "width": 190,
                    "word": 8,
                    "fontsize": 22,
                    "fontcolor": "#000000",
                    "backcolor": "#D9E2F1"
                })

        json_data = {
            "data": data,
            "message": "OK",
            "success": True
        }
        # print(json_data)
        file_path = "./tmp/data" + str(i) + ".json"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(json_data, f)


def createcsv(textnumber=20):
    csv_file_path = "./fakeLabel/data.csv"
    fieldnames = ["filename","left", "top", "right", "bottom", "text"]
    data_list = []
    for i in range(textnumber):
        json_file_path = f"./tmp/data{i}.json"
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
            for item in json_data["data"]:
                left = item["left"]
                top = item["top"]
                right = left + item["width"]
                bottom = top + item["height"]
                text = item["text"]
                filename = "image"+str(i)+".jpg"
                data_list.append({"filename":filename,"left": left, "top": top, "right": right, "bottom": bottom, "text": text})

    with open(csv_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in data_list:
            writer.writerow(item)


if __name__ == '__main__':
    filennumber = 20
    createjson(filennumber)
    createcsv(filennumber)
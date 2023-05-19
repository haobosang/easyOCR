from faker import Faker
# import os
# import json
import random
import csv
import json
import os

fake = Faker('ja_JP')
measures = ["個", "台", "箱", "碗", "袋", "款", "セット", "枚", "本", "台", "万", "着", "対", "枚", "包", "部", "袋", "隻", "本", "福", "本",
            "本", "粒", "冊", "枚", "層", "頭", "頭", "トン", "匹"]


def createjson(textnumber=20):
    """
    这个模块用来实现创建 json 文件。

    作者：Haobo Li
    日期：2023-05-18
    """
    for num in range(textnumber):
        data = []
        company = fake.company()
        name = fake.name()
        job = fake.job()
        phone_number = fake.phone_number()
        region = fake.address()

        date = fake.date_this_century(before_today=True, after_today=False)
        # Format the date as a string in the desired format
        formatted_date = date.strftime('%Y-%m-%d')

        product_amount = fake.pyfloat(left_digits=9, right_digits=0, positive=True, min_value=100000000, max_value=999999999)

        num_item = random.randint(5, 24)
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
            "height": 45,
            "left": 443,
            "line": 2,
            "page": 1,
            "text": str(product_amount),
            "top": 490,
            "width": 320,
            "word": 1,
            "fontsize": 36,
            "fontcolor": "#000000",
            "backcolor": "#D9E2F1"
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

        for i in range(num_item):
            product_name = fake.word() + ' ' + fake.word()
            product_count = fake.random_int(min=1, max=100)
            measures_number = random.randint(0, len(measures) - 1)

            data.append({
                "block": i,
                "height": 30,
                "left": 298,
                "line": 1,
                "page": 1,
                "text": product_name,
                "top": 660 + i * 45,
                "width": 470,
                "word": 1,
                "fontsize": 22,
                "fontcolor": "#000000",
                "backcolor": "#FFFFFF" if i % 2 == 0 else "#D9E2F1"
            })

            data.append({
                "block": i,
                "height": 30,
                "left": 800,
                "line": 1,
                "page": 1,
                "text": str(product_count) + measures[measures_number],
                "top": 660 + i * 45,
                "width": 190,
                "word": 8,
                "fontsize": 22,
                "fontcolor": "#000000",
                "backcolor": "#FFFFFF" if i % 2 == 0 else "#D9E2F1"
            })

        json_data = {
            "data": data,
            "message": "OK",
            "success": True
        }
        # print(json_data)
        file_path = "./fakeLabel/label" + str(num) + ".json"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(json_data, f)
            f.close()
        print(">> Create json: ", file_path)


def createcsv(num=20):
    """
    这个模块根据生成的 json 文件生成 csv 文件。

    作者：Haobo Li
    日期：2023-05-18
    """
    fieldnames = ["image_name", "left", "top", "right", "bottom", "text"]
    data_list = []
    print("Converting to csv: [", end='')
    for i in range(num):
        json_file = f"./fakeLabel/label{i}.json"
        filename = json_file.split("/")[-1].split(".")[0]
        with open(json_file, "r") as f:
            json_data = json.load(f)
            for item in json_data["data"]:
                left = item["left"]
                top = item["top"]
                right = left + item["width"]
                bottom = top + item["height"]
                text = item["text"]
                data_list.append({"image_name": filename.replace("label", "image"),
                                  "left": left,
                                  "top": top,
                                  "right": right,
                                  "bottom": bottom,
                                  "text": text})
            f.close()

        print('.', end='')
    print("] Done!")

    with open("./fakeLabel/label.csv", "w", newline="", encoding="Shift_JIS") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in data_list:
            writer.writerow(item)

        f.close()


def readcsv():
    fieldnames = ["image_name", "left", "top", "right", "bottom", "text"]
    with open("./fakeLabel/label.csv", "r", newline="", encoding="Shift_JIS") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in reader:
            print(row)

        f.close()


if __name__ == '__main__':
    num_data = 20
    createjson(num_data)
    createcsv(num_data)

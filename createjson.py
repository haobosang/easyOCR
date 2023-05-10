from faker import Faker
import os
import json
import random
from datetime import datetime
fake = Faker('ja_JP')
#generator = fake.generator
data = []
for i in range(5):

    company = fake.company()
    name = fake.name()
    text = fake.text()
    job = fake.job()
    phone_number = fake.phone_number()
    region = fake.address()
    #generator = fake.generator

    date = fake.date_this_century(before_today=True, after_today=False)

    # Format the date as a string in the desired format
    formatted_date = date.strftime('%Y年%m月%d日')
    amount = fake.pyfloat(left_digits=6, right_digits=2, positive=True, min_value=10000, max_value=999999)
    product_name1 = fake.word() + ' ' + fake.word()
    product_count1 = fake.random_int(min=1, max=100)
    product_name2 = fake.word() + ' ' + fake.word()
    product_count2 = fake.random_int(min=1, max=100)
    product_name3 = fake.word() + ' ' + fake.word()
    product_count3 = fake.random_int(min=1, max=100)
    product_name4 = fake.word() + ' ' + fake.word()
    product_count4 = fake.random_int(min=1, max=100)
    product_name5 = fake.word() + ' ' + fake.word()
    product_count5 = fake.random_int(min=1, max=100)

    data.append({
        "block": 2,
        "height": 35,
        "left": 1083,
        "line": 1,
        "page": 1,
        "text": "    No. :"+phone_number,
        "top": 160,
        "width": 320,
        "word": 1,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
         "block": 2,
        "height": 35,
        "left": 1083,
        "line": 1,
        "page": 1,
        "text": "請求日 :    "+formatted_date,
        "top": 195,
        "width": 320,
        "word": 1,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 2,
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
        "block": 2,
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
        "block": 2,
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
        "block": 3,
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
         "block": 3,
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
        "block": 4,
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
        "backcolor": "#2F4097"
    })

    data.append({
        "block": 5,
        "height": 30,
        "left": 298,
        "line": 1,
        "page": 1,
        "text": product_name1,
        "top": 660,
        "width": 470,
        "word": 1,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 800,
        "line": 1,
        "page": 1,
        "text": str(product_count1)+"個数",
        "top": 660,
        "width": 190,
        "word": 8,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 298,
        "line": 2,
        "page": 1,
        "text": product_name2,
        "top": 705,
        "width": 470,
        "word": 1,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#D9E2F1"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 800,
        "line": 2,
        "page": 1,
        "text": str(product_count2)+"台",
        "top": 705,
        "width": 190,
        "word": 15,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#D9E2F1"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 298,
        "line": 3,
        "page": 1,
        "text": product_name3,
        "top": 750,
        "width": 470,
        "word": 1,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
            "height": 30,
            "left": 445,
            "line": 3,
            "page": 1,
            "text": "の取付作業",
            "top": 750,
            "width": 200,
            "word": 8,
            "fontsize": 22,
            "fontcolor": "#000000",
            "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 800,
        "line": 3,
        "page": 1,
        "text": str(product_count3)+" X",
        "top": 750,
        "width": 190,
        "word": 11,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 298,
        "line": 4,
        "page": 1,
        "text": product_name4,
        "top": 795,
        "width": 470,
        "word": 1,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#D9E2F1"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 495,
        "line": 4,
        "page": 1,
        "text": "の操作説明講習会",
        "top": 795,
        "width": 200,
        "word": 8,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#D9E2F1"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 800,
        "line": 4,
        "page": 1,
        "text": str(product_count4)+" 個数",
        "top": 795,
        "width": 190,
        "word": 13,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#D9E2F1"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 298,
        "line": 5,
        "page": 1,
        "text": "ロロロロ",
        "top": 840,
        "width": 470,
        "word": 1,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 395,
        "line": 5,
        "page": 1,
        "text": product_name5,
        "top": 840,
        "width": 200,
        "word": 3,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 800,
        "line": 5,
        "page": 1,
        "text": str(product_count5)+" Kg",
        "top": 840,
        "width": 190,
        "word": 3,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 1200,
        "line": 5,
        "page": 1,
        "text": amount-amount*0.1,
        "top": 1745,
        "width": 200,
        "word": 3,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 1200,
        "line": 5,
        "page": 1,
        "text": amount*0.1,
        "top": 1790,
        "width": 200,
        "word": 3,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 1200,
        "line": 5,
        "page": 1,
        "text": amount,
        "top": 1830,
        "width": 200,
        "word": 3,
        "fontsize": 22,
        "fontcolor": "#000000",
        "backcolor": "#FFFFFF"
    })



    json_data = {
        "data": data,
        "message": "OK",
        "success": True
    }
    #print(json_data)
    file_path = "./tmp/data"+str(i)+".json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(json_data, f)
#print(json_data)
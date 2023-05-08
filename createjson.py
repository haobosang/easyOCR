from faker import Faker
import os
import json
fake = Faker('ja_JP')

data = []
for i in range(5):
    company = fake.company()
    name = fake.name()
    text = fake.text()
    job = fake.job()
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
        "height": 32,
        "left": 88,
        "line": 1,
        "page": 1,
        "text": company,
        "top": 250,
        "width": 269,
        "word": 1
    })

    data.append({
        "block": 2,
        "height": 31,
        "left": 89,
        "line": 1,
        "page": 1,
        "text": "{} 担当者:{}".format(job, name),
        "top": 282,
        "width": 288,
        "word": 1
    })

    data.append({
        "block": 3,
        "height": 18,
        "left": 89,
        "line": 2,
        "page": 1,
        "text": "下記",
        "top": 328,
        "width": 155,
        "word": 1
    })

    data.append({
        "block": 3,
        "height":  23,
        "left": 119,
        "line": 3,
        "page": 1,
        "text": "の通りご請求申し上げます。",
        "top":  324,
        "width":  219,
        "word": 2
    })
    data.append({
        "block": 4,
        "height": 23,
        "left": 114,
        "line": 1,
        "page": 1,
        "text": "商品名/品目",
        "top": 468,
        "width": 148,
        "word": 1
    })
    data.append({
        "block": 5,
        "height": 16,
        "left": 94,
        "line": 1,
        "page": 1,
        "text": product_name1,
        "top": 513,
        "width": 295,
        "word": 1
    })
    data.append({
        "block": 5,
        "height": 29,
        "left": 622,
        "line": 1,
        "page": 1,
        "text": "個数",
        "top": 508,
        "width": 36,
        "word": 8
    })
    data.append({
        "block": 5,
        "height": 18,
        "left": 95,
        "line": 2,
        "page": 1,
        "text": product_name2,
        "top": 547,
        "width": 361,
        "word": 1
    })
    data.append({
        "block": 5,
        "height": 16,
        "left": 604,
        "line": 2,
        "page": 1,
        "text": str(product_count1)+"個数",
        "top": 545,
        "width": 35,
        "word": 15
    })
    data.append({
        "block": 5,
        "height": 17,
        "left": 96,
        "line": 3,
        "page": 1,
        "text": product_name3,
        "top": 579,
        "width": 259,
        "word": 1
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 244,
        "line": 3,
        "page": 1,
        "text": product_name4,
        "top": 574,
        "width": 113,
        "word": 8
    })
    data.append({
        "block": 5,
        "height": 16,
        "left": 604,
        "line": 3,
        "page": 1,
        "text": str(product_count2)+"X",
        "top": 578,
        "width": 35,
        "word": 11
    })
    data.append({
         "block": 5,
        "height": 16,
        "left": 97,
        "line": 4,
        "page": 1,
        "text": product_name5,
        "top": 612,
        "width": 258,
        "word": 1
    })
    data.append({
        "block": 5,
        "height": 30,
        "left": 243,
        "line": 4,
        "page": 1,
        "text": "の操作説明講習会",
        "top": 606,
        "width": 190,
        "word": 8
    })
    data.append({
        "block": 5,
        "height": 29,
        "left": 583,
        "line": 4,
        "page": 1,
        "text": str(product_count5)+"個数",
        "top": 606,
        "width": 78,
        "word": 13
    })
    data.append({
        "block": 5,
        "height": 15,
        "left": 97,
        "line": 5,
        "page": 1,
        "text": "ロロ",
        "top": 645,
        "width": 47,
        "word": 1
    })
    data.append({
        "block": 5,
        "height": 26,
        "left": 246,
        "line": 5,
        "page": 1,
        "text": "素材(XXを含む)|50Kg",
        "top": 643,
        "width": 397,
        "word": 3
    })



    json_data = {
        "data": data,
        "message": "OK",
        "success": True
    }
    file_path = "/home/ubuntu/testjson/data"+str(i)+".json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(json_data, f)
#print(json_data)
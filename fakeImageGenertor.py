import random
from typing import Union

from faker import Faker
from PIL import Image, ImageDraw, ImageFont


def main():
    # 创建画布
    # width = 1000
    # height = 1500
    # img = Image.new('RGB', (width, height), color = (255, 255, 255))
    img = Image.open('.\\img\\2.png')
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)

    # 生成虚假数据
    faker = Faker()
    invoice_number = faker.random_number(digits=6)
    invoice_date = faker.date_between(start_date='-30d', end_date='today')
    company_name = faker.company()
    company_address = faker.address()
    item_names = ['Item A', 'Item B', 'Item C', 'Item D', 'Item E']
    item_prices = [random.uniform(10, 100) for i in range(len(item_names))]
    subtotal = sum(item_prices)
    tax_rate = 0.05
    tax = subtotal * tax_rate
    total = subtotal + tax

    # 添加标题
    font = ImageFont.truetype('arial.ttf', size=48)
    title_text = 'Invoice'  # Invoice
    title_width, title_height = draw.textsize(title_text, font=font)
    title_x = width * 0.20  # (width - title_width) / 2
    title_y = height * 0.072  # 50
    draw.rectangle(xy=[title_x - 50, title_y, title_x + 50 + title_width, title_y + title_height],
                   fill=(47, 84, 151))
    draw.text((title_x, title_y), title_text, font=font, fill=(255, 255, 255))

    # 添加发票编号
    font = ImageFont.truetype('arial.ttf', size=24)
    invoice_number_text = f'Invoice Number: {invoice_number}'
    invoice_number_width, invoice_number_height = draw.textsize(invoice_number_text, font=font)
    invoice_number_x = width * 0.655  # 50
    invoice_number_y = title_y - 3  # title_y + title_height + 50
    draw.rectangle(xy=[invoice_number_x - 10, invoice_number_y,
                       invoice_number_x + width * 0.30, invoice_number_y + invoice_number_height],
                   fill=(255, 255, 255))
    draw.text((invoice_number_x, invoice_number_y), invoice_number_text, font=font, fill=(0, 0, 0))

    # 添加日期
    font = ImageFont.truetype('arial.ttf', size=24)
    invoice_date_text = f'Date: {invoice_date.strftime("%m/%d/%Y")}'
    invoice_date_width, invoice_date_height = draw.textsize(invoice_date_text, font=font)
    invoice_date_x = width * 0.655  # width - invoice_date_width - 50
    invoice_date_y = title_y + 30  # invoice_number_y
    draw.rectangle(xy=[invoice_date_x - 10, invoice_date_y,
                       invoice_date_x + 150 + width * 0.30, invoice_date_y + invoice_date_height], fill=(255, 255, 255))
    draw.text((invoice_date_x, invoice_date_y), invoice_date_text, font=font, fill=(0, 0, 0))

    # 添加公司信息
    font = ImageFont.truetype('arial.ttf', size=36)
    company_name_text = f'Company Name: {company_name}'
    company_name_width, company_name_height = draw.textsize(company_name_text, font=font)
    company_name_x = width * 0.145  # 50
    company_name_y = height * 0.112  # invoice_number_y + invoice_number_height + 50
    draw.rectangle(xy=[company_name_x - 10, company_name_y,
                       company_name_x + width * 0.40, company_name_y + company_name_height],
                   fill=(255, 255, 255))
    draw.text((company_name_x, company_name_y), company_name_text, font=font, fill=(0, 0, 0))

    # 添加地址信息
    font = ImageFont.truetype('arial.ttf', size=28)
    company_address_text = f'Address: {company_address}'
    company_address_width, company_address_height = draw.textsize(company_address_text, font=font)
    company_address_x = width * 0.145  # 50
    company_address_y = height * 0.135  # company_name_y + company_name_height + 10
    draw.rectangle(xy=[company_address_x - 10, company_address_y,
                       company_address_x + width * 0.40, company_address_y + company_address_height],
                   fill=(255, 255, 255))
    draw.rectangle(xy=[company_address_x - 10, company_address_y + company_address_height,
                       company_address_x + width * 0.40, company_address_y + company_address_height * 2],
                   fill=(255, 255, 255))
    draw.text((company_address_x, company_address_y), company_address_text, font=font, fill=(0, 0, 0))

    # 添加项目
    font = ImageFont.truetype('arial.ttf', size=30)
    item_header_text1 = 'Item Name'
    item_header_width1, item_header_height1 = draw.textsize(item_header_text1, font=font)
    item_header_x1 = width * 0.19  # 50
    item_header_y1 = height * 0.261  # company_address_y + company_address_height + 50
    draw.rectangle(xy=[item_header_x1, item_header_y1,
                       item_header_x1 + width * 0.28, item_header_y1 + item_header_height1],
                   fill=(47, 84, 151))
    draw.text((item_header_x1, item_header_y1), item_header_text1, font=font, fill=(255, 255, 255))

    # 表头
    item_header_text2 = 'Item Price'
    item_header_width2, item_header_height2 = draw.textsize(item_header_text2, font=font)
    item_header_x2 = width * 0.615  # 50
    item_header_y2 = height * 0.261  # company_address_y + company_address_height + 50
    draw.rectangle(xy=[item_header_x2, item_header_y2,
                       item_header_x2 + width * 0.10, item_header_y2 + item_header_height2],
                   fill=(47, 84, 151))
    draw.text((item_header_x2, item_header_y2), item_header_text2, font=font, fill=(255, 255, 255))

    item_y = item_header_y1 + item_header_height1 + 22
    for i in range(len(item_names)):
        item_name = item_names[i]
        # item_price = item_prices[i]
        # item_text = f'{item_name}\t${item_price:.2f}'
        item_text = f'{item_name}'
        item_width, item_height = draw.textsize(item_text, font=font)
        draw.rectangle(xy=[item_header_x1 - 18, item_y,
                           item_header_x1 + width * 0.26, item_y + item_height],
                       fill=(255, 255, 255) if i % 2 == 0 else (217, 226, 241))
        draw.text((item_header_x1, item_y), item_text, font=font, fill=(0, 0, 0))

        item_price = item_prices[i]
        item_text = f'${item_price:.2f}'
        item_width, item_height = draw.textsize(item_text, font=font)
        draw.rectangle(xy=[item_header_x2, item_y,
                           item_header_x2 + width * 0.09, item_y + item_height],
                       fill=(255, 255, 255) if i % 2 == 0 else (217, 226, 241))
        draw.text((item_header_x2, item_y), item_text, font=font, fill=(0, 0, 0))

        item_y += item_height + 15



    font = ImageFont.truetype('arial.ttf', size=28)

    # 添加小计、税率和总计
    subtotal_text = f'Subtotal: ${subtotal:.2f}'
    subtotal_width, subtotal_height = draw.textsize(subtotal_text, font=font)
    subtotal_x = width * 0.602  # width - subtotal_width - 50
    subtotal_y = height * 0.745
    draw.rectangle(xy=[subtotal_x, subtotal_y,
                       subtotal_x + width * 0.25, subtotal_y + subtotal_height + 5],
                   fill=(255, 255, 255))
    draw.text((subtotal_x, subtotal_y), subtotal_text, font=font, fill=(0, 0, 0))

    tax_rate_text = f'Tax Rate: {tax_rate*100:.0f}%'
    tax_rate_width, tax_rate_height = draw.textsize(tax_rate_text, font=font)
    tax_rate_x = subtotal_x
    tax_rate_y = subtotal_y + subtotal_height + 18
    draw.rectangle(xy=[tax_rate_x, tax_rate_y,
                       tax_rate_x + width * 0.25, tax_rate_y + tax_rate_height + 5],
                   fill=(255, 255, 255))
    draw.text((tax_rate_x, tax_rate_y), tax_rate_text, font=font, fill=(0, 0, 0))

    tax_text = f'Tax: ${tax:.2f}'
    tax_width, tax_height = draw.textsize(tax_text, font=font)
    tax_x = subtotal_x
    tax_y = tax_rate_y + tax_rate_height + 18
    draw.rectangle(xy=[tax_x, tax_y,
                       tax_x + width * 0.25, tax_y + tax_height + 5],
                   fill=(255, 255, 255))
    draw.text((tax_x, tax_y), tax_text, font=font, fill=(0, 0, 0))

    total_text = f'Total: ${total:.2f}'
    total_width, total_height = draw.textsize(total_text, font=font)
    total_x = subtotal_x
    total_y = tax_y + tax_height + 18
    draw.rectangle(xy=[total_x, total_y,
                       total_x + width * 0.25, total_y + total_height + 5],
                   fill=(255, 255, 255))
    draw.text((total_x, total_y), total_text, font=font, fill=(0, 0, 0))

    # 保存图片
    img.save('invoice.png')


if __name__ == '__main__':
    main()

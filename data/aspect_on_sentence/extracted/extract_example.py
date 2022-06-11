from lxml import etree as ET
import csv
import random

train = ET.parse("../translated/ABSA16_Laptops_Train_Polish_SB1.xml")
test = ET.parse("../translated/ABSA16_Laptops_Test_Polish_SB1.xml")

reviews = train.getroot().findall("Review")
ids = []
for review in reviews:
    ids.append(review.attrib["rid"])

random.shuffle(ids)
valid_ids=ids[:75]
valid_data=[]

with open('train.csv', 'w') as file:
    writer = csv.writer(file)
    for review in train.getroot()[10:]:
        for sentence in review[0]:
            for opinion in sentence[1]:
                record = [sentence.attrib["id"], len(sentence[2].text), len(sentence[2].text.split(" ")), opinion.attrib["category"], opinion.attrib["polarity"], sentence[2].text]
                if not review.attrib["rid"] in valid_ids:
                    writer.writerow(record)
                else:
                    valid_data.append(record)

with open('valid.csv', 'w') as file:
    writer = csv.writer(file)
    for record in valid_data:
        writer.writerow(record)

with open('test.csv', 'w') as file:
    writer = csv.writer(file)
    for review in test.getroot():
        for sentence in review[0]:
            for opinion in sentence[1]:
                writer.writerow([sentence.attrib["id"], len(sentence[2].text),len(sentence[2].text.split(" ")), opinion.attrib["category"], opinion.attrib["polarity"], sentence[2].text])


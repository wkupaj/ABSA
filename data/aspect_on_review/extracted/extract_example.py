from lxml import etree as ET
import csv

tree = ET.parse("../translated/ABSA16_Laptops_Train_Polish_SB2.xml")
root = tree.getroot()

with open('train.csv', 'w') as file:
    writer = csv.writer(file)
    for review in root[20:]:
        data = []
        sentences = ""
        for sentence in review[0]:
            sentences = sentences + " " + sentence[1].text
        for opinion in review[1]:
            writer.writerow([review.attrib["rid"], len(sentences),len(sentences.split(" ")), opinion.attrib["category"], opinion.attrib["polarity"], sentences])


with open('valid.csv', 'w') as file:
    writer = csv.writer(file)
    for review in root[:10]:
        data = []
        sentences = ""
        for sentence in review[0]:
            sentences = sentences + " " + sentence[1].text
        for opinion in review[1]:
            writer.writerow([review.attrib["rid"], len(sentences),len(sentences.split(" ")), opinion.attrib["category"], opinion.attrib["polarity"], sentences])

with open('test.csv', 'w') as file:
    writer = csv.writer(file)
    for review in root[10:20]:
        data = []
        sentences = ""
        for sentence in review[0]:
            sentences = sentences + " " + sentence[1].text
        for opinion in review[1]:
            writer.writerow([review.attrib["rid"], len(sentences),len(sentences.split(" ")), opinion.attrib["category"], opinion.attrib["polarity"], sentences])


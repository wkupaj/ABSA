import csv
import random

from lxml import etree as ET


def save_file(filename, data):
    with open(filename + '.csv', 'w') as file:
        writer = csv.writer(file)
        for record in data:
            writer.writerow(record)


def reduce_label(data):
    records_grouped_by_review_id = {}
    for record in data:
        review_id = record[0].split(':')[0]
        if review_id in records_grouped_by_review_id:
            records_grouped_by_review_id[review_id].append(record)
        else:
            records_grouped_by_review_id[review_id] = [record]

    result = []
    for records in records_grouped_by_review_id.values():
        save = True
        records_grouped_by_sentence_id = {}
        for record in records:
            sentence_id= record[0].split(':')[1]
            if sentence_id in records_grouped_by_sentence_id:
                records_grouped_by_sentence_id[sentence_id].append(record)
            else:
                records_grouped_by_sentence_id[sentence_id] = [record]

        to_save = []
        for records_grouped in records_grouped_by_sentence_id.values():
            aspects = {}
            tmp = {}

            for record in records_grouped:
                aspect = record[3].split('#')[0]
                if aspect in aspects and aspect != 'LAPTOP':
                    if aspects[aspect] != record[4]:
                        save = False
                else:
                    aspects[aspect] = record[4]
                if aspect != 'LAPTOP':
                    record[3] = aspect
                    tmp[aspect] = record
                else:
                    tmp[record[3]] = record

            to_save = to_save + [i for i in tmp.values()]
        if save:
            for record in to_save:
                result.append(record)
    return result


def main():
    train = ET.parse("translated/ABSA16_Laptops_Train_Polish_SB1.xml")
    test = ET.parse("translated/ABSA16_Laptops_Test_Polish_SB1.xml")

    reviews = train.getroot().findall("Review")
    ids = []
    for review in reviews:
        ids.append(review.attrib["rid"])

    random.seed(42)
    random.shuffle(ids)
    valid_ids = ids[:45]
    valid_data = []
    train_data = []
    test_data = []

    for review in train.getroot():
        for sentence in review[0]:
            for opinion in sentence[1]:
                record = [sentence.attrib["id"], len(sentence[2].text), len(sentence[2].text.split(" ")),
                          opinion.attrib["category"], opinion.attrib["polarity"], sentence[2].text]
                if not review.attrib["rid"] in valid_ids:
                    train_data.append(record)
                else:
                    valid_data.append(record)

    for review in test.getroot():
        for sentence in review[0]:
            for opinion in sentence[1]:
                record = [sentence.attrib["id"], len(sentence[2].text), len(sentence[2].text.split(" ")),
                          opinion.attrib["category"], opinion.attrib["polarity"], sentence[2].text]
                test_data.append(record)

    save_file("extracted/train", train_data)
    save_file("extracted/valid", valid_data)
    save_file("extracted/test", test_data)

    train_data = reduce_label(train_data)
    valid_data = reduce_label(valid_data)
    test_data = reduce_label(test_data)

    save_file("reduced/train", train_data)
    save_file("reduced/valid", valid_data)
    save_file("reduced/test", test_data)


if __name__ == "__main__":
    main()

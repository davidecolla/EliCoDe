import os
import argparse
from pathlib import Path

def main(language_base_path, output_folder, output_folder_all):

    language_base_path = str(language_base_path) + "/"
    output_folder = str(output_folder) + "/"
    output_folder_all = str(output_folder_all) + "/"

    # Build output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_all, exist_ok=True)

    # Load train/dev/test data
    train_lines = open(language_base_path + 'train.tsv', 'r', encoding='utf-8').read().split('\n')
    dev_lines = open(language_base_path + 'dev.tsv', 'r', encoding='utf-8').read().split('\n')
    test_lines = open(language_base_path + 'test.tsv', 'r', encoding='utf-8').read().split('\n')

    # Convert TSV to BIO
    output_train_str = convert_tsv_to_bio(train_lines)
    output_dev_str = convert_tsv_to_bio(dev_lines)
    output_test_str = convert_tsv_to_bio(test_lines)

    # Stats
    train_sentences_number = len(output_train_str.split("\n\n"))
    dev_sentences_number = len(output_dev_str.split("\n\n"))
    test_sentences_number = len(output_test_str.split("\n\n"))

    # Write train/dev/test 
    open(output_folder + 'train.txt', 'w', encoding='utf-8').write(output_train_str)
    open(output_folder + 'dev.txt', 'w', encoding='utf-8').write(output_dev_str)
    open(output_folder + 'test.txt', 'w', encoding='utf-8').write(output_test_str)
    
    # Write train/dev/test ALL
    open(output_folder_all + 'train.txt', 'w', encoding='utf-8').write(output_train_str + "\n" + output_dev_str)
    open(output_folder_all + 'dev.txt', 'w', encoding='utf-8').write(output_dev_str)
    open(output_folder_all + 'test.txt', 'w', encoding='utf-8').write(output_test_str)

    # Print statistics
    print("##### Statistics #####")
    print("Dev sentences", dev_sentences_number)
    print("Test sentences", test_sentences_number)
    print("Train sentences", train_sentences_number)
    print("Join sentences", len((output_train_str + "\n" + output_dev_str).split("\n\n")))
    
    assert len((output_train_str + "\n" + output_dev_str).split("\n\n")) == train_sentences_number + dev_sentences_number
    
    print("Total sentences", dev_sentences_number + train_sentences_number + test_sentences_number)
    print("#############")


def convert_tsv_to_bio(lines):
    bio_data = []
    for line in lines:
        if line != "":
            split = line.split('\t')
            token = split[0].replace(" ", "")
            assert " " not in token
            label = split[1] if len(split) > 1 else "c"

            bio_data.append(token.replace('\\"', "\"") + " O O " + ("O" if label == "c" else "B-Error"))
        else:
            if bio_data[-1] != "":
                bio_data.append(line)


    if bio_data[-1] == "" and bio_data[-2] == "":
        bio_data = bio_data[:-2]

    return "\n".join(bio_data)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--language_folder", type=Path)
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--output_folder_all", type=Path)
    p = parser.parse_args()
    
    main(p.language_folder, p.output_folder, p.output_folder_all)


import argparse
from pathlib import Path


def main(bio_file, tsv_file, output_file):

    print("Converting ", bio_file)
    prediction_lines = open(bio_file, 'r', encoding='utf-8').read().split('\n')
    original_lines = open(tsv_file, 'r', encoding='utf-8').read().split('\n')

    out_lines = []
    for index, line in enumerate(prediction_lines):
        original_line = original_lines[index]
        split = line.split(" ")

        if "\"" in split[0]:
            split[0] = split[0].replace("\"", '\\"')

        assert original_line == split[0]

        if line != "":
            tag = "c"
            if split[1] != "O":
                tag = "i"
            out_lines.append(split[0] + "\t" + tag)
        else:
            out_lines.append("")

    out_str = "\n".join(out_lines)
    open(output_file, 'w').write(out_str)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bio_file", type=Path)
    parser.add_argument("--tsv_file", type=Path)
    parser.add_argument("--output_file", type=Path)
    p = parser.parse_args()
    
    main(p.bio_file, p.tsv_file, p.output_file)
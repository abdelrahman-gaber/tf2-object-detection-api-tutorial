import os
import argparse
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for file in os.scandir(path):
        if file.is_file() and file.name.endswith(('.xml')):
            xml_file = os.path.join(path, file.name)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member[0].text,
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text) )
                xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate csv file from xml annotations')
    parser.add_argument('--annot_dir', help='directory of input xml files', default='../data/xml_annot')
    parser.add_argument('--out_csv_path', help='path to output csv file', default='../data/train.csv')
    args = parser.parse_args()

    xml_df = xml_to_csv(args.annot_dir)
    xml_df.to_csv(args.out_csv_path, index=None)
    print('Successfully converted xml to csv.')

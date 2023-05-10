from src.util.data_decoder import data_input, return_feature_dict
from src.util.feature_engineering import norm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest='address', help="data address directory")
    args = parser.parse_args()
    address_dir = args.address






if __name__ == '__main__':
    main()


    # addr = 'sample_data/PE01.csv'
    # data = data_input(addr)
    # print(data)
    # dict_feature = return_feature_dict(data)
    # print(dict_feature)



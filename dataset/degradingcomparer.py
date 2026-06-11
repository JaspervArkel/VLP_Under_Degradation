import argparse


def main():
    parser = argparse.ArgumentParser("comparee_degrading")
    parser.add_argument(
        "--src", help="first degrading to compare", type=str,
        default="dataset/exported/age-series/aged_samples.csv"
    )
    parser.add_argument(
        "--src2", help="second degrading to compare", type=str,
        default="dataset/exported/age-series/degraded_samples.csv"
    )
    args = parser.parse_args()


if __name__ == '__main__':
    main()
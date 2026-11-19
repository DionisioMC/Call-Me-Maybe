from .parser import parse
from sys import argv


def main() -> None:
    args = parse(argv)
    print("Function definitions:", args[0])
    print("Input file:", args[1])
    print("Output file:", args[2])


if __name__ == "__main__":
    main()

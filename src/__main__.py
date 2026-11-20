from .parser import parse
from sys import argv


def main() -> None:
    try:
        args = parse(argv)
        print("Function definitions:", args[0])
        print("Prompts:", args[1])
        print("Output file:", args[2])
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()

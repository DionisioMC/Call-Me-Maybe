from sys import argv
from .parsing import parse
from .constrained_decoder import ConstrainedDecoder
from time import time


def main() -> None:
    try:
        initial_time = time()
        functions_json_data, tests_json_data, output_path = parse(argv)
        constrained_decoder = ConstrainedDecoder(
            functions_json_data=functions_json_data,
            tests_json_data=tests_json_data,
            output_path=output_path)
        constrained_decoder.decode()
        final_time = time()
        print(ConstrainedDecoder.print_time(
            initial_time, final_time, "\n\nFinal execution time"))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

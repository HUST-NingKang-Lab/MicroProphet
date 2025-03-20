import argparse
import os.path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Select the mode of operation and other parameters.")
    # Mode parameter
    parser.add_argument('mode', type=str, choices=['predict','split','attention','shap'],
                        help='Choose the operation mode. Use "train" for training, "predict" for prediction.')
    parser.add_argument('data_path', type=str, help='Path to the input raw CSV data. If mode is "predict", specify the test set.')
    parser.add_argument('--export_path', type=str, default='./', help='Path to export analysis results. Default is "./".')
    parser.add_argument('--export_split', action='store_true',
                        help='Whether to export the split dataset. Default is False, set to True if enabled.')
    # Add export_model parameter, default is False
    parser.add_argument('--export_model', action='store_true',
                        help='Whether to export the model. Default is False, set to True if enabled.')
    # Parse command line arguments
    args = parser.parse_args()
    return args

def main():
    # Parse command line arguments
    args = parse_arguments()
    assert args.data_path[-4:] == ".csv", "The input file is not in CSV format."
    if not os.path.exists(args.export_path) and args.mode == "predict":
        os.makedirs(args.export_path)
    if args.export_path[-1] != '/':
        args.export_path += '/'
    import run
    run.run(args=args)

if __name__ == "__main__":
    main()

"""Parsing the model parameters."""

import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MLPClassification.")

    parser.add_argument("--save-path",
                        nargs="?",
                        default="model_weights.pth",
	                help="trained model.")


    parser.add_argument("--epochs",
                        type=int,
                        default=100,
	                help="Number of training iterations. Default is 100.")


    parser.add_argument("--learning-rate",
                        type=float,
                        default=1e-3,
	                help="learning_rate parameter. Default is 1e-3.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 512 256 10.")

    parser.set_defaults(layers=[512, 256, 10])

    return parser.parse_args()

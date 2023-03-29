import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from perf_single_generative import parse_perf_single_generative_model

import onnxruntime

general_exporting_args = []
gtp2_perf_config = {
    "model_type": "gpt2",
    "model_names": ["gpt2", "gpt2-large"],  # "distilgpt2",  "gpt2-medium" , "gpt2-xl"
    "exporting_args": {
        "-b",  # no block operator
        "--use_decoder_masked_self_attention",
        "--past_present_share_buffer",
        "--use_external_data_format",
        "--use_gpu",
        "--disable_parity",
        "--disable_perf_test",
        "--total_runs=1",
    },
    "perf_variants": {
        "default": [
            # small context
            "--batch_size=1 --context_length 128 --min_length=1 --max_length=32",
            "--batch_size=2 --context_length 128 --min_length=1 --max_length=32",
            "--batch_size=4 --context_length 128 --min_length=1 --max_length=32",
            "--batch_size=8 --context_length 128 --min_length=1 --max_length=32",
            "--batch_size=16 --context_length 128 --min_length=1 --max_length=32",
            "--batch_size=32 --context_length 128 --min_length=1 --max_length=32",
            "--batch_size=64 --context_length 128 --min_length=1 --max_length=32",
            # middle context
            "--batch_size=1 --context_length 512 --min_length=1 --max_length=32",
            "--batch_size=2 --context_length 512 --min_length=1 --max_length=32",
            "--batch_size=4 --context_length 512 --min_length=1 --max_length=32",
            "--batch_size=8 --context_length 512 --min_length=1 --max_length=32",
            "--batch_size=16 --context_length 512 --min_length=1 --max_length=32",
            "--batch_size=32 --context_length 512 --min_length=1 --max_length=32",
            # varint context len
            "--batch_size=4 --context_length 32 64 96 128 160 192 224 256 --min_length=1 --max_length=32",
            "--batch_size=8 --context_length 32 64 96 128 160 192 224 256 --min_length=1 --max_length=32",
            "--batch_size=16 --context_length 32 64 96 128 160 192 224 256 --min_length=1 --max_length=32",
            "--batch_size=32 --context_length 32 64 96 128 160 192 224 256 --min_length=1 --max_length=32",
            # big initial context length
            "--batch_size=1 --context_length=1024 --min_length=1 --max_length=32",
            "--batch_size=2 --context_length=1024 --min_length=1 --max_length=32",
            "--batch_size=4 --context_length=1024 --min_length=1 --max_length=32",
            "--batch_size=8 --context_length=1024 --min_length=1 --max_length=32",
            "--batch_size=16 --context_length=1024 --min_length=1 --max_length=32",
        ]
    },
}

logger = logging.getLogger("")


def parse_arguments(argv):
    parser = argparse.ArgumentParser("perf_group_generative.py")

    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        default="gpt2",
        choices=["gpt2", "t5", "mt5"],
        help="Model type (currently only support gpt2) in the list: " + ", ".join(["gpt2", "t5", "mt5"]),
    )

    parser.add_argument(
        "--model_names",
        required=False,
        nargs="*",
        type=str,
        default=[],
        choices=["gpt2", "gpt2-large", "distilgpt2", "gpt2-medium", "gpt2-xl"],
        help="Model names to test. if not set, default list in config will be used.",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="using fp16(default) model or fp32 model",
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite existing models to be exported",
    )

    parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        help="In debug mode, only first 2 test variant for each model will be test",
    )

    parser.add_argument(
        "--workspace",
        required=False,
        type=str,
        default=os.path.join(".", "workspace"),
        help="Directory to save and perf various models and test result, final result is saved here as perf_result.txt",
    )

    args, extra = parser.parse_known_args(argv)
    return args, extra


def perform_group_perf(args, extra_exporting_args, perf_test_config):
    assert args.model_type == perf_test_config["model_type"]

    all_perf_result = {}
    all_exporting_configs = {}
    if len(args.model_names) == 0:
        args.model_names = perf_test_config["model_names"]
    for model_name in args.model_names:
        print(f"====> Model name: {model_name}")
        exporting_cmd = ["python", "convert_generation.py"]
        exporting_cmd.extend(perf_test_config["exporting_args"])
        output_model_dir = os.path.join(args.workspace, model_name)
        output_model_path = os.path.join(output_model_dir, f"model_{args.precision}.onnx")
        exporting_cmd.extend(
            [
                "-m",
                f"{model_name}",
                "--cache_dir",
                f"{args.cache_dir}",
                "--output",
                f"{output_model_path}",
                "-p",
                f"{args.precision}",
            ]
        )
        exporting_cmd.extend(extra_exporting_args)
        all_exporting_configs[model_name] = exporting_cmd

        Path(output_model_dir).mkdir(parents=True, exist_ok=True)
        if args.overwrite and os.path.exists(output_model_path):
            os.remove(output_model_path)

        if not os.path.exists(output_model_path):
            print(f"  ====> {exporting_cmd}")
            subprocess.run(exporting_cmd)

        if not os.path.exists(output_model_path):
            raise RuntimeError(f"Model {output_model_path} not found, convert_generate error?")

        all_perf_result.update({model_name: []})
        single_model_perf_result = all_perf_result[model_name]

        varconf = perf_test_config["perf_variants"]
        perf_variants = varconf["default"] if model_name not in varconf else varconf[model_name]
        for idx, perf_variant in enumerate(perf_variants):
            if args.debug and idx >= 2:
                break

            perf_args = [
                "-m",
                f"{model_name}",
                "--cache_dir",
                f"{args.cache_dir}",
                "--onnx_model",
                f"{output_model_path}",
            ]
            perf_args.extend(perf_variant.split())
            result, _ = parse_perf_single_generative_model(perf_args)
            single_model_perf_result.append({"config": perf_variant, "result": result})

    result_perf_file = os.path.join(args.workspace, "all_test_result.txt")
    with open(result_perf_file, "w") as f:
        f.write("================all_perf_result=\n")
        f.write(f"{all_perf_result}")
        f.write("\n================all_exporting_configs=\n")
        f.write(f"{all_exporting_configs}")


if __name__ == "__main__":
    # Sample usage:
    # Test on greedy
    #   python perf_group_generative.py --workspace ~/perf_gpt2/greedy --cache_dir ~/perf_gpt2/cache_models --num_beams 1
    # TODO, add Test on beam / topp args here
    #
    args, extra_exporting_args = parse_arguments(sys.argv[1:])
    perform_group_perf(args, extra_exporting_args, gtp2_perf_config)

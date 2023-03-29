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
        # WARP_OPTIONS="--num_beams=1"
        "-b",  # no block operator
        # "--use_decoder_masked_self_attention",
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
            "--batch_size=1 --context_length 128 --min_length=1 --max_length=50",
            "--batch_size=2 --context_length 128 --min_length=1 --max_length=50",
            # "--batch_size=8 --context_length 128 --min_length=1 --max_length=50",
            # "--batch_size=32 --context_length 128 --min_length=1 --max_length=50",
            # # middle context
            # "--batch_size=1 --context_length 256 --min_length=1 --max_length=50",
            # "--batch_size=2 --context_length 256 --min_length=1 --max_length=50",
            # "--batch_size=8 --context_length 256 --min_length=1 --max_length=50",
            # "--batch_size=32 --context_length 256 --min_length=1 --max_length=50",
            # # varint context len
            # "--batch_size=8 --context_length 32 64 96 128 160 192 224 256 --min_length=1 --max_length=50",
            # "--batch_size=16 --context_length 32 64 96 128 160 192 224 256 --min_length=1 --max_length=50",
            # "--batch_size=32 --context_length 32 64 96 128 160 192 224 256 --min_length=1 --max_length=50",
            # # big initial context length
            # "--batch_size=1 --context_length=1024 --min_length=1 --max_length=50",
            # "--batch_size=2 --context_length=1024 --min_length=1 --max_length=50",
            # "--batch_size=8 --context_length=1024 --min_length=1 --max_length=50",
            # "--batch_size=16 --context_length=1024 --min_length=1 --max_length=50",
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
    for model_name in perf_test_config["model_names"]:
        exporting_args = ["python", "convert_generation.py"]
        exporting_args.extend(perf_test_config["exporting_args"])
        output_model_dir = os.path.join(args.workspace, model_name)
        output_model_path = os.path.join(output_model_dir, f"model_{args.precision}.onnx")
        exporting_args.extend(
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
        # OPTIONS like "--num_beams 1", etc
        exporting_args.extend(extra_exporting_args)
        all_exporting_configs[model_name] = exporting_args

        Path(output_model_dir).mkdir(parents=True, exist_ok=True)
        if args.overwrite and os.path.exists(output_model_path):
            os.remove(output_model_path)

        if not os.path.exists(output_model_path):
            subprocess.run(exporting_args)

        if not os.path.exists(output_model_path):
            raise RuntimeError(f"Model {output_model_path} not found, convert_generate error?")

        all_perf_result.update({model_name: []})
        single_model_perf_result = all_perf_result[model_name]

        varconf = perf_test_config["perf_variants"]
        perf_variants = varconf["default"] if model_name not in varconf else varconf[model_name]
        for perf_variant in perf_variants:
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
    #   python perf_group_generative.py --workspace ~/perf_gpt2/group --cache_dir ~/perf_gpt2/cache_models --num_beams 1
    # TODO, add Test on beam / topp args here
    #
    args, extra_exporting_args = parse_arguments(sys.argv[1:])
    perform_group_perf(args, extra_exporting_args, gtp2_perf_config)

"""Generate preprocessing pipeline configurations with specified parameters."""

import json
import argparse
from typing import List, Dict, Any
from pathlib import Path
from itertools import product


def generate_pipelines(
    gaussian_blur_kernels: List[int],
    edge_detection_mins: List[int],
    edge_detection_maxs: List[int],
    dilate_erode_kernels: List[int],
    include_inversion: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate all valid pipeline configurations based on specified parameters.

    Args:
        gaussian_blur_kernels: List of kernel sizes for gaussian_blur (e.g., [5, 7, 9])
        edge_detection_mins: List of hysteresis_min values (e.g., [100, 125, 150])
        edge_detection_maxs: List of hysteresis_max values (e.g., [150, 175, 200, 225])
        dilate_erode_kernels: List of kernel sizes for dilate/erode pairs (e.g., [3, 5, 7, 9])
        include_inversion: Whether to include inversion in pipelines

    Returns:
        List of pipeline configurations
    """
    pipelines = []

    # Generate valid edge_detection parameter combinations
    # (hysteresis_max must be > hysteresis_min)
    valid_edge_params = [
        {"hysteresis_min": h_min, "hysteresis_max": h_max}
        for h_min in edge_detection_mins
        for h_max in edge_detection_maxs
        if h_max > h_min
    ]

    # Generate dilate+erode pairs (must use same kernel size)
    dilate_erode_pairs = [
        ({"kernel": (k, k), "iterations": 1}, {"kernel": (k, k), "iterations": 1})
        for k in dilate_erode_kernels
    ]

    # Generate inversion placement options
    # 0: no inversion
    # 1: inversion before edge_detection
    # 2: inversion after edge_detection
    inversion_placements = [0]  # Always include no inversion
    if include_inversion:
        inversion_placements.extend([1, 2])

    # Generate all combinations
    for gb_kernel in gaussian_blur_kernels:
        for edge_params in valid_edge_params:
            # Option 1: No dilate+erode
            for inversion_placement in inversion_placements:
                pipeline = [
                    {"method": "grayscale", "params": {}},
                    {"method": "gaussian_blur", "params": {"kernel": (gb_kernel, gb_kernel), "sigmaX": 0, "sigmaY": 0}},
                ]

                # Add inversion before edge_detection if placement == 1
                if inversion_placement == 1:
                    pipeline.append({"method": "inversion", "params": {}})

                # Add edge_detection
                pipeline.append({"method": "edge_detection", "params": edge_params})

                # Add inversion after edge_detection if placement == 2
                if inversion_placement == 2:
                    pipeline.append({"method": "inversion", "params": {}})

                pipelines.append(pipeline)

            # Option 2: With dilate+erode pairs
            for dilate_params, erode_params in dilate_erode_pairs:
                for inversion_placement in inversion_placements:
                    # Variant A: dilate+erode before edge_detection
                    pipeline_before = [
                        {"method": "grayscale", "params": {}},
                        {"method": "gaussian_blur", "params": {"kernel": (gb_kernel, gb_kernel), "sigmaX": 0, "sigmaY": 0}},
                        {"method": "dilate", "params": dilate_params},
                        {"method": "erode", "params": erode_params},
                    ]

                    # Add inversion before edge_detection if placement == 1
                    if inversion_placement == 1:
                        pipeline_before.append({"method": "inversion", "params": {}})

                    # Add edge_detection
                    pipeline_before.append({"method": "edge_detection", "params": edge_params})

                    # Add inversion after edge_detection if placement == 2
                    if inversion_placement == 2:
                        pipeline_before.append({"method": "inversion", "params": {}})

                    pipelines.append(pipeline_before)

                    # Variant B: dilate+erode after edge_detection
                    pipeline_after = [
                        {"method": "grayscale", "params": {}},
                        {"method": "gaussian_blur", "params": {"kernel": (gb_kernel, gb_kernel), "sigmaX": 0, "sigmaY": 0}},
                    ]

                    # Add inversion before edge_detection if placement == 1
                    if inversion_placement == 1:
                        pipeline_after.append({"method": "inversion", "params": {}})

                    # Add edge_detection
                    pipeline_after.append({"method": "edge_detection", "params": edge_params})

                    # Add inversion after edge_detection if placement == 2
                    if inversion_placement == 2:
                        pipeline_after.append({"method": "inversion", "params": {}})

                    # Add dilate+erode after edge_detection
                    pipeline_after.append({"method": "dilate", "params": dilate_params})
                    pipeline_after.append({"method": "erode", "params": erode_params})

                    pipelines.append(pipeline_after)

    return pipelines


def create_config(pipelines: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Create a list of pipeline configurations.

    Args:
        pipelines: List of pipeline configurations.

    Returns:
        List of pipeline configuration dictionaries.
    """
    preprocessing_steps = []

    for i, pipeline in enumerate(pipelines):
        # Create a descriptive name based on pipeline contents
        methods = [step["method"] for step in pipeline]
        step_config = {
            "name": f"pipeline_{i:05d}",
            "description": f"Pipeline: {' -> '.join(methods)}",
            "steps": pipeline
        }
        preprocessing_steps.append(step_config)

    return preprocessing_steps


def main():
    """Main function to parse arguments and generate configuration."""
    parser = argparse.ArgumentParser(
        description="Generate preprocessing pipeline configurations"
    )
    parser.add_argument(
        "--gaussian-blur-kernels",
        type=int,
        nargs="+",
        default=[5],
        help="Gaussian blur kernel sizes (default: 5 7 9)"
    )
    parser.add_argument(
        "--edge-detection-mins",
        type=int,
        nargs="+",
        default=[100],
        help="Edge detection hysteresis_min values (default: 100 125 150)"
    )
    parser.add_argument(
        "--edge-detection-maxs",
        type=int,
        nargs="+",
        default=[150],
        help="Edge detection hysteresis_max values (default: 150 175 200 225)"
    )
    parser.add_argument(
        "--dilate-erode-kernels",
        type=int,
        nargs="+",
        default=[3],
        help="Dilate/erode kernel sizes (default: 3 5 7 9)"
    )
    parser.add_argument(
        "--include-inversion",
        action="store_true",
        default=True,
        help="Include inversion in pipelines (default: True)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/kylelawrence/mario_kart_scores/generated_pipeline_config.json",
        help="Output file path for the configuration"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PREPROCESSING PIPELINE CONFIGURATION GENERATOR")
    print("=" * 80)
    print(f"Gaussian blur kernels: {args.gaussian_blur_kernels}")
    print(f"Edge detection hysteresis_min: {args.edge_detection_mins}")
    print(f"Edge detection hysteresis_max: {args.edge_detection_maxs}")
    print(f"Dilate/erode kernels: {args.dilate_erode_kernels}")
    print(f"Include inversion: {args.include_inversion}")
    print("=" * 80)

    print("\nGenerating pipelines...")
    pipelines = generate_pipelines(
        gaussian_blur_kernels=args.gaussian_blur_kernels,
        edge_detection_mins=args.edge_detection_mins,
        edge_detection_maxs=args.edge_detection_maxs,
        dilate_erode_kernels=args.dilate_erode_kernels,
        include_inversion=args.include_inversion,
    )

    print("\nCreating configuration...")
    config = create_config(pipelines)

    # Save configuration
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {output_path}")
    print(f"Total pipeline configurations: {len(config)}")


if __name__ == "__main__":
    main()

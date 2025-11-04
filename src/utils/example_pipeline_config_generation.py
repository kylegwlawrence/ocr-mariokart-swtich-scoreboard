#!/usr/bin/env python3
"""Example usage of the generate_pipeline_configs.py script.

This example demonstrates how to use the pipeline configuration generator
with the current set of parameters.
"""

from utils.generate_pipeline_configs import generate_pipelines, create_config
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def select_random_pipelines(
    config: List[Dict[str, Any]],
    n: int,
    seed: int = None
) -> List[Dict[str, Any]]:
    """
    Randomly select n pipeline configurations from the set.

    Args:
        config: List of pipeline configurations.
        n: Number of pipelines to randomly select.
        seed: Optional random seed for reproducibility.

    Returns:
        List of randomly selected pipeline configurations.

    Raises:
        ValueError: If n is greater than the number of available pipelines.
    """
    if n > len(config):
        raise ValueError(
            f"Cannot select {n} pipelines from a set of {len(config)} pipelines"
        )

    if seed is not None:
        random.seed(seed)

    return random.sample(config, n)


def main(parameter_combos: dict, n_random: int = 5,):
    """Generate and save pipeline configurations.

    Args:
        parameter_combos: Dictionary containing parameter combinations with keys:
            - gaussian_blur_kernels: List of kernel sizes
            - edge_detection_mins: List of minimum values
            - edge_detection_maxs: List of maximum values
            - dilate_erode_kernels: List of kernel sizes
            - include_inversion: Boolean flag
        n_random: Number of random pipelines to select for display (default: 5)
    """

    # Extract parameters from dictionary
    gaussian_blur_kernels = parameter_combos.get("gaussian_blur_kernels", [])
    edge_detection_mins = parameter_combos.get("edge_detection_mins", [])
    edge_detection_maxs = parameter_combos.get("edge_detection_maxs", [])
    dilate_erode_kernels = parameter_combos.get("dilate_erode_kernels", [])
    include_inversion = parameter_combos.get("include_inversion", False)

    print("=" * 80)
    print("EXAMPLE: GENERATING PREPROCESSING PIPELINES")
    print("=" * 80)
    print(f"Gaussian blur kernels: {gaussian_blur_kernels}")
    print(f"Edge detection hysteresis_min: {edge_detection_mins}")
    print(f"Edge detection hysteresis_max: {edge_detection_maxs}")
    print(f"Dilate/erode kernels: {dilate_erode_kernels}")
    print(f"Include inversion: {include_inversion}")
    print("=" * 80)

    # Generate pipelines
    print("\nGenerating pipelines...")
    pipelines = generate_pipelines(
        gaussian_blur_kernels=gaussian_blur_kernels,
        edge_detection_mins=edge_detection_mins,
        edge_detection_maxs=edge_detection_maxs,
        dilate_erode_kernels=dilate_erode_kernels,
        include_inversion=include_inversion,
    )
    print(f"✓ Generated {len(pipelines)} total pipelines")

    # Create configuration
    print("\nCreating configuration...")
    config = create_config(pipelines)
    print(f"✓ Created configuration with {len(config)} pipeline entries")

    # Save to file
    output_path = Path("src/config/example_gen_configs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Saved to: {output_path}")

    # Print sample pipelines
    print("\n" + "=" * 80)
    print("SAMPLE PIPELINES")
    print("=" * 80)

    # Show first 3 pipelines
    for i in range(min(3, len(config))):
        pipeline = config[i]
        print(f"\n{pipeline['name']}:")
        print(f"  Description: {pipeline['description']}")
        print("  Steps:")
        for step in pipeline["steps"]:
            print(f"    - {step['method']}: {step['params']}")

    # Show statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    # Count pipelines by structure
    no_dilate_erode = sum(1 for p in config if "dilate" not in [s["method"] for s in p["steps"]])
    with_dilate_erode = len(config) - no_dilate_erode
    with_inversion = sum(1 for p in config if "inversion" in [s["method"] for s in p["steps"]])

    print(f"Total pipelines: {len(config)}")
    print(f"  - Without dilate/erode: {no_dilate_erode}")
    print(f"  - With dilate/erode: {with_dilate_erode}")
    print(f"  - With inversion: {with_inversion}")

    # Example: Select random pipelines
    print("\n" + "=" * 80)
    print("RANDOM PIPELINE SELECTION")
    print("=" * 80)

    
    print(f"\nRandomly selecting {n_random} pipelines from {len(config)} total...")
    random_pipelines = select_random_pipelines(config, n_random, seed=42)

    print(f"✓ Selected {len(random_pipelines)} random pipelines")
    print("\nRandom selection:")
    for pipeline in random_pipelines:
        print(f"  - {pipeline['name']}: {pipeline['description']}")

    # Save random selection to file
    random_output_path = Path("src/config/example_random_selection.json")
    with open(random_output_path, "w") as f:
        json.dump(random_pipelines, f, indent=2)
    print(f"\n✓ Saved random selection to: {random_output_path}")


if __name__ == "__main__":
    # Define parameter combinations
    parameter_combos = {
        "gaussian_blur_kernels": [5, 7],
        "edge_detection_mins": [115, 135, 160],
        "edge_detection_maxs": [165, 185, 215],
        "dilate_erode_kernels": [5, 7],
        "include_inversion": True,
    }

    main(parameter_combos, 100)

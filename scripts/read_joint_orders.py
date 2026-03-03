#!/usr/bin/env python3
"""Read the joint order from a USD articulation file.

Usage::

    python3 scripts/read_joint_orders.py <path/to/robot.usd>
"""

import argparse
import sys

from pxr import Sdf


def parse_joint_names(usd_path: str) -> list[str]:
    """Extract joint names from a USD file in traversal order.

    Parses the text representation of the USD layer and collects
    every ``over "<name>_joint"`` prim, excluding fixed joints.
    """
    layer = Sdf.Layer.FindOrOpen(usd_path)
    if layer is None:
        raise FileNotFoundError(f"Cannot open USD file: {usd_path}")

    joints = []
    for line in layer.ExportToString().splitlines():
        stripped = line.strip()
        if (
            stripped.startswith('over "')
            and 'joint"' in stripped
            and "fixed" not in stripped
        ):
            joints.append(stripped.split('"')[1])
    return joints


def main() -> None:
    """Entry point: parse arguments and print the joint order."""
    parser = argparse.ArgumentParser(
        description="Read the simulation joint order from a USD file.",
    )
    parser.add_argument(
        "usd_path",
        help="Path to the robot USD file.",
    )
    args = parser.parse_args()

    joints = parse_joint_names(args.usd_path)
    if not joints:
        print("No joints found in the USD file.", file=sys.stderr)
        sys.exit(1)

    print("USD Simulation Joint Order:")
    for i, name in enumerate(joints):
        print(f"  {i:2d}: {name}")
    print(f"\nTotal: {len(joints)}")


if __name__ == "__main__":
    main()

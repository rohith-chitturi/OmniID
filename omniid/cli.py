import argparse
from omniid.sdk.client import DatasetClient
import logging

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="OmniID Data Engine CLI")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("dataset", help="Dataset operations")
    build_parser.add_argument("action", choices=["build"], help="Action to perform")
    build_parser.add_argument("dataset_path", type=str, help="Path to raw dataset")
    build_parser.add_argument("--output", type=str, default="./artifacts", help="Output directory")

    args = parser.parse_args()

    if args.command == "dataset" and args.action == "build":
        client = DatasetClient()
        manifest_client = (
            client
            .ingest(args.dataset_path)
            .validate()
            .assess_quality()
            .normalize(output_dir=f"{args.output}/normalized")
            .fingerprint()
            .generate_manifest(output_dir=args.output)
            .publish(args.output)
        )
        print(f"Dataset fingerprint: {manifest_client._fingerprint}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

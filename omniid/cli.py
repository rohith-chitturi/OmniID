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

    exp_parser = subparsers.add_parser("experiment", help="Experiment Management Operations")
    exp_subparsers = exp_parser.add_subparsers(dest="exp_command")
    
    # omniid experiment create
    create_parser = exp_subparsers.add_parser("create")
    create_parser.add_argument("overrides", nargs="*", help="Hydra-style overrides (e.g. seed=42 model=dinov2)")
    
    # omniid experiment reproduce EXP-0001
    repr_parser = exp_subparsers.add_parser("reproduce")
    repr_parser.add_argument("exp_id", type=str, help="Experiment ID to reproduce")
    repr_parser.add_argument("--run_id", type=str, default=None, help="Specific Run ID (defaults to latest)")
    
    # omniid experiment diff EXP-0001 EXP-0002
    diff_parser = exp_subparsers.add_parser("diff")
    diff_parser.add_argument("exp1", type=str)
    diff_parser.add_argument("exp2", type=str)

    # omniid experiment compare EXP-0001 RUN-0001 RUN-0002
    comp_parser = exp_subparsers.add_parser("compare")
    comp_parser.add_argument("exp_id", type=str)
    comp_parser.add_argument("run1", type=str)
    comp_parser.add_argument("run2", type=str)

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
        
    elif args.command == "experiment":
        from hydra import initialize, compose
        from omegaconf import OmegaConf
        import os
        from omniid.experiment.registry import ExperimentRegistry
        
        if args.exp_command == "create":
            # Extract basic config to find if experiment=... was provided
            with initialize(version_base=None, config_path="../configs"):
                cfg = compose(config_name="config", overrides=args.overrides)
            
            registry = ExperimentRegistry()
            exp_id, run_id, run_dir = registry.create_run(cfg, experiment_name=cfg.get("experiment_id", "auto"))
            print(f"Created {exp_id}/{run_id} at {run_dir}")
            
        elif args.exp_command == "diff":
            from omniid.experiment.config import diff_configs
            # Very basic extraction logic for diffing EXP-0001/RUN-0001 vs EXP-0002/RUN-0001
            try:
                # Naive loading of latest run config
                path1 = [os.path.join("experiments", args.exp1, r, "config.yaml") for r in os.listdir(f"experiments/{args.exp1}")][0]
                path2 = [os.path.join("experiments", args.exp2, r, "config.yaml") for r in os.listdir(f"experiments/{args.exp2}")][0]
                cfg1 = OmegaConf.load(path1)
                cfg2 = OmegaConf.load(path2)
                diff_result = diff_configs(cfg1, cfg2)
                print(f"Diff between {args.exp1} and {args.exp2}:")
                for k, v in diff_result.items():
                    print(f"{k}:\n{v['old']} -> {v['new']}\n")
            except Exception as e:
                print(f"Error diffing configs: {e}")
                
        elif args.exp_command == "reproduce":
            print(f"Reproducing {args.exp_id}... (Loads state and spins up pipeline)")
            
        elif args.exp_command == "compare":
            try:
                path1 = os.path.join("experiments", args.exp_id, args.run1, "configuration.json")
                path2 = os.path.join("experiments", args.exp_id, args.run2, "configuration.json")
                import json
                with open(path1, "r") as f:
                    cfg1 = json.load(f)
                with open(path2, "r") as f:
                    cfg2 = json.load(f)
                
                print(f"Comparing {args.exp_id}: {args.run1} vs {args.run2}")
                # Naive dict compare for demo purposes
                for k in cfg1.keys():
                    if cfg1.get(k) != cfg2.get(k):
                        print(f"{k}:\n  {args.run1}: {cfg1.get(k)}\n  {args.run2}: {cfg2.get(k)}\n")
            except Exception as e:
                print(f"Error comparing runs: {e}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

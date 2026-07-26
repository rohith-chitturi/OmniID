import argparse
import click
import json
import logging
from omniid.sdk.client import DatasetClient

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
    
    models_parser = subparsers.add_parser("models", help="Foundation Encoder Operations")
    models_subparsers = models_parser.add_subparsers(dest="models_command")
    
    # omniid models list
    list_parser = models_subparsers.add_parser("list")
    
    # omniid models info dinov2
    info_parser = models_subparsers.add_parser("info")
    info_parser.add_argument("encoder", type=str)
    
    # omniid models verify dinov2
    verify_parser = models_subparsers.add_parser("verify")
    verify_parser.add_argument("encoder", type=str)
    
    # omniid models benchmark dinov2
    bench_parser = models_subparsers.add_parser("benchmark")
    bench_parser.add_argument("encoder", type=str, help="Name of the encoder to benchmark (e.g. dinov2)")

    fusion_parser = subparsers.add_parser("fusion", help="Manage Multimodal Fusion modules")
    fusion_subparsers = fusion_parser.add_subparsers(dest="fusion_command")
    
    bench_fusion_parser = fusion_subparsers.add_parser("benchmark")
    bench_fusion_parser.add_argument("name", type=str, help="Name of fusion strategy to benchmark")

    obj_parser = subparsers.add_parser("objectives", help="Manage Optimization Objectives")
    obj_subparsers = obj_parser.add_subparsers(dest="obj_command")
    
    obj_list_parser = obj_subparsers.add_parser("list")
    obj_info_parser = obj_subparsers.add_parser("info")
    obj_info_parser.add_argument("name", type=str)

    eval_parser = subparsers.add_parser("eval", help="Manage Evaluation & Benchmark Framework")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")
    
    eval_list_parser = eval_subparsers.add_parser("list")
    
    eval_info_parser = eval_subparsers.add_parser("info")
    eval_info_parser.add_argument("task", type=str)
    
    eval_run_parser = eval_subparsers.add_parser("run")
    eval_run_parser.add_argument("task", type=str)
    
    eval_bench_parser = eval_subparsers.add_parser("benchmark")
    eval_bench_parser.add_argument("task", type=str)
    
    zoo_parser = subparsers.add_parser("zoo", help="Manage Model Zoo & Checkpoint Registry")
    zoo_subparsers = zoo_parser.add_subparsers(dest="zoo_command")
    
    zoo_list_parser = zoo_subparsers.add_parser("list")
    
    zoo_info_parser = zoo_subparsers.add_parser("info")
    zoo_info_parser.add_argument("model", type=str)
    
    zoo_verify_parser = zoo_subparsers.add_parser("verify")
    zoo_verify_parser.add_argument("model", type=str)
    
    zoo_load_parser = zoo_subparsers.add_parser("load")
    zoo_load_parser.add_argument("model", type=str)

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
                
    elif args.command == "models":
        if args.models_command == "list":
            from omniid.models.registry import BACKBONE_REGISTRY
            print("Registered Foundation Encoders:")
            for k in BACKBONE_REGISTRY._registry.keys():
                print(f"  - {k}")
                
        elif args.models_command == "info":
            from omniid.models.builder import build_encoder
            encoder = build_encoder(args.encoder)
            print(encoder.metadata.model_dump_json(indent=2))
            
        elif args.models_command == "verify":
            print(f"Verifying {args.encoder} checkpoint checksums...")
            from omniid.models.builder import build_encoder
            encoder = build_encoder(args.encoder)
            encoder.load_weights()
            print("Verification successful.")
            
        elif args.models_command == "benchmark":
            from omniid.models.benchmark import BenchmarkHarness
            import json
            try:
                harness = BenchmarkHarness()
                results = harness.run(args.encoder)
                print(f"\n--- Benchmark: {args.encoder} ---")
                print(json.dumps(results, indent=2))
            except Exception as e:
                print(f"Benchmark failed: {e}")
                
    elif args.command == "fusion":
        if args.fusion_command == "benchmark":
            from omniid.fusion.benchmark import FusionBenchmarkHarness
            import json
            try:
                print(f"Initializing Fusion Benchmark for '{args.name}'...\n")
                modality_dims = {"face": 384, "document": 768, "voice": 512}
                target_dim = 512
                harness = FusionBenchmarkHarness(args.name, target_dim=target_dim, modality_dims=modality_dims)
                results = harness.run()
                print(json.dumps(results, indent=2))
            except Exception as e:
                print(f"Error: {str(e)}")
                
    elif args.command == "objectives":
        if args.obj_command == "list":
            from omniid.objectives.registry import OBJECTIVE_REGISTRY
            # Import to trigger registration
            import omniid.objectives.builder
            print("Registered Optimization Objectives:")
            for k in OBJECTIVE_REGISTRY._registry.keys():
                print(f"  - {k}")
                
        elif args.obj_command == "info":
            from omniid.objectives.builder import build_objective
            try:
                objective = build_objective(args.name)
                print(objective.metadata.model_dump_json(indent=2))
            except Exception as e:
                print(f"Error: {str(e)}")
                
    elif args.command == "eval":
        if args.eval_command == "list":
            from omniid.eval.registry import EVALUATOR_REGISTRY
            import omniid.eval.builder
            print("Registered Evaluation Suites:")
            for k in EVALUATOR_REGISTRY._registry.keys():
                print(f"  - {k}")
                
        elif args.eval_command == "info":
            from omniid.eval.builder import build_evaluator
            try:
                evaluator = build_evaluator(args.task)
                print(evaluator.metadata.model_dump_json(indent=2))
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif args.eval_command == "run":
            from omniid.eval.builder import build_evaluator
            import torch
            try:
                print(f"Running mock evaluation for '{args.task}'...\n")
                evaluator = build_evaluator(args.task)
                
                # Generate mock data
                embeddings = torch.randn(100, 128)
                labels = torch.randint(0, 10, (100,))
                
                report = evaluator.evaluate(embeddings, labels)
                import json
                print(json.dumps({
                    "evaluator": report.evaluator_name,
                    "metrics": report.metrics,
                    "runtime_ms": report.runtime.get("total_ms", 0)
                }, indent=2))
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif args.eval_command == "benchmark":
            print(f"Benchmarking (System performance) for {args.task} is not fully implemented in CLI yet.")
            
    elif args.command == "zoo":
        from omniid.model_zoo.api import ModelZoo
        zoo = ModelZoo()
        
        if args.zoo_command == "list":
            print("Discovered Models:")
            for m in zoo.list_models():
                print(f"  - {m}")
                
        elif args.zoo_command == "info":
            try:
                card = zoo.get_model_card(args.model)
                print(card.model_dump_json(indent=2))
            except Exception as e:
                print(f"Error fetching model info: {str(e)}")
                
        elif args.zoo_command == "verify":
            try:
                # Mock current execution config
                import omniid
                current_config = {
                    "encoder": "dinov2",
                    "fusion": "cross_attention"
                }
                manifest = zoo.get_model_info(args.model)
                is_compatible, msg = zoo.validator.validate(manifest, current_config)
                print(f"Verification Result: {'PASS' if is_compatible else 'FAIL'}")
                print(f"Details: {msg}")
            except Exception as e:
                print(f"Verification Error: {str(e)}")
                
        elif args.zoo_command == "load":
            print(f"Mock loading {args.model} checkpoint via Zoo API...")
            try:
                # Mock current execution config
                current_config = {
                    "encoder": "dinov2",
                    "fusion": "cross_attention"
                }
                path = zoo.load(args.model, current_config)
                print(f"Successfully resolved checkpoint to: {path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

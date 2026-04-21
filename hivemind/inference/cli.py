"""
Command-Line Interface for Decentralized Inference

Provides simple, user-friendly CLI commands for:
- Contributing GPU/CPU resources to the network
- Running large models using distributed compute
- Monitoring contributions and rewards
"""

import argparse
import sys
import time
import threading
from typing import Optional, List
from pathlib import Path

import torch

from hivemind.dht import DHT
from hivemind.p2p import P2P
from hivemind.utils import get_logger
from hivemind.inference.pipeline import ModelChunkProvider, ModelChunkConfig, PipelineParallelRunner
from hivemind.inference.contribution import ContributionTracker, TokenRewardCalculator

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the inference CLI"""
    parser = argparse.ArgumentParser(
        prog="hivemind-inference",
        description="Decentralized AI Inference - Share GPUs, Run Large Models"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Command: contribute
    contribute_parser = subparsers.add_parser(
        "contribute",
        help="Contribute your GPU/CPU to run large model chunks"
    )
    contribute_parser.add_argument(
        "--model", 
        type=str, 
        default="kimi-k2.6",
        help="Model name to contribute to (default: kimi-k2.6)"
    )
    contribute_parser.add_argument(
        "--layers",
        type=str,
        default="0-4",
        help="Layer range to host, e.g., '0-4' (default: 0-4)"
    )
    contribute_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    contribute_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Hidden dimension size (default: 4096)"
    )
    contribute_parser.add_argument(
        "--initial-peers",
        type=str,
        nargs="+",
        default=[],
        help="Multiaddrs of initial DHT peers to connect to"
    )
    contribute_parser.add_argument(
        "--track-rewards",
        action="store_true",
        help="Enable contribution tracking for token rewards"
    )
    
    # Command: run
    run_parser = subparsers.add_parser(
        "run",
        help="Run a large model using distributed compute"
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default="kimi-k2.6",
        help="Model to run (default: kimi-k2.6)"
    )
    run_parser.add_argument(
        "--prompt",
        type=str,
        default="What is artificial intelligence?",
        help="Input prompt (default: 'What is artificial intelligence?')"
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    run_parser.add_argument(
        "--initial-peers",
        type=str,
        nargs="+",
        default=[],
        help="Multiaddrs of initial DHT peers to connect to"
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed pipeline information"
    )
    
    # Command: status
    status_parser = subparsers.add_parser(
        "status",
        help="Check your contributions and rewards"
    )
    status_parser.add_argument(
        "--model",
        type=str,
        help="Filter by model name"
    )
    status_parser.add_argument(
        "--rewards",
        action="store_true",
        help="Show reward history"
    )
    
    # Command: discover
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover available models and compute resources"
    )
    discover_parser.add_argument(
        "--model",
        type=str,
        help="Specific model to check"
    )
    discover_parser.add_argument(
        "--coverage",
        action="store_true",
        help="Show layer coverage report"
    )
    
    return parser


def cmd_contribute(args):
    """Handle the 'contribute' command"""
    print(f"🚀 Starting GPU contribution for {args.model}")
    print(f"   Device: {args.device}")
    print(f"   Layers: {args.layers}")
    
    # Parse layer range
    try:
        layer_start, layer_end = map(int, args.layers.split("-"))
    except ValueError:
        print("❌ Invalid layer format. Use 'start-end', e.g., '0-4'")
        return 1
    
    # Initialize DHT
    print("📡 Connecting to DHT network...")
    dht = DHT(initial_peers=args.initial_peers, start=True)
    
    visible_maddrs = [str(a) for a in dht.get_visible_maddrs()]
    print(f"✅ Connected! Visible at: {visible_maddrs[0] if visible_maddrs else 'local'}")
    
    # Create model chunk provider
    config = ModelChunkConfig(
        model_name=args.model,
        layer_start=layer_start,
        layer_end=layer_end,
        device=args.device,
        hidden_dim=args.hidden_dim
    )
    
    provider = ModelChunkProvider(dht, config, initial_peers=args.initial_peers)
    
    # Initialize contribution tracker if requested
    tracker = None
    if args.track_rewards:
        print("📊 Enabling contribution tracking...")
        tracker = ContributionTracker(dht)
        tracker.run_in_background()
    
    # Advertise to network
    print(f"📢 Advertising layers {layer_start}-{layer_end} to network...")
    provider.advertise()
    
    print("\n✅ You are now contributing to the decentralized inference network!")
    print("   Press Ctrl+C to stop\n")
    
    # Keep running
    try:
        while True:
            time.sleep(10)
            
            if tracker:
                stats = provider.get_stats()
                print(f"\r⏱️  Compute time: {stats['total_compute_time']:.2f}s | "
                      f"Tokens processed: {stats['tokens_processed']}", end="", flush=True)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        
        if tracker:
            contributions = tracker.get_my_total_contributions()
            if contributions:
                print(f"\n📊 Your total contributions:")
                for model, units in contributions.items():
                    print(f"   {model}: {units:.2f} compute units")
        
        dht.shutdown()
        print("✅ Shutdown complete. Thank you for contributing!")
    
    return 0


def cmd_run(args):
    """Handle the 'run' command"""
    import asyncio
    
    print(f"🤖 Running {args.model} on distributed compute")
    print(f"   Prompt: '{args.prompt}'")
    print(f"   Max tokens: {args.max_tokens}")
    
    # Initialize DHT
    print("📡 Connecting to DHT network...")
    dht = DHT(initial_peers=args.initial_peers, start=True)
    
    visible_maddrs = [str(a) for a in dht.get_visible_maddrs()]
    print(f"✅ Connected! Visible at: {visible_maddrs[0] if visible_maddrs else 'local'}")
    
    # Create pipeline runner
    runner = PipelineParallelRunner(dht, args.model)
    
    async def run_inference():
        # Discover topology
        print("🔍 Discovering available compute resources...")
        try:
            topology = await runner.discover_topology()
            
            if not topology:
                print("❌ No compute resources found for this model.")
                print("   Try running 'hivemind-inference contribute' first,")
                print("   or wait for others to contribute resources.")
                return 1
            
            print(f"✅ Found {len(topology)} model chunks from "
                  f"{len(set(str(e.peer_id) for e in topology))} peers")
            
            if args.verbose:
                stats = runner.get_pipeline_stats()
                print(f"\n📊 Pipeline Statistics:")
                print(f"   Total layers: {stats['total_layers']}")
                print(f"   Unique peers: {stats['unique_peers']}")
            
            # Run inference
            print("\n⚙️  Running inference...")
            start_time = time.time()
            
            result = await runner.generate(args.prompt, max_tokens=args.max_tokens)
            
            elapsed = time.time() - start_time
            
            print(f"\n✅ Generated in {elapsed:.2f}s:\n")
            print("=" * 60)
            print(result)
            print("=" * 60)
            
            return 0
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            logger.exception("Inference failed")
            return 1
        
        finally:
            dht.shutdown()
    
    # Run async inference
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(run_inference())


def cmd_status(args):
    """Handle the 'status' command"""
    print("📊 Checking contribution status...")
    
    # This would query the DHT for contribution records
    # For now, show placeholder
    print("\nℹ️  Note: Full status tracking requires active contribution session.")
    print("   Start contributing with 'hivemind-inference contribute'")
    
    return 0


def cmd_discover(args):
    """Handle the 'discover' command"""
    import asyncio
    from hivemind.inference.discovery import LayerDiscoveryProtocol, ResourceRegistry
    
    print("🔍 Discovering available models and resources...")
    
    dht = DHT(start=True)
    
    async def discover():
        if args.model:
            protocol = LayerDiscoveryProtocol(dht, args.model)
            
            if args.coverage:
                registry = ResourceRegistry(dht)
                await registry.discover_resources(args.model)
                coverage = registry.get_layer_coverage(args.model)
                
                print(f"\n📊 Coverage Report for {args.model}:")
                print(f"   Coverage: {coverage['coverage_percentage']:.1f}%")
                print(f"   Providers: {coverage['num_providers']}")
                
                if coverage['gaps']:
                    print(f"   Gaps (first 10): {coverage['gaps']}")
                else:
                    print("   ✅ All layers covered!")
            else:
                topology = await protocol.assemble_pipeline()
                print(f"\n✅ Found {len(topology)} chunks for {args.model}")
        else:
            # Discover all models
            registry = ResourceRegistry(dht)
            # Would need to implement scanning for all models
            print("\nℹ️  No specific model requested.")
            print("   Use --model <name> to check a specific model")
            print("   Use --coverage to see layer coverage")
        
        dht.shutdown()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(discover())
    
    return 0


def run_inference_cli(argv: Optional[List[str]] = None):
    """Main entry point for the inference CLI"""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    commands = {
        "contribute": cmd_contribute,
        "run": cmd_run,
        "status": cmd_status,
        "discover": cmd_discover
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(run_inference_cli())

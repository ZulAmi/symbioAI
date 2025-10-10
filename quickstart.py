#!/usr/bin/env python3
"""
Symbio AI - Quick Start CLI
Run all major demos and check system health
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show results."""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode == 0

def main():
    """Main CLI entrypoint."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                     SYMBIO AI - QUICK START                       ║
║                  Production-Ready Modular AI System               ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    demos = [
        ("python examples/theorem_proving_demo.py", "Automated Theorem Proving Demo"),
        ("python examples/recursive_self_improvement_demo.py", "Recursive Self-Improvement"),
        ("python examples/cross_task_transfer_demo.py", "Cross-Task Transfer Learning"),
        ("python examples/metacognitive_causal_demo.py", "Metacognitive Monitoring & Causal Diagnosis"),
        ("python examples/neural_symbolic_demo.py", "Neural-Symbolic Architecture"),
        ("python examples/compositional_concept_demo.py", "Compositional Concept Learning"),
        ("python examples/dynamic_architecture_demo.py", "Dynamic Neural Architecture Evolution"),
        ("python examples/memory_enhanced_moe_demo.py", "Memory-Enhanced Mixture of Experts"),
        ("python examples/multi_scale_temporal_demo.py", "Multi-Scale Temporal Reasoning"),
        ("python examples/unified_multimodal_demo.py", "Unified Multi-Modal Foundation"),
        ("python examples/embodied_ai_demo.py", "Embodied AI Simulation"),
        ("python examples/multi_agent_collaboration_demo.py", "Multi-Agent Collaboration Networks"),
        ("python examples/continual_learning_demo.py", "Continual Learning Without Catastrophic Forgetting"),
    ]
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        
        if choice == "all":
            print("\n🎯 Running ALL demos (this may take a few minutes)...\n")
            success_count = 0
            for cmd, desc in demos:
                if run_command(cmd, desc):
                    success_count += 1
            
            print(f"\n{'='*70}")
            print(f"✅ Completed {success_count}/{len(demos)} demos successfully")
            print(f"{'='*70}\n")
        
        elif choice == "system":
            run_command("python main.py", "Full System Orchestrator")
        
        elif choice == "api":
            print("\n🌐 Starting API Gateway (Service Mesh)...")
            print("Visit http://127.0.0.1:8080/admin/metrics when ready\n")
            run_command("""python - << 'PY'
import asyncio
from api.gateway import ServiceMesh
asyncio.run(ServiceMesh({}).run(host='127.0.0.1', port=8080))
PY""", "API Gateway")
        
        elif choice == "test":
            run_command("pytest -v", "Test Suite")
        
        elif choice == "health":
            print("\n🏥 System Health Check")
            print("="*70)
            
            # Check Python version
            result = subprocess.run(["python", "--version"], capture_output=True, text=True)
            print(f"✓ Python: {result.stdout.strip()}")
            
            # Check key imports
            try:
                import torch
                print(f"✓ PyTorch: {torch.__version__}")
            except:
                print("✗ PyTorch: Not installed")
            
            try:
                import transformers
                print(f"✓ Transformers: {transformers.__version__}")
            except:
                print("✗ Transformers: Not installed")
            
            try:
                import z3
                print("✓ Z3 Solver: Available")
            except:
                print("✗ Z3 Solver: Not installed")
            
            # Check theorem provers
            print("\n🔧 Theorem Prover Availability:")
            subprocess.run(["python", "-c", """
from training.automated_theorem_proving import create_theorem_prover
prover = create_theorem_prover()
print(f"  Z3: {'✓ Available' if prover.z3_prover.is_available() else '✗ Not installed'}")
print(f"  Lean: {'✓ Available' if prover.lean_prover.is_available() else '✗ Not installed'}")
print(f"  Coq: {'✓ Available' if prover.coq_prover.is_available() else '✗ Not installed'}")
"""])
            
        else:
            print(f"Unknown command: {choice}")
            print_usage()
    else:
        print_usage()

def print_usage():
    """Print usage information."""
    print("""
📋 Usage: python quickstart.py [command]

Commands:
  all      - Run all demos (recommended for first time)
  system   - Run full system orchestrator
  api      - Start API gateway/service mesh
  test     - Run test suite
  health   - Check system health and dependencies

Individual Demos:
  python examples/theorem_proving_demo.py              # Formal verification
  python examples/recursive_self_improvement_demo.py    # Meta-evolution
  python examples/cross_task_transfer_demo.py          # Transfer learning
  python examples/metacognitive_causal_demo.py         # Self-diagnosis
  python examples/neural_symbolic_demo.py              # Neural-symbolic AI
  python examples/compositional_concept_demo.py        # Concept learning
  python examples/dynamic_architecture_demo.py         # Dynamic architecture evolution
  python examples/memory_enhanced_moe_demo.py          # Memory-enhanced MoE
  python examples/multi_scale_temporal_demo.py         # Multi-scale temporal reasoning
  python examples/unified_multimodal_demo.py           # Unified multi-modal foundation

Examples:
  python quickstart.py all       # Run all demos
  python quickstart.py health    # Check system health
  python quickstart.py api       # Start API server

Documentation:
  QUICKSTART.md                          # Complete setup guide
  docs/automated_theorem_proving.md      # Theorem proving docs
  AUTOMATED_THEOREM_PROVING_COMPLETE.md  # Completion report

For help: python quickstart.py
    """)

if __name__ == "__main__":
    main()

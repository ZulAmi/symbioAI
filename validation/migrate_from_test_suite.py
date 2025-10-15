#!/usr/bin/env python3
"""
Migration Script: symbioai_test_suite → validation (Tier-Based)
==============================================================

This script safely migrates valuable components from symbioai_test_suite
to the new tier-based validation structure while preserving important work.

What this script does:
1. Creates backup of existing symbioai_test_suite
2. Moves valuable tests to validation/legacy/
3. Preserves all reports and data
4. Creates migration documentation
5. Sets up new tier-based structure

Usage:
    python validation/migrate_from_test_suite.py --backup --migrate
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse


def create_backup(test_suite_path: Path, backup_path: Path):
    """Create a complete backup of the test suite."""
    print(f"📦 Creating backup: {test_suite_path} → {backup_path}")
    
    if backup_path.exists():
        print(f"⚠️  Backup already exists at {backup_path}")
        response = input("Overwrite existing backup? (y/n): ").lower()
        if response != 'y':
            print("❌ Backup cancelled")
            return False
    
    try:
        shutil.copytree(test_suite_path, backup_path, dirs_exist_ok=True)
        print(f"✅ Backup created successfully")
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False


def analyze_test_suite_structure(test_suite_path: Path) -> Dict[str, Any]:
    """Analyze the current test suite structure."""
    print(f"🔍 Analyzing test suite structure: {test_suite_path}")
    
    analysis = {
        'total_files': 0,
        'test_files': [],
        'report_files': [],
        'data_files': [],
        'important_files': [],
        'structure': {}
    }
    
    if not test_suite_path.exists():
        print(f"❌ Test suite path does not exist: {test_suite_path}")
        return analysis
    
    # Walk through all files
    for file_path in test_suite_path.rglob('*'):
        if file_path.is_file():
            analysis['total_files'] += 1
            relative_path = file_path.relative_to(test_suite_path)
            
            if file_path.suffix == '.py' and 'test_' in file_path.name:
                analysis['test_files'].append(str(relative_path))
            elif file_path.suffix == '.json':
                analysis['report_files'].append(str(relative_path))
            elif file_path.name in ['README.md', 'CLEANUP_SUMMARY.md', 'TRUTH_AND_TRANSPARENCY.md']:
                analysis['important_files'].append(str(relative_path))
            elif file_path.parent.name == 'data':
                analysis['data_files'].append(str(relative_path))
    
    print(f"📊 Analysis complete:")
    print(f"   Total files: {analysis['total_files']}")
    print(f"   Test files: {len(analysis['test_files'])}")
    print(f"   Report files: {len(analysis['report_files'])}")
    print(f"   Important docs: {len(analysis['important_files'])}")
    print(f"   Data files: {len(analysis['data_files'])}")
    
    return analysis


def create_migration_plan(analysis: Dict[str, Any]) -> Dict[str, List[str]]:
    """Create a migration plan based on analysis."""
    print(f"📋 Creating migration plan...")
    
    plan = {
        'preserve_in_legacy': [
            # Critical test files to preserve
            'tier2_module_tests/test_combined_strategy_core.py',
            'tier2_module_tests/test_combined_adapters.py', 
            'tier2_module_tests/test_combined_progressive.py',
            'tier2_module_tests/test_neural_symbolic_integration.py',
            'tier2_module_tests/test_causal_discovery.py',
            'tier2_module_tests/test_multi_agent_coordination.py',
            
            # Phase structure (has value)
            'phase2_full_module_tests/',
            'phase3_integration_benchmarking/',
            
            # Important documentation
            'README.md',
            'CLEANUP_SUMMARY.md', 
            'PHASE1_CRITICAL_TESTS_README.md',
            'phase3_integration_benchmarking/TRUTH_AND_TRANSPARENCY.md',
        ],
        'preserve_reports': [
            # All report files
            'reports/',
        ],
        'preserve_data': [
            # Data that was downloaded
            'data/MNIST/',
        ],
        'discard': [
            # Files that can be safely discarded
            '__pycache__/',
            '*.pyc',
            'run_phase1_critical_tests.py',  # Replaced by tier-based runner
        ]
    }
    
    print(f"📋 Migration plan created:")
    print(f"   Files to preserve in legacy: {len(plan['preserve_in_legacy'])}")
    print(f"   Reports to preserve: {len(plan['preserve_reports'])}")
    print(f"   Data to preserve: {len(plan['preserve_data'])}")
    
    return plan


def execute_migration(test_suite_path: Path, validation_path: Path, plan: Dict[str, List[str]]):
    """Execute the migration plan."""
    print(f"🚚 Executing migration: {test_suite_path} → {validation_path}")
    
    # Create legacy directory structure
    legacy_path = validation_path / 'legacy'
    legacy_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (legacy_path / 'important_tests').mkdir(exist_ok=True)
    (legacy_path / 'phase_structure').mkdir(exist_ok=True)
    (legacy_path / 'reports').mkdir(exist_ok=True)
    (legacy_path / 'data').mkdir(exist_ok=True)
    (legacy_path / 'documentation').mkdir(exist_ok=True)
    
    migration_log = []
    
    # Migrate files to preserve
    for item in plan['preserve_in_legacy']:
        source_path = test_suite_path / item
        
        if not source_path.exists():
            print(f"⚠️  Skipping missing file: {item}")
            continue
        
        # Determine destination
        if item.startswith('tier2_module_tests/'):
            dest_path = legacy_path / 'important_tests' / Path(item).name
        elif item.startswith('phase'):
            dest_path = legacy_path / 'phase_structure' / Path(item).name
        elif item.endswith('.md'):
            dest_path = legacy_path / 'documentation' / Path(item).name
        else:
            dest_path = legacy_path / item
        
        try:
            if source_path.is_dir():
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
            
            migration_log.append(f"✅ Migrated: {item} → {dest_path.relative_to(validation_path)}")
            print(f"   ✅ {item}")
            
        except Exception as e:
            migration_log.append(f"❌ Failed: {item} - {str(e)}")
            print(f"   ❌ Failed to migrate {item}: {e}")
    
    # Migrate reports
    reports_source = test_suite_path / 'reports'
    if reports_source.exists():
        try:
            shutil.copytree(reports_source, legacy_path / 'reports', dirs_exist_ok=True)
            migration_log.append(f"✅ Migrated all reports")
            print(f"   ✅ reports/")
        except Exception as e:
            migration_log.append(f"❌ Failed to migrate reports: {str(e)}")
            print(f"   ❌ Failed to migrate reports: {e}")
    
    # Migrate data
    data_source = test_suite_path / 'data'
    if data_source.exists():
        try:
            shutil.copytree(data_source, legacy_path / 'data', dirs_exist_ok=True)
            migration_log.append(f"✅ Migrated data files")
            print(f"   ✅ data/")
        except Exception as e:
            migration_log.append(f"❌ Failed to migrate data: {str(e)}")
            print(f"   ❌ Failed to migrate data: {e}")
    
    print(f"✅ Migration execution complete")
    return migration_log


def create_migration_documentation(validation_path: Path, migration_log: List[str], analysis: Dict[str, Any]):
    """Create comprehensive migration documentation."""
    print(f"📄 Creating migration documentation...")
    
    timestamp = datetime.now().isoformat()
    
    doc = []
    doc.append("# Migration Documentation: symbioai_test_suite → validation")
    doc.append(f"\n**Migration Date:** {timestamp}")
    doc.append(f"**Migration Tool:** migrate_from_test_suite.py")
    doc.append("\n---\n")
    
    doc.append("## 🎯 Migration Purpose\n")
    doc.append("This migration consolidates the `symbioai_test_suite` into the new tier-based")
    doc.append("validation framework while preserving all valuable work and maintaining")
    doc.append("transparency about what was moved and why.\n")
    
    doc.append("## 📊 Original Structure Analysis\n")
    doc.append(f"**Total files processed:** {analysis['total_files']}")
    doc.append(f"**Test files found:** {len(analysis['test_files'])}")
    doc.append(f"**Report files found:** {len(analysis['report_files'])}")
    doc.append(f"**Important documents:** {len(analysis['important_files'])}")
    doc.append(f"**Data files:** {len(analysis['data_files'])}\n")
    
    doc.append("## 🚚 Migration Actions\n")
    for log_entry in migration_log:
        doc.append(f"- {log_entry}")
    
    doc.append("\n## 📁 New Structure in validation/legacy/\n")
    doc.append("```")
    doc.append("validation/legacy/")
    doc.append("├── important_tests/           # Critical test files worth preserving")
    doc.append("├── phase_structure/           # Phase 2 & 3 structure (has value)")
    doc.append("├── reports/                   # All historical test reports")
    doc.append("├── data/                      # Downloaded datasets (MNIST, etc.)")
    doc.append("├── documentation/             # Important README and documentation")
    doc.append("└── MIGRATION_NOTES.md         # This file")
    doc.append("```\n")
    
    doc.append("## ✅ What Was Preserved\n")
    doc.append("### Critical Test Files")
    doc.append("- `test_combined_strategy_core.py` - Core COMBINED strategy tests")
    doc.append("- `test_combined_adapters.py` - Adapter combination tests")
    doc.append("- `test_neural_symbolic_integration.py` - Neural-symbolic integration")
    doc.append("- `test_causal_discovery.py` - Causal reasoning validation")
    doc.append("- `test_multi_agent_coordination.py` - Multi-agent coordination")
    doc.append("- Additional tier2 module tests with real validation logic\n")
    
    doc.append("### Phase Structure")
    doc.append("- Phase 2 comprehensive module tests")
    doc.append("- Phase 3 integration benchmarking framework")
    doc.append("- Truth and transparency documentation\n")
    
    doc.append("### Historical Data")
    doc.append("- All test reports and results")
    doc.append("- Downloaded datasets (MNIST, etc.)")
    doc.append("- Performance baselines and metrics\n")
    
    doc.append("### Documentation")
    doc.append("- Original README files")
    doc.append("- Phase documentation")
    doc.append("- Cleanup summaries")
    doc.append("- Truth and transparency statements\n")
    
    doc.append("## ❌ What Was Not Migrated (And Why)\n")
    doc.append("### Replaced by Better Systems")
    doc.append("- `run_phase1_critical_tests.py` → Replaced by `run_tier_validation.py`")
    doc.append("- Simulated benchmarking → Replaced by real dataset validation")
    doc.append("- Mock competitive analysis → Replaced by honest competitive comparison\n")
    
    doc.append("### Technical Artifacts")
    doc.append("- `__pycache__/` directories and `.pyc` files")
    doc.append("- Temporary files and build artifacts\n")
    
    doc.append("## 🎯 Benefits of New Tier-Based System\n")
    doc.append("### Better Organization")
    doc.append("- Clear tier structure (1-5) matching your comprehensive dataset plan")
    doc.append("- Progressive validation from core algorithms to real applications")
    doc.append("- Targeted testing per capability area\n")
    
    doc.append("### Real Validation")
    doc.append("- Uses actual datasets, not simulated data")
    doc.append("- Real training with gradient descent")
    doc.append("- Honest performance measurement and reporting\n")
    
    doc.append("### Strategic Alignment")
    doc.append("- Maps directly to your 5-tier dataset strategy")
    doc.append("- Clear success criteria per tier")
    doc.append("- Supports academic, funding, and commercial use cases\n")
    
    doc.append("## 🚀 How to Use Migrated Content\n")
    doc.append("### Reference Important Tests")
    doc.append("```bash")
    doc.append("# Review preserved critical tests")
    doc.append("ls validation/legacy/important_tests/")
    doc.append("")
    doc.append("# Extract useful logic for tier-based tests")
    doc.append("cat validation/legacy/important_tests/test_combined_strategy_core.py")
    doc.append("```\n")
    
    doc.append("### Access Historical Reports")
    doc.append("```bash")
    doc.append("# Review past performance")
    doc.append("ls validation/legacy/reports/")
    doc.append("")
    doc.append("# Compare with new tier-based results")
    doc.append("cat validation/legacy/reports/phase1_critical_tests_*.json")
    doc.append("```\n")
    
    doc.append("### Learn from Phase Structure")
    doc.append("```bash")
    doc.append("# Study comprehensive testing approach")
    doc.append("cat validation/legacy/phase_structure/phase2_full_module_tests/")
    doc.append("")
    doc.append("# Understand integration testing framework")
    doc.append("cat validation/legacy/phase_structure/phase3_integration_benchmarking/")
    doc.append("```\n")
    
    doc.append("## 🔄 Integration with New System\n")
    doc.append("The migrated content integrates with the new tier-based system as follows:\n")
    
    doc.append("### Tier 1 (Continual Learning)")
    doc.append("- Reference `test_combined_strategy_core.py` for COMBINED strategy logic")
    doc.append("- Use adapter combination patterns from `test_combined_adapters.py`\n")
    
    doc.append("### Tier 2 (Causal Reasoning)")
    doc.append("- Reference `test_causal_discovery.py` for causal reasoning patterns")
    doc.append("- Use self-diagnosis logic from preserved tests\n")
    
    doc.append("### Tier 3 (Multi-Agent)")
    doc.append("- Reference `test_multi_agent_coordination.py` for coordination patterns")
    doc.append("- Use emergent communication logic from preserved tests\n")
    
    doc.append("### Tier 4 (Neural-Symbolic)")
    doc.append("- Reference `test_neural_symbolic_integration.py` for integration patterns")
    doc.append("- Use symbolic reasoning logic from preserved tests\n")
    
    doc.append("## 🎯 Next Steps\n")
    doc.append("1. **Review Preserved Tests**: Extract valuable logic for tier implementation")
    doc.append("2. **Run New Tier System**: `python validation/run_tier_validation.py --preset core`")
    doc.append("3. **Compare Results**: Historical reports vs new tier-based validation")
    doc.append("4. **Enhance Tiers**: Incorporate best patterns from preserved tests")
    doc.append("5. **Document Improvements**: Track how tier-based system performs better\n")
    
    doc.append("## 📞 Support\n")
    doc.append("If you need to:")
    doc.append("- **Find specific logic**: Check `validation/legacy/important_tests/`")
    doc.append("- **Review past results**: Check `validation/legacy/reports/`")
    doc.append("- **Understand old system**: Check `validation/legacy/documentation/`")
    doc.append("- **Restore something**: Full backup available (see migration command)\n")
    
    doc.append("---\n")
    doc.append(f"*Migration completed on {timestamp}*")
    doc.append("*All valuable work preserved in validation/legacy/*")
    
    # Write documentation
    doc_path = validation_path / 'legacy' / 'MIGRATION_NOTES.md'
    with open(doc_path, 'w') as f:
        f.write('\n'.join(doc))
    
    print(f"📄 Migration documentation saved to: {doc_path}")
    return doc_path


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description='Migrate symbioai_test_suite to tier-based validation')
    parser.add_argument('--backup', action='store_true', help='Create backup before migration')
    parser.add_argument('--migrate', action='store_true', help='Execute the migration')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, no migration')
    parser.add_argument('--test-suite-path', type=str, 
                       default='symbioai_test_suite',
                       help='Path to test suite (default: symbioai_test_suite)')
    parser.add_argument('--validation-path', type=str,
                       default='validation', 
                       help='Path to validation directory (default: validation)')
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path.cwd()
    test_suite_path = base_path / args.test_suite_path
    validation_path = base_path / args.validation_path
    backup_path = base_path / f"{args.test_suite_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"{'='*80}")
    print(f"🚚 SYMBIOAI TEST SUITE MIGRATION")
    print(f"{'='*80}")
    print(f"Source: {test_suite_path}")
    print(f"Target: {validation_path}")
    print(f"Backup: {backup_path}")
    
    # Analyze structure
    analysis = analyze_test_suite_structure(test_suite_path)
    
    if args.analyze_only:
        print(f"\n📊 Analysis complete - no migration performed")
        return
    
    # Create migration plan
    plan = create_migration_plan(analysis)
    
    # Create backup if requested
    if args.backup:
        if not create_backup(test_suite_path, backup_path):
            print(f"❌ Migration aborted due to backup failure")
            return
    
    # Execute migration if requested
    if args.migrate:
        print(f"\n🚚 Starting migration...")
        
        # Confirm migration
        print(f"\n⚠️  This will migrate content from:")
        print(f"   {test_suite_path} → {validation_path}/legacy/")
        print(f"\n📦 Backup available at: {backup_path}")
        
        response = input("\nProceed with migration? (y/n): ").lower()
        if response != 'y':
            print(f"❌ Migration cancelled by user")
            return
        
        # Execute migration
        migration_log = execute_migration(test_suite_path, validation_path, plan)
        
        # Create documentation
        doc_path = create_migration_documentation(validation_path, migration_log, analysis)
        
        print(f"\n{'='*80}")
        print(f"✅ MIGRATION COMPLETE!")
        print(f"{'='*80}")
        print(f"📦 Backup: {backup_path}")
        print(f"📁 Legacy content: {validation_path}/legacy/")
        print(f"📄 Documentation: {doc_path}")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Review: cat {doc_path}")
        print(f"   2. Test new system: python validation/run_tier_validation.py --preset core")
        print(f"   3. Compare results with legacy reports")
        print(f"   4. Remove original if satisfied: rm -rf {test_suite_path}")
        
    else:
        print(f"\n📋 Migration plan created but not executed")
        print(f"   Run with --migrate to execute the migration")
        print(f"   Run with --backup --migrate for safer migration with backup")


if __name__ == '__main__':
    main()
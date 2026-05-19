#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Initialize environment
from config.env_manager import init_env
if not init_env(require_edit=False):
    print("❌ Environment init failed")
    sys.exit(1)

import yaml

# Load and run single source
sources = yaml.safe_load((ROOT/"config/sources.yaml").read_text())["sources"]
enabled = [s for s in sources if s.get("enabled", True)]

if not enabled:
    print("❌ No enabled sources")
    sys.exit(1)

source = enabled[0]
print(f"\n▶️  Running: {source['id']}")
print(f"   Path: {source['path']}")
print(f"   Port: {source.get('port', 8081)}")
print(f"   Location: {source.get('location', '?')}")
print()

from main import run_source

try:
    run_source(source)
except KeyboardInterrupt:
    print("\n✅ Stopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

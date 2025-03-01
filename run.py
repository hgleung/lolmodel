#!/usr/bin/env python3
import os
import sys
import subprocess

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <script_name> [args...]")
        sys.exit(1)
        
    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    
    # Automatically append .py if not present
    if not script_path.endswith('.py'):
        script_path += '.py'
    
    # Set PYTHONPATH to include the project root
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    
    # Run the script as a separate process
    result = subprocess.run(
        [sys.executable, script_path] + script_args,
        env=env,
        cwd=project_root
    )
    sys.exit(result.returncode)

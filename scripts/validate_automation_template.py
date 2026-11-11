"""Validate only the fixed runner template and a trusted no-op fixture, never LLM output."""
import json
from pathlib import Path
import shutil
import subprocess

from src.automation.workspace import PACKAGE, WORLD, INSPECTOR


def main():
    root = Path('.local/template-validation').resolve()
    files = {
        'package.json': json.dumps(PACKAGE, indent=2),
        'support/world.ts': WORLD,
        'inspect.cjs': INSPECTOR,
        'tsconfig.json': json.dumps({'compilerOptions': {'target': 'ES2022', 'module': 'commonjs',
            'strict': True, 'esModuleInterop': True, 'skipLibCheck': True, 'types': ['node']},
            'include': ['**/*.ts'], 'exclude': ['node_modules']}),
        'features/template.feature': 'Feature: Trusted template validation\n  Scenario: Steps are wired\n    Given the template is ready\n',
        'steps/template.ts': "import { Given } from '@cucumber/cucumber';\nGiven('the template is ready', function () {});\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    (root / 'reports').mkdir(exist_ok=True)
    npm = shutil.which('npm.cmd') or shutil.which('npm')
    node = shutil.which('node')
    if not npm or not node:
        raise SystemExit('Node.js and npm are required')
    for command in ([npm, 'install', '--ignore-scripts', '--no-audit', '--no-fund'],
                    [node, '--check', 'inspect.cjs'], [npm, 'run', 'check'], [npm, 'run', 'validate']):
        subprocess.run(command, cwd=root, check=True, timeout=300)
    print('Trusted TypeScript, inspector syntax and Cucumber dry-run passed; no browser test executed.')


if __name__ == '__main__':
    main()

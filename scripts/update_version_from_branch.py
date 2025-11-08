import re
import subprocess
from pathlib import Path


def get_git_branch():
    result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()

def find_about_py():
    for path in Path(".").rglob("__about__.py"):
        return path
    raise FileNotFoundError("Could not find __about__.py")

def update_about_py(version):
    about_path = find_about_py()
    content = about_path.read_text()
    new_content = re.sub(r'__version__\s*=\s*["\'].*?["\']', f'__version__ = "{version}"', content)
    about_path.write_text(new_content)
    print(f"✅ Updated version to v{version} in {about_path}")

def main():
    branch = get_git_branch()
    match = re.match(r"(hotfix|release)/(\d+\.\d+\.\d+)", branch)
    if match:
        version = match.group(2)
        update_about_py(version)
    else:
        print("ℹ️ Not a hotfix/release branch. Skipping version update.")

if __name__ == "__main__":
    main()

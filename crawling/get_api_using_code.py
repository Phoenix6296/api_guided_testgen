import os
import re
import argparse
import shutil
import logging
from git import Repo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Library patterns with common import aliases
LIBRARY_PATTERNS = {
    'tensorflow': re.compile(r'^\s*(import tensorflow|from tensorflow|import tf\b)', re.IGNORECASE),
    'pytorch': re.compile(r'^\s*(import torch|from torch|import pytorch\b)', re.IGNORECASE),
    'scikit-learn': re.compile(r'^\s*(import sklearn|from sklearn|import scikit-learn\b)', re.IGNORECASE),
    'jax': re.compile(r'^\s*(import jax|from jax|import flax\b)', re.IGNORECASE),
    'xgboost': re.compile(r'^\s*(import xgboost|from xgboost|import xgb\b)', re.IGNORECASE)
}

def process_repository(repo_url, output_base, temp_dir='temp_clones'):
    """Process a single repository"""
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    clone_path = os.path.join(temp_dir, repo_name)
    classified_files = set()

    try:
        # Clone repository
        logging.info(f"Cloning {repo_url}")
        Repo.clone_from(repo_url, clone_path, depth=1)
        
        # Scan Python files
        for root, _, files in os.walk(clone_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.readlines()
                    
                    # Check for library usage
                    for line in content:
                        for lib, pattern in LIBRARY_PATTERNS.items():
                            if pattern.search(line):
                                dest_dir = os.path.join(output_base, lib, repo_name)
                                os.makedirs(dest_dir, exist_ok=True)
                                shutil.copy(file_path, dest_dir)
                                classified_files.add(lib)
                                break  # Stop checking other libs for this line

    except Exception as e:
        logging.error(f"Error processing {repo_url}: {str(e)}")
    finally:
        # Cleanup cloned repository
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path, ignore_errors=True)
            logging.info(f"Cleaned up {clone_path}")

    return classified_files

def main(input_file, output_dir, temp_dir='temp_clones'):
    """Main processing function"""
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r') as f:
        repos = [line.strip() for line in f if line.strip()]

    summary = {lib: 0 for lib in LIBRARY_PATTERNS}
    for repo_url in repos:
        found_libs = process_repository(repo_url, output_dir, temp_dir)
        for lib in found_libs:
            summary[lib] += 1

    logging.info("\nProcessing Summary:")
    for lib, count in summary.items():
        logging.info(f"{lib.capitalize():<12}: {count} repos")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify GitHub repositories by ML library usage')
    parser.add_argument('--input', required=True, help='Text file with GitHub repo URLs')
    parser.add_argument('--output', default='classified_files', help='Output directory')
    args = parser.parse_args()

    main(args.input, args.output)
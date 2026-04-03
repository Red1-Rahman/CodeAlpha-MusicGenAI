import os
import argparse
import subprocess

def main(mode):
    print(f"Starting end-to-end pipeline for mode: {mode}")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Preprocess
    subprocess.run(["python", os.path.join(base_dir, "src", "preprocess.py"), "--mode", mode], check=True)
    
    # 2. Train
    subprocess.run(["python", os.path.join(base_dir, "src", "train.py"), "--mode", mode], check=True)
    
    # 3. Generate
    subprocess.run(["python", os.path.join(base_dir, "src", "generate.py"), "--mode", mode], check=True)
    
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the whole AI Music Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['hiphop', 'retro', 'mixed'])
    args = parser.parse_args()
    main(args.mode)
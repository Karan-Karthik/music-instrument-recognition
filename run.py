import subprocess

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(["python", script_name], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        exit(1)


run_script("src/data_loader.py")
run_script("src/train.py")
run_script("src/evaluate.py")

print("All scripts executed successfully.")

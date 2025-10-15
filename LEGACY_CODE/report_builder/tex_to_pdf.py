import subprocess

def run_pdflatex(tex_path, output_dir, runs=1):
    for _ in range(runs):
        process = subprocess.Popen(
            ["pdflatex", "--shell-escape", "-interaction=nonstopmode", tex_path],
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()

        # Uncomment these if you need to debug:
        # if stdout:
        #     print(stdout)
        # if stderr:
        #     print(stderr)

        if process.returncode != 0:
            print(f"Error generating PDF. Return code: {process.returncode}")
            break
    
    if process.returncode == 0:
        print("Compiling successful")

import subprocess

def generate_urdf(xacro_path, output_path, args_dict):
    # Prepare command
    cmd = ["xacro", str(xacro_path)]

    # Add arguments
    for key, value in args_dict.items():
        cmd.append(f"{key}:={value}")

    # Write output to file
    with open(output_path, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)

    print(f"Generated URDF saved to: {output_path}")
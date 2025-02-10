import modal

# Create an image with system dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(["git", "procps", "build-essential", "cmake"])
    .pip_install_from_requirements("requirements-dev.txt")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        [
            "git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git",
            "cd bitsandbytes && cmake -DCOMPUTE_BACKEND=cpu -S . && make && pip --no-cache install . ",
        ]
    )
    .add_local_dir("hivemind", remote_path="/root/hivemind/hivemind")
    .add_local_file("requirements.txt", remote_path="/root/hivemind/requirements.txt")
    .add_local_file("requirements-dev.txt", remote_path="/root/hivemind/requirements-dev.txt")
    .add_local_file("requirements-docs.txt", remote_path="/root/hivemind/requirements-docs.txt")
    .add_local_file("setup.py", remote_path="/root/hivemind/setup.py")
    .add_local_file("pyproject.toml", remote_path="/root/hivemind/pyproject.toml")
    .add_local_dir("tests", remote_path="/root/hivemind/tests")
)

app = modal.App("hivemind-ci", image=image)


@app.function(image=image, timeout=1800, cpu=16, memory=8192)
def setup_and_test():
    import os
    import subprocess

    # Clone and install hivemind
    os.chdir("/root/hivemind")

    subprocess.run(["pip", "install", "."], check=True)

    environment = os.environ.copy()
    environment["HIVEMIND_MEMORY_SHARING_STRATEGY"] = "file_descriptor"
    environment["HIVEMIND_DHT_NUM_WORKERS"] = "1"

    subprocess.run(
        ["prlimit", f"--pid={os.getpid()}", "--nofile=8192"],
        check=True,
    )

    # Run tests
    subprocess.run(
        ["pytest", "--durations=0", "--durations-min=1.0", "-v", "-n", "16", "tests"],
        check=True,
        env=environment,
    )

import os
import subprocess

import modal

# Create an image with system dependencies
image = (
    modal.Image.debian_slim(python_version=os.environ["PYTHON_VERSION"])
    .apt_install(["git", "procps", "build-essential", "cmake", "golang-1.20"])
    .pip_install_from_requirements("requirements-dev.txt")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        [
            "git clone --branch 0.45.2 --depth 1 https://github.com/bitsandbytes-foundation/bitsandbytes.git",
            "cd bitsandbytes && cmake -DCOMPUTE_BACKEND=cpu -S . && make && pip --no-cache install . ",
            "ln -s /usr/lib/go-1.20/bin/go /usr/bin/go",  # Make go available in PATH
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

codecov_secret = modal.Secret.from_dict({"CODECOV_TOKEN": os.getenv("CODECOV_TOKEN")})


def setup_environment(*, build_p2pd=False):
    os.chdir("/root/hivemind")

    install_cmd = ["uv", "pip", "install", "--no-cache-dir", "--system"]
    if build_p2pd:
        install_cmd.extend([".", "--global-option=build_py", "--global-option=--buildgo", "--no-use-pep517"])
    else:
        install_cmd.append(".")
    
    subprocess.run(install_cmd, check=True)

    environment = os.environ.copy()
    environment["HIVEMIND_MEMORY_SHARING_STRATEGY"] = "file_descriptor"

    return environment


@app.function(image=image, timeout=600, cpu=8, memory=8192)
def run_tests():
    environment = setup_environment(build_p2pd=False)

    subprocess.run(
        [
            "pytest",
            "--durations=0",
            "--durations-min=1.0",
            "-v",
            "-n",
            "8",
            "--dist",
            "worksteal",
            "--timeout=60",
            "tests",
        ],
        check=True,
        env=environment,
    )


@app.function(image=image, timeout=600, cpu=8, memory=8192, secrets=[codecov_secret])
def run_codecov():
    environment = setup_environment(build_p2pd=False)

    subprocess.run(
        [
            "pytest",
            "--cov",
            "hivemind",
            "--cov-config=pyproject.toml",
            "-v",
            "-n",
            "8",
            "--dist",
            "worksteal",
            "--timeout=60",
            "tests",
        ],
        check=True,
        env=environment,
    )

    environment["CODECOV_TOKEN"] = os.environ["CODECOV_TOKEN"]

    subprocess.run(
        ["bash", "-c", "curl -Os https://uploader.codecov.io/latest/linux/codecov && chmod +x codecov && ./codecov"],
        check=True,
        env=environment,
    )


@app.function(image=image, timeout=600, cpu=1, memory=4096)
def build_and_test_p2pd():
    environment = setup_environment(build_p2pd=True)

    subprocess.run(
        [
            "pytest",
            "-k",
            "p2p",
            "-v",
            "tests",
        ],
        check=True,
        env=environment,
    )

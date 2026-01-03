import os
import subprocess

import modal

# Create an image with system dependencies
image = (
    modal.Image.debian_slim(python_version=os.environ["PYTHON_VERSION"])
    .apt_install(["git", "procps", "build-essential", "cmake", "wget"])
    .pip_install("uv")
    .add_local_file("requirements.txt", remote_path="/root/requirements.txt", copy=True)
    .add_local_file("requirements-dev.txt", remote_path="/root/requirements-dev.txt", copy=True)
    .run_commands(
        [
            "uv pip install --system -r /root/requirements-dev.txt",
            "uv pip install --system -r /root/requirements.txt",
            "git clone --branch 0.45.2 --depth 1 https://github.com/bitsandbytes-foundation/bitsandbytes.git",
            "cd bitsandbytes && cmake -DCOMPUTE_BACKEND=cpu -S . && make && pip --no-cache install . ",
        ]
    )
    .add_local_dir(
        "src/hivemind",
        remote_path="/root/repo/src/hivemind",
        ignore=["src/hivemind/proto/*_pb2.py", "**/*/p2pd"],
    )
    .add_local_file("requirements.txt", remote_path="/root/repo/requirements.txt")
    .add_local_file("requirements-dev.txt", remote_path="/root/repo/requirements-dev.txt")
    .add_local_file("requirements-docs.txt", remote_path="/root/repo/requirements-docs.txt")
    .add_local_file("setup.py", remote_path="/root/repo/setup.py")
    .add_local_file("pyproject.toml", remote_path="/root/repo/pyproject.toml")
    .add_local_dir("tests", remote_path="/root/repo/tests")
)

# Create an image with golang and other system dependencies
image_with_golang = (
    modal.Image.debian_slim(python_version=os.environ["PYTHON_VERSION"])
    .apt_install(["git", "procps", "build-essential", "cmake", "wget"])
    .pip_install("uv")
    .add_local_file("requirements.txt", remote_path="/root/requirements.txt", copy=True)
    .add_local_file("requirements-dev.txt", remote_path="/root/requirements-dev.txt", copy=True)
    .run_commands(
        [
            "uv pip install --system -r /root/requirements-dev.txt",
            "uv pip install --system -r /root/requirements.txt",
            "wget https://go.dev/dl/go1.20.11.linux-amd64.tar.gz",
            "rm -rf /usr/local/go && tar -C /usr/local -xzf go1.20.11.linux-amd64.tar.gz",
            "ln -s /usr/local/go/bin/go /usr/bin/go",
            "git clone --branch 0.45.2 --depth 1 https://github.com/bitsandbytes-foundation/bitsandbytes.git",
            "cd bitsandbytes && cmake -DCOMPUTE_BACKEND=cpu -S . && make && pip --no-cache install . ",
        ]
    )
    .add_local_dir("src/hivemind", remote_path="/root/repo/src/hivemind")
    .add_local_file("requirements.txt", remote_path="/root/repo/requirements.txt")
    .add_local_file("requirements-dev.txt", remote_path="/root/repo/requirements-dev.txt")
    .add_local_file("requirements-docs.txt", remote_path="/root/repo/requirements-docs.txt")
    .add_local_file("setup.py", remote_path="/root/repo/setup.py")
    .add_local_file("pyproject.toml", remote_path="/root/repo/pyproject.toml")
    .add_local_dir("tests", remote_path="/root/repo/tests")
)


app = modal.App("hivemind-ci")

codecov_secret = modal.Secret.from_dict(
    {
        "CODECOV_TOKEN": os.getenv("CODECOV_TOKEN"),
        "GITHUB_EVENT_PULL_REQUEST_HEAD_SHA": os.getenv("GITHUB_EVENT_PULL_REQUEST_HEAD_SHA"),
        "GITHUB_EVENT_NUMBER": os.getenv("GITHUB_EVENT_NUMBER"),
        "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY"),
    }
)


def setup_environment(*, build_p2pd=False):
    os.chdir("/root/repo")

    environment = os.environ.copy()
    environment["HIVEMIND_MEMORY_SHARING_STRATEGY"] = "file_descriptor"

    if build_p2pd:
        environment["HIVEMIND_BUILDGO"] = "1"
        install_cmd = ["pip", "install", "--no-cache-dir", "."]
    else:
        install_cmd = ["pip", "install", "--no-cache-dir", "-e", "."]

    subprocess.run(install_cmd, check=True, env=environment)

    return environment


@app.function(image=image, timeout=1200, cpu=8, memory=8192)
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
            "/root/repo/tests",
        ],
        check=True,
        env=environment,
        cwd="/tmp",
    )


@app.function(image=image, timeout=1200, cpu=8, memory=8192, secrets=[codecov_secret])
def run_codecov():
    environment = setup_environment(build_p2pd=False)

    subprocess.run(
        [
            "pytest",
            "--cov",
            "hivemind",
            "--cov-config=/root/repo/pyproject.toml",
            "-v",
            "/root/repo/tests",
        ],
        check=True,
        env=environment,
        cwd="/tmp",
    )

    environment.update(
        {
            "CODECOV_TOKEN": os.environ["CODECOV_TOKEN"],
            "CC_SHA": os.environ["GITHUB_EVENT_PULL_REQUEST_HEAD_SHA"],
            "CC_PR": os.environ["GITHUB_EVENT_NUMBER"],
            "CC_SLUG": os.environ["GITHUB_REPOSITORY"],
        }
    )

    subprocess.run(
        [
            "bash",
            "-c",
            "wget -q https://uploader.codecov.io/latest/linux/codecov && chmod +x codecov "
            "&& ./codecov create-commit -C $CC_SHA -P $CC_PR -r $CC_SLUG --git-service github "
            "&& ./codecov create-report -C $CC_SHA -r $CC_SLUG --git-service github "
            "&& ./codecov do-upload -C $CC_SHA -r $CC_SLUG -P $CC_PR --git-service github",
        ],
        check=True,
        env=environment,
    )


@app.function(image=image_with_golang, timeout=600, cpu=1, memory=4096)
def build_and_test_p2pd():
    environment = setup_environment(build_p2pd=True)

    subprocess.run(
        [
            "pytest",
            "-k",
            "p2p",
            "-v",
            "/root/repo/tests",
        ],
        check=True,
        env=environment,
        cwd="/tmp",
    )

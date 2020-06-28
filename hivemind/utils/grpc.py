"""
Utilities for running GRPC services: compile protobuf, patch legacy versions, etc
"""
import functools
import os
import sys
import tempfile
from typing import Tuple
from argparse import Namespace
import grpc_tools.protoc


@functools.lru_cache(maxsize=None)
def compile_grpc(proto: str, *args: str) -> Tuple[Namespace, Namespace]:
    """
    Compiles and loads grpc protocol defined by protobuf string

    :param proto: protocol buffer code as a string, as in open('file.proto').read()
    :param args: extra cli args for grpc_tools.protoc compiler, e.g. '-Imyincludepath'
    :returns: messages, services protobuf
    """
    base_include = grpc_tools.protoc.pkg_resources.resource_filename('grpc_tools', '_proto')

    with tempfile.TemporaryDirectory(prefix='compile_grpc_') as build_dir:
        proto_path = tempfile.mktemp(prefix='grpc_', suffix='.proto', dir=build_dir)
        with open(proto_path, 'w') as fproto:
            fproto.write(proto)

        cli_args = (
            grpc_tools.protoc.__file__, f"-I{base_include}",
            f"--python_out={build_dir}", f"--grpc_python_out={build_dir}",
            f"-I{build_dir}", *args, os.path.basename(proto_path))
        code = grpc_tools.protoc._protoc_compiler.run_main([arg.encode() for arg in cli_args])
        if code:  # hint: if you get this error in jupyter, run in console for richer error message
            raise ValueError(f"{' '.join(cli_args)} finished with exit code {code}")

        try:
            sys.path.append(build_dir)
            pb2_fname = os.path.basename(proto_path)[:-len('.proto')] + '_pb2'
            messages, services = __import__(pb2_fname, fromlist=['*']), __import__(pb2_fname + '_grpc')
            return messages, services
        finally:
            if sys.path.pop() != build_dir:
                raise ImportError("Something changed sys.path while compile_grpc was in progress.")

import yaml
import resource
from hivemind.server.run_server import make_dummy_server, background_server

with open("config.yml", "r") as f:
    args = yaml.safe_load(f)

if args.pop('increase_file_limit'):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        print("Setting open file limit to soft={}, hard={}".format(max(soft, 2 ** 15), max(hard, 2 ** 15)))
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 2 ** 15), max(hard, 2 ** 15)))
    except:
        print("Could not increase open file limit, currently at soft={}, hard={}".format(soft, hard))

try:
    server = make_dummy_server(**args, start=True, verbose=True)
    server.join()
finally:
    server.shutdown()
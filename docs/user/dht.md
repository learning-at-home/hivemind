# Hivemind DHT

In order to coordinate, hivemind peers form a Distributed Hash Table: distributed "dictionary" where each peer
can store and get values. To initialize the first DHT node, run

```python
from hivemind import DHT, get_dht_time

dht = DHT(start=True)
# create the first DHT node that listens for incoming connections from localhost only

print("For incoming connections, use:", dht.get_visible_maddrs())
```

You can now start more peers that connect to an existing DHT node using its listen address:
```python
dht2 = DHT(initial_peers=dht.get_visible_maddrs(), start=True)
```

Note that `initial_peers` contains the address of the first DHT node.
This implies that the new node will share the key-value data with the first node, __as well as any other
nodes connected to it.__ When the two nodes are connected, subsequent peers can use any one of them (or both)
as `initial_peers` to connect to the shared "dictionary".

### Store/get operations

Once the DHT is formed, all participants can `dht.store` key-value pairs in the DHT and `dht.get` them by key:

```python
# first node: store a key-value pair for 600 seconds
store_ok = dht.store('my_key', ('i', 'love', 'bees'),
                     expiration_time=get_dht_time() + 600)

# second node: get the value stored by the first node
value, expiration = dht2.get('my_key', latest=True)
assert value == ('i', 'love', 'bees')
```

As you can see, each value in a hivemind DHT is associated with an expiration time,
computed current `get_dht_time()` with some offset.
This expiration time is used to cleanup old data and resolve write conflicts: 
DHT nodes always prefer values with higher expiration time and may delete any value past its expiration.

### Values with subkeys

Hivemind DHT also supports a special value type that is itself a dictionary. When nodes store such a value,
they add sub-keys to the dictionary instead of overwriting it.

Consider an example where three DHT nodes want to find out who is going to attend the party:

```python
alice_dht = DHT(initial_peers=dht.get_visible_maddrs(), start=True)
bob_dht = DHT(initial_peers=dht2.get_visible_maddrs(), start=True)
carol_dht = DHT(initial_peers=alice_dht.get_visible_maddrs(), start=True)


# first, each peer stores a subkey for the same key
alice_dht.store('party', subkey='alice', value='yes', expiration_time=get_dht_time() + 600)
bob_dht.store('party', subkey='bob', value='yes', expiration_time=get_dht_time() + 600)
carol_dht.store('party', subkey='carol', value='no', expiration_time=get_dht_time() + 600)

# then, any peer can get the full list of attendees
attendees, expiration = alice_dht.get('party', latest=True)
print(attendees)
# {'alice': ValueWithExpiration(value='yes', expiration_time=1625504352.2668974),
#  'bob': ValueWithExpiration(value='yes', expiration_time=1625504352.2884178),
#  'carol': ValueWithExpiration(value='no', expiration_time=1625504352.3046832)}
```

When training over the Internet, some `dht.get/store` requests may run for hundreds of milliseconds and even seconds.
To minimize the wait time, you can call these requests asynchronously via 
[`dht.store/get/run_coroutine(..., return_future=True)`__](https://learning-at-home.readthedocs.io/en/latest/modules/dht.html#hivemind.dht.DHT.get)
. This will run the corresponding command in background and return a [Future-like](https://docs.python.org/3/library/concurrent.futures.html) object that can be awaited.
Please also note that the returned future is compatible with asyncio (i.e. can be awaited inside the event loop).

For more details on DHT store/get and expiration time, please refer to the [documentation for DHT and DHTNode](https://learning-at-home.readthedocs.io/en/latest/modules/dht.html#dht-and-dhtnode)

### Running across the Internet

By default, DHT nodes are only accessible from your localhost. In order to run with multiple geographically
distributed computers, one must connect DHT to a global network. Currently, there are two ways achieve this.

The recommended approach is to grow the network from one or several initial peers. These can be any computers with a
public IP address that are always online. Each of these peers should simply create `hivemind.DHT` and set it to
accept incoming connections from the internet:

```python
import hivemind
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    start=True)

print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))
```

Running this code will print several, typically, 4 or 6 strings of the following form (example):
```shell
/ip4/185.185.123.124/tcp/40615/p2p/QmaVTB2LwayToK2rzMkaCbkCaH7nF2rTHIS0IS0AN0EXAMPLE
/ip4/127.0.0.1/tcp/40615/p2p/QmaVTB2LwayToK2rzMkaCbkCaH7nF2rTHIS0IS0AN0EXAMPLE
/ip4/185.185.123.124/udp/40346/quic/p2p/QmaVTB2LwayToK2rzMkaCbkCaH7nF2rTHIS0IS0AN0EXAMPLE
/ip4/127.0.0.1/udp/40346/quic/p2p/QmaVTB2LwayToK2rzMkaCbkCaH7nF2rTHIS0IS0AN0EXAMPLE
Global IP: 185.185.123.124
```
The lines that contain addresses that other nodes can use to connect to the network:
- `127.0.0.1` or `192.168.X.Y` are only accessible from your computer or local network, respectively.
- The remaining address is __global__ (`185.185.123.124` in the example, yours will be different).

To connect a new peer to the network, you should specify `initial_peers` as the addresses that 
correspond to the public IP:

```python
import hivemind
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    initial_peers=[
        "/ip4/185.185.123.124/tcp/40615/p2p/QmaVTB2LwayToK2rzMkaCbkCaH7nF2rTHIS0IS0AN0EXAMPLE",
        "/ip4/185.185.123.124/udp/40346/quic/p2p/QmaVTB2LwayToK2rzMkaCbkCaH7nF2rTHIS0IS0AN0EXAMPLE",
    ], start=True)
```

That's it, now the two DHT nodes are connected. If you connect additional peers to the network, you only need to specify
one (or a subset) of peers as `initial_peers`.
In case your peer operates behind a restrictive firewall, you may find it beneficial to set `client_mode=True`. In this
 case, the DHT instance will access others, but it will not announce that other peers can connect to it.

Another (experimental) way is to use [IPFS](https://ipfs.io/): a global decentralized network for file storage.
We are not storing any files here: instead, we can use IPFS nodes to help hivemind peers find each other.
To use this strategy, set `use_ipfs=True` in each DHT node you create. This allows you to connect DHT multiple even if
all of them are behind NAT. However, this strategy may be unreliable and depend heavily on the availability of public
IPFS nodes.

To learn more about the network address format, read [libp2p addressing](https://docs.libp2p.io/concepts/addressing/)
For an example of how to set up DHT in a distributed training experiment, see
 [examples/albert](https://github.com/learning-at-home/hivemind/tree/master/examples/albert)

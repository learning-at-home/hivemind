# Hivemind DHT

In order to coordinate, hivemind peers form a Distributed Hash Table: distributed "dictionary" where each peer
can store and get values. To initialize the first DHT node, run

```python
from hivemind import DHT, get_dht_time

dht = DHT(listen_on="127.0.0.1:1337", start=True)
# create the first DHT node that listens for incoming connections on port 1337 from localhost only
```

You can now start more peers that connect to an existing DHT node using its listen address:
```python
dht = DHT(listen_on="127.0.0.1:1338", initial_peers=["127.0.0.1:1337"], start=True)
```

Note that `initial_peers` contains the address of the first DHT node.
This implies that the resulting node will have shared key-value with the first node, __as well as any other
nodes connected to it.__ When the two nodes are connected, subsequent peers can use any one of them (or both)
as `initial_peers` to connect to the shared "dictionary".

### Store/get operations

Once the DHT is formed, all participants can `dht.store` key-value pairs to the DHT and `dht.get` them by key:

```python
# first node: store a key-value for 600 seconds
store_ok = dht.store('my_key', ('i', 'love', 'bees'),
                     expiration_time=get_dht_time() + 600)

# second node: get the value stored by the first node
value, expiration = dht.get('my_key', latest=True)
assert value == ('i', 'love', 'bees')
```

As you can see, each value in a hivemind DHT is associated with an expiration time,
computed current `get_dht_time()` with some offset.
This expiration time is used to cleanup old data and resolve write conflicts: 
DHT nodes always prefer values with higher expiration time and may delete any value past its expiration.

### Values with subkeys

Hivemind DHT also supports a special value type that is itself a dictionary. When nodes store to such a value,
they add sub-keys to the dictionary instead of overwriting it.

Consider an example where three DHT nodes want to find out who going to attend the party:

```python
from hivemind import DHT, get_dht_time
alice_dht = DHT(listen_on="127.0.0.1:3030", start=True)
bob_dht = DHT(listen_on="127.0.0.1:3031", initial_peers=["127.0.0.1:3030"], start=True)
carol_dht = DHT(listen_on="127.0.0.1:3032", initial_peers=["127.0.0.1:3031"], start=True)


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


For a more detailed explanation of DHT store/get and expiration time, please refer to the [documentation for DHT and DHTNode](https://learning-at-home.readthedocs.io/en/latest/modules/dht.html#dht-and-dhtnode)


### Advanced usage
#TODO REMOVE!!!
# but mention that they are awaitable
In addition to regular store/get, hivemind.DHT implements a number of features that are often useful in the context of
decentralized training.

* [`DHT.run_coroutine`](https://learning-at-home.readthedocs.io/en/latest/modules/dht.html#hivemind.dht.DHT.run_coroutine)
  allows user to define a composite async function that contains multiple bulk store/get request. This function will then
  be executed on DHT side with less overhead, than using multiple independent store/get request.
* [`dht.store/get/run_coroutine(..., return_future=True)`__](https://learning-at-home.readthedocs.io/en/latest/modules/dht.html#hivemind.dht.DHT.get)
  will run the corresponding command in background and return an awaitable Future-like object.
* [`cache_locally / cache_refresh_before_expiry`](https://learning-at-home.readthedocs.io/en/latest/modules/dht.html#hivemind.dht.DHTNode.create)
  if will cache existing keys locally and, if they were accessed from cache, re-fetch them in background a few seconds before
  expiration. This trick allows DHT to ensure that frequently accessed keys are always available instantly. This option is
  enabled by default.

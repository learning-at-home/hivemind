# Networking and libp2p

Hivemind utilizes [libp2p](https://libp2p.io/) as its underlying networking stack. This library provides a robust, decentralized peer-to-peer network infrastructure, handling peer discovery, connection management, encryption, and multiplexing.

Understanding some concepts of libp2p may be helpful when working with or contributing to Hivemind's networking code.

## Multiaddrs

A peer on a libp2p network can have multiple addresses, which are represented as [Multiaddrs](https://docs.libp2p.io/concepts/addressing/). A `Multiaddr` is a standard way to represent a peer's location and transport protocols in a compact, readable format.

Example of a multiaddr:
`/ip4/192.168.1.10/tcp/8000/p2p/QmYyQSo1c1Ym7orWxLYvCrM2EmxFTANf8wXmmE7DWjhx5N`

A single peer (`p2p/Qm...`) might be accessible through multiple distinct multiaddrs (e.g., an internal IPv4 address, a public IPv4 address, or via different transport protocols).

## Connection Deduplication

When Hivemind connects to a peer, it may be aware of multiple multiaddrs associated with that peer. Because multiple connections to the same peer are redundant and consume resources, `libp2p` manages connection deduplication.

If multiple multiaddrs correspond to the same running libp2p daemon (identified by its unique peer ID), the underlying stack will deduplicate these connections. Consequently, you interact with an abstract peer ID instead of an individual network endpoint or address.

## The P2P Daemon

Hivemind uses the [Go implementation of the libp2p daemon](https://github.com/libp2p/go-libp2p-daemon). Rather than implementing libp2p directly in Python, Hivemind communicates with a background `p2pd` process via an RPC interface. This prevents many concurrency issues in Python and leverages a highly optimized Go networking stack.

## Further Reading

- [libp2p Documentation](https://docs.libp2p.io/)
- [Addressing (Multiaddr)](https://docs.libp2p.io/concepts/addressing/)
- [Peer Identity](https://docs.libp2p.io/concepts/peer-id/)
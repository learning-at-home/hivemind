class Error(Exception):
    pass


class LookupError(LookupError, Error):
    pass


class ProtocolLookupError(LookupError):
    """
    MultiAddr did not contain a protocol with the requested code
    """

    def __init__(self, proto, string):
        self.proto = proto
        self.string = string

        super().__init__("MultiAddr {0!r} does not contain protocol {1}".format(string, proto))


class ParseError(ValueError, Error):
    pass


class StringParseError(ParseError):
    """
    MultiAddr string representation could not be parsed
    """

    def __init__(self, message, string, protocol=None, original=None):
        self.message = message
        self.string = string
        self.protocol = protocol
        self.original = original

        if protocol:
            message = "Invalid MultiAddr {0!r} protocol {1}: {2}".format(string, protocol, message)
        else:
            message = "Invalid MultiAddr {0!r}: {1}".format(string, message)

        super().__init__(message)


class BinaryParseError(ParseError):
    """
    MultiAddr binary representation could not be parsed
    """

    def __init__(self, message, binary, protocol, original=None):
        self.message = message
        self.binary = binary
        self.protocol = protocol
        self.original = original

        message = "Invalid binary MultiAddr protocol {0}: {1}".format(protocol, message)

        super().__init__(message)


class ProtocolRegistryError(Error):
    pass


ProtocolManagerError = ProtocolRegistryError


class ProtocolRegistryLocked(Error):
    """Protocol registry was locked and doesn't allow any further additions"""

    def __init__(self):
        super().__init__("Protocol registry is locked and does not accept any new values")


class ProtocolExistsError(ProtocolRegistryError):
    """Protocol with the given name or code already exists"""

    def __init__(self, proto, kind="name"):
        self.proto = proto
        self.kind = kind

        super().__init__("Protocol with {0} {1!r} already exists".format(kind, getattr(proto, kind)))


class ProtocolNotFoundError(ProtocolRegistryError):
    """No protocol with the given name or code found"""

    def __init__(self, value, kind="name"):
        self.value = value
        self.kind = kind

        super().__init__("No protocol with {0} {1!r} found".format(kind, value))

"""
Exceptions used in the ChimeraBot codebase.
"""


class ChimeraBotBaseException(Exception):
    """
    Most errors raised in ChimeraBot should inherit this class so we can
    differentiate them from errors that come from dependencies.
    """


class ArgumentParserError(ChimeraBotBaseException):
    """
    Unable to parse a command (like start, stop, etc) from the chimerabot client
    """


class OracleRateUnavailable(ChimeraBotBaseException):
    """
    Asset value from third party is unavailable
    """


class InvalidScriptModule(ChimeraBotBaseException):
    """
    The file does not contain a ScriptBase subclass
    """


class InvalidController(ChimeraBotBaseException):
    """
    The file does not contain a ControllerBase subclass
    """

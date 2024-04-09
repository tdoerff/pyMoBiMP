class Error(Exception):
    pass

class WrongNumberOfArguments(Error, TypeError):
    pass

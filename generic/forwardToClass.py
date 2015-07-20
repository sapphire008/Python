class forwardTo(object):
    """
    A descriptor based recipe that makes it possible to write shorthands
    that forward attribute access from one object onto another.

    >>> class C(object):
    ...     def __init__(self):
    ...         class CC(object):
    ...             def xx(self, extra):
    ...                 return 100 + extra
    ...             foo = 42
    ...         self.cc = CC()
    ...
    ...     localcc = forwardTo('cc', 'xx')
    ...     localfoo = forwardTo('cc', 'foo')
    ...
    >>> print C().localcc(10)
    110
    >>> print C().localfoo
    42

    Arguments: objectName - name of the attribute containing the second object.
               attrName - name of the attribute in the second object.
    Returns:   An object that will forward any calls as described above.

    * For a more robust code, you may want to consider replacing
    getattr(instance, self.objectName) with
    operator.attrgetter(self.objectName)(instance).
    This would allow objectName to be a dotted name
    (e.g., so you could have A.c be a proxy for A.x.y.z.d).
    """
    def __init__(self, objectName, attrName):
        self.objectName = objectName
        self.attrName = attrName
    def __get__(self, instance, owner=None):
        return getattr(getattr(instance, self.objectName), self.attrName)
    def __set__(self, instance, value):
        setattr(getattr(instance, self.objectName), self.attrName, value)
    def __delete__(self, instance):
        delattr(getattr(instance, self.objectName), self.attrName)

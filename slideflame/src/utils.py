def extend_instance(obj, mixin):
    """
    Apply a mixin to an already-instantiated object by creating a new dynamic type.

    NOTE: mixin must come first so its methods (e.g., forward override) take priority.
    """
    base_cls = obj.__class__
    base_cls_name = base_cls.__name__
    obj.__class__ = type(base_cls_name, (mixin, base_cls), {})


def getattr_recursive(obj, att: str):
    """
    Return nested attribute of obj.
    Example: getattr_recursive(obj, 'a.b.c') == obj.a.b.c
    """
    if att == "":
        return obj
    parts = att.split(".")
    for p in parts:
        obj = getattr(obj, p)
    return obj


def setattr_recursive(obj, att: str, val):
    """
    Set nested attribute of obj.
    Example: setattr_recursive(obj, 'a.b.c', val) == (obj.a.b.c = val)
    """
    parts = att.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], val)


def apply_with_stopping_condition(
    module,
    apply_fn,
    apply_condition=None,
    stopping_condition=None,
    **other_args,
):
    """
    Walk module tree and apply apply_fn(child) where apply_condition(child) is True,
    but stop descending into a subtree if stopping_condition(child) is True.
    """
    if stopping_condition is not None and stopping_condition(module):
        return
    if apply_condition is None or apply_condition(module):
        apply_fn(module, **other_args)
    for child in module.children():
        apply_with_stopping_condition(
            child,
            apply_fn,
            apply_condition=apply_condition,
            stopping_condition=stopping_condition,
            **other_args,
        )
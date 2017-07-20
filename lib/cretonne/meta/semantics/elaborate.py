from .primitives import GROUP as PRIMITIVES
from cdsl.ti import ti_rtl, TypeEnv, get_type_env
from cdsl.typevar import TypeVar
import base.semantics  # noqa

try:
    from typing import TYPE_CHECKING, Dict, Union, List # noqa
    from cdsl.xform import Rtl, XForm # noqa
    from cdsl.ast import Var, Def, VarMap # noqa
    from cdsl.ti import VarTyping # noqa
except ImportError:
    TYPE_CHECKING = False


def is_rtl_concrete(r):
    # type: (Rtl) -> bool
    """Return True iff every Var in the Rtl r has a single type."""
    return all(v.get_typevar().singleton_type() is not None for v in r.vars())


def cleanup_concrete_rtl(r):
    # type: (Rtl) -> Rtl
    """
    Given an Rtl r
    1) assert that there is only 1 possible concrete typing T for r
    2) Assign a singleton TV with the single type t \in T for each Var v \in r
    """
    # 1) Infer the types of any of the remaining vars in res
    typenv = get_type_env(ti_rtl(r, TypeEnv()))
    typenv.normalize()
    typenv = typenv.extract()

    # 2) Make sure there is only one possible type assignment
    typings = list(typenv.concrete_typings())
    assert len(typings) == 1
    typing = typings[0]

    # 3) Assign the only possible type to each variable.
    for v in typenv.vars:
        if v.get_typevar().singleton_type() is not None:
            continue

        v.set_typevar(TypeVar.singleton(typing[v].singleton_type()))

    return r


def apply(r, x, suffix=None):
    # type: (Rtl, XForm) -> Rtl
    """
    Given a concrete Rtl r and XForm x, s.t. r matches x.src, return the
    corresponding concrete x.dst. If suffix is provided, any temporary defs are
    renamed with '.suffix' appended to their old name.
    """
    assert is_rtl_concrete(r)
    s = x.src.substitution(r, {})  # type: VarMap
    assert s is not None

    if (suffix is not None):
        for v in x.dst.vars():
            if v.is_temp():
                assert v not in s
                s[v] = Var(v.name + '.' + suffix)

    dst = x.dst.copy(s)
    return cleanup_concrete_rtl(dst)


def find_matching_xform(d):
    # type: (Def) -> XForm
    res = []  # type: List[XForm]
    typing = {v:   v.get_typevar() for v in d.vars()}  # type: VarTyping

    for x in d.expr.inst.semantics:
        subst = d.substitution(x.src.rtl[0], {})

        if x.ti.permits({subst[v]: tv for (v, tv) in typing.items()}):
            res.append(x)

    assert len(res) == 1
    return res[0]


def elaborate(r):
    # type: (Rtl) -> Rtl
    """
    Given an Rtl r, return a semantically equivalent Rtl r1 consisting only
    primitive instructions.
    """
    fp = False
    primitives = set(PRIMITIVES.instructions)
    idx = 0

    while not fp:
        assert is_rtl_concrete(r)
        new_defs = []  # type: List[Def]
        fp = True

        for d in r.rtl:
            inst = d.expr.inst

            if (inst not in primitives):
                transformed = apply(Rtl(d), find_matching_xform(d), str(idx))
                idx += 1
                # TODO: Get fresh variable names
                new_defs.extend(transformed.rtl)
                fp = False
            else:
                new_defs.append(d)

        r.rtl = tuple(new_defs)

    return r

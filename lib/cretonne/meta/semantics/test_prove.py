from __future__ import absolute_import
from base.instructions import iadd, isub, vsplit, load
from base.immediates import memflags, offset32
from base.types import i16, i64, b1
from cdsl.ast import Var
from cdsl.xform import Rtl, XForm
from unittest import TestCase
import base.semantics  # noqa
from .smtlib import xform_correct
from cdsl.typevar import TypeVar
from .types import mem64


class TestXFormCorrect(TestCase):
    def test_trivial_good(self):
        # type: () -> None
        x = Var('x')
        y = Var('y')
        z = Var('z')
        x = XForm(
            Rtl((y, z) << vsplit(x)),
            Rtl((y, z) << vsplit(x))
        )
        for t in x.ti.concrete_typings():
            assert xform_correct(x, t)

    def test_trivial_bad(self):
        # type: () -> None
        x = Var('x')
        y = Var('y')
        z = Var('z')
        x = XForm(
            Rtl(z << iadd(x, y)),
            Rtl(z << isub(x, y))
        )
        for t in x.ti.concrete_typings():
            assert not xform_correct(x, t)
            break

    def test_double_load_notrap_remove(self):
        # type: () -> None
        addr = Var('addr')
        val1 = Var('val1')
        val2 = Var('val2')
        in_trapped = Var('in_trapped', typevar=TypeVar.singleton(b1))
        in_mem = Var('in_mem', typevar=TypeVar.singleton(mem64))
        x = XForm(
            Rtl(
                val1 << load(memflags(notrap=False, aligned=False), addr,
                             offset32(0)),
                val2 << load(memflags(notrap=False, aligned=False), addr,
                             offset32(0))
            ),
            Rtl(
                val1 << load(memflags(notrap=False, aligned=False), addr,
                             offset32(0)),
                val2 << load(memflags(notrap=True, aligned=False), addr,
                             offset32(0))
            ),
            implicit_inputs=[in_trapped, in_mem])

        t = {
                x.symtab['addr']:   TypeVar.singleton(i64),
                x.symtab['val1']:   TypeVar.singleton(i16),
                x.symtab['val2']:   TypeVar.singleton(i16),
                x.symtab['in_trapped']: TypeVar.singleton(b1),
        }
        assert xform_correct(x, t)

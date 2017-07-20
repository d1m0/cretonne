from __future__ import absolute_import
from base.instructions import vselect, vsplit, vconcat, iconst, iadd, bint # noqa
from base.instructions import b1, icmp, iadd_cout, iadd_cin, uextend # noqa
from base.instructions import fdemote, sextend, ireduce, fpromote # noqa
from base.legalize import narrow, expand # noqa
from base.immediates import intcc # noqa
from base.types import i64, i8, b32, i32
from cdsl.typevar import TypeVar
from cdsl.ast import Var
from cdsl.xform import Rtl, XForm # noqa
from cdsl.ti import ti_rtl, subst, TypeEnv, get_type_env, TypesEqual, WiderOrEq # noqa
from unittest import TestCase
from functools import reduce # noqa
from .elaborate import cleanup_concrete_rtl, elaborate,\
        is_rtl_concrete
from .primitives import prim_to_bv, bvsplit, prim_from_bv, bvconcat, bvadd

try:
    from .ti import TypeMap, ConstraintList, VarTyping, TypingOrError # noqa
    from typing import List, Dict, Tuple, TYPE_CHECKING, cast # noqa
except ImportError:
    TYPE_CHECKING = False


class ElaborationBaseTest(TestCase):
    def setUp(self):
        # type: () -> None
        self.v0 = Var("v0")
        self.v1 = Var("v1")
        self.v2 = Var("v2")
        self.v3 = Var("v3")
        self.v4 = Var("v4")
        self.v5 = Var("v5")
        self.v6 = Var("v6")
        self.v7 = Var("v7")
        self.v8 = Var("v8")
        self.v9 = Var("v9")
        self.imm0 = Var("imm0")
        self.IxN_nonscalar = TypeVar("IxN_nonscalar", "", ints=True,
                                     scalars=False, simd=True)
        self.TxN = TypeVar("TxN", "", ints=True, bools=True, floats=True,
                           scalars=False, simd=True)
        self.b1 = TypeVar.singleton(b1)


def concrete_rtls_eq(r1, r2):
    # type: (Rtl, Rtl) -> bool
    assert is_rtl_concrete(r1)
    assert is_rtl_concrete(r2)

    s = r1.substitution(r2, {})

    if s is None:
        return False

    for (v, v1) in s.items():
        if v.get_typevar().singleton_type() !=\
           v1.get_typevar().singleton_type():
            print("Differ on {}->{}: {} != {}"
                  .format(v, v1, v.get_typevar().singleton_type(),
                          v1.get_typevar().singleton_type()))
            return False

    return True


class TestCleanupConcreteRtl(ElaborationBaseTest):
    def test_cleanup_concrete_rtl(self):
        # type: () -> None
        typ = i64.by(4)
        r = Rtl(
                (self.v0, self.v1) << vsplit(self.v2),
        )
        r1 = cleanup_concrete_rtl(r)

        s = r.substitution(r1, {})
        assert s is not None
        assert s[self.v2].get_typevar().singleton_type() == typ
        assert s[self.v0].get_typevar().singleton_type() == i64.by(2)
        assert s[self.v1].get_typevar().singleton_type() == i64.by(2)

    def test_cleanup_concrete_rtl_fail(self):
        # type: () -> None
        r = Rtl(
                (self.v0, self.v1) << vsplit(self.v2),
        )

        with self.assertRaises(AssertionError):
            cleanup_concrete_rtl(r)

    def test_cleanup_concrete_rtl_ireduce(self):
        # type: () -> None
        r = Rtl(
                self.v0 << ireduce(self.v1),
        )

        r1 = cleanup_concrete_rtl(r)

        s = r.substitution(r1, {})
        assert s is not None
        assert s[self.v0].get_typevar().singleton_type() == i8.by(2)
        assert s[self.v1].get_typevar().singleton_type() == i8.by(2)

    def test_cleanup_concrete_rtl_ireduce_bad(self):
        # type: () -> None
        r = Rtl(
                self.v0 << ireduce(self.v1),
        )

        with self.assertRaises(AssertionError):
            cleanup_concrete_rtl(r)

    def test_vselect_icmpimm(self):
        # type: () -> None
        r = Rtl(
                self.v0 << iconst(self.imm0),
                self.v1 << icmp(intcc.eq, self.v2, self.v0),
                self.v5 << vselect(self.v1, self.v3, self.v4),
        )

        r1 = cleanup_concrete_rtl(r)

        s = r.substitution(r1, {})
        assert s is not None
        assert s[self.v0].get_typevar().singleton_type() == i32.by(4)
        assert s[self.v2].get_typevar().singleton_type() == i32.by(4)
        assert s[self.v1].get_typevar().singleton_type() == b32.by(4)

    def test_bint(self):
        # type: () -> None
        r = Rtl(
            self.v4 << iadd(self.v1, self.v2),
            self.v5 << bint(self.v3),
            self.v0 << iadd(self.v4, self.v5)
        )

        r1 = cleanup_concrete_rtl(r)

        s = r.substitution(r1, {})
        assert s is not None
        assert s[self.v0].get_typevar().singleton_type() == i32.by(8)
        assert s[self.v1].get_typevar().singleton_type() == i32.by(8)
        assert s[self.v2].get_typevar().singleton_type() == i32.by(8)
        assert s[self.v4].get_typevar().singleton_type() == i32.by(8)
        assert s[self.v5].get_typevar().singleton_type() == i32.by(8)
        assert s[self.v3].get_typevar().singleton_type() == b1.by(8)


class TestElaborate(ElaborationBaseTest):
    def test_elaborate_vsplit(self):
        # type: () -> None
        i32.by(4)  # Make sure i32x4 exists.
        i32.by(2)  # Make sure i32x2 exists.
        r = Rtl(
                (self.v0, self.v1) << vsplit.i32x4(self.v2),
        )
        sem = elaborate(cleanup_concrete_rtl(r))
        bvx = Var('bvx')
        bvlo = Var('bvlo')
        bvhi = Var('bvhi')
        x = Var('x')
        lo = Var('lo')
        hi = Var('hi')

        assert concrete_rtls_eq(sem, cleanup_concrete_rtl(Rtl(
            bvx << prim_to_bv.i32x4(x),
            (bvlo, bvhi) << bvsplit.bv128(bvx),
            lo << prim_from_bv.i32x2.bv64(bvlo),
            hi << prim_from_bv.i32x2.bv64(bvhi))))

    def test_elaborate_vconcat(self):
        # type: () -> None
        i32.by(4)  # Make sure i32x4 exists.
        i32.by(2)  # Make sure i32x2 exists.
        r = Rtl(
                self.v0 << vconcat.i32x2(self.v1, self.v2),
        )
        sem = elaborate(cleanup_concrete_rtl(r))
        bvx = Var('bvx')
        bvlo = Var('bvlo')
        bvhi = Var('bvhi')
        x = Var('x')
        lo = Var('lo')
        hi = Var('hi')

        assert concrete_rtls_eq(sem, cleanup_concrete_rtl(Rtl(
            bvlo << prim_to_bv.i32x2(lo),
            bvhi << prim_to_bv.i32x2(hi),
            bvx << bvconcat.bv64(bvlo, bvhi),
            x << prim_from_bv.i32x4.bv128(bvx))))

    def test_elaborate_iadd_simple(self):
        # type: () -> None
        i32.by(4)  # Make sure i32x4 exists.
        i32.by(2)  # Make sure i32x2 exists.
        r = Rtl(
                self.v0 << iadd.i32(self.v1, self.v2),
        )
        sem = elaborate(cleanup_concrete_rtl(r))
        x = Var('x')
        y = Var('y')
        a = Var('a')
        bvx = Var('bvx')
        bvy = Var('bvy')
        bva = Var('bva')

        assert concrete_rtls_eq(sem, cleanup_concrete_rtl(Rtl(
            bvx << prim_to_bv.i32(x),
            bvy << prim_to_bv.i32(y),
            bva << bvadd.bv32(bvx, bvy),
            a << prim_from_bv.i32.bv32(bva))))

    def test_elaborate_iadd_elaborate_1(self):
        # type: () -> None
        i32.by(4)  # Make sure i32x4 exists.
        i32.by(2)  # Make sure i32x2 exists.
        r = Rtl(
                self.v0 << iadd.i32x2(self.v1, self.v2),
        )
        sem = elaborate(cleanup_concrete_rtl(r))
        print ("\n".join(map(str, sem.rtl)))
        """
        x = Var('x')
        y = Var('y')
        a = Var('a')
        bvx = Var('bvx')
        bvy = Var('bvy')
        bva = Var('bva')

        assert concrete_rtls_eq(sem, cleanup_concrete_rtl(Rtl(
            bvx << prim_to_bv.i32(x),
            bvy << prim_to_bv.i32(y),
            bva << bvadd.bv32(bvx, bvy),
            a << prim_from_bv.i32.bv32(bva))))
        """

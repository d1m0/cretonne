from __future__ import absolute_import
from base.instructions import vselect, vsplit, vconcat, iconst, iadd, bint,\
    b1, icmp, iadd_cout, iadd_cin, uextend, sextend, ireduce, fpromote, \
    fdemote
from base.legalize import narrow, expand
from base.immediates import intcc
from base.types import i32, i8
from .typevar import TypeVar
from .ast import Var, Def
from .xform import Rtl, XForm
from .ti import ti_rtl, subst, TypeEnv, get_type_env, TypesEqual, WiderOrEq
from unittest import TestCase
from functools import reduce

try:
    from .ti import TypeMap, TypeConstraint, VarMap, TypingOrError # noqa
    from typing import List, Dict, Tuple, TYPE_CHECKING, cast # noqa
except ImportError:
    TYPE_CHECKING = False


def check_typing(got_or_err, expected, got_symtab=None):
    # type: (TypingOrError, Tuple[VarMap, List[TypeConstraint]], Dict[str, Var]) -> None # noqa
    """
    Check that a the typing we received (got_or_err) complies with the
    expected typing (expected)
    """
    (var_m, constr) = expected
    typ_m = {v.get_typevar(): tv for (v, tv) in var_m.items()}
    expected_typ = TypeEnv((typ_m, constr, set(var_m.keys())))
    got_typ = get_type_env(got_or_err)

    if got_symtab is not None:
        exp_symtab = {str(v): v for v in var_m.keys()}
        assert set(exp_symtab) == set(got_symtab)
        vartab = {got_symtab[v]: exp_symtab[v] for v in got_symtab.keys()}  # type: Dict[Var, Var] # noqa
    else:
        vartab = {v: v for v in got_typ.vars}

    # Make sure that the set of concrete typings of got is a subset of the set
    # of concrete typings of expected
    assert got_typ.agree(expected_typ, vartab)


def check_concrete_typing_rtl(var_types, rtl):
    # type: (VarMap, Rtl) -> None
    """
    Check that a concrete type assignment var_types (Dict[Var, TypeVar]) is
    valid for an Rtl rtl.  Specifically check that:

    1) For each Var v \in rtl, v is defined in var_types

    2) For all v, var_types[v] is a singleton type

    3) For each v, and each location u, where v is used with expected type
       tv_u, var_types[v].get_typeset() is a subset of
       subst(tv_u, m).get_typeset() where m is the substitution of
       formals->actuals we are building so far.

    4) If tv_u is non-derived and not in m, set m[tv_u]= var_types[v]
    """
    for d in rtl.rtl:
        assert isinstance(d, Def)
        inst = d.expr.inst
        # Accumulate all actual TVs for value defs/opnums in actual_tvs
        actual_tvs = [var_types[d.defs[i]] for i in inst.value_results]
        for v in [d.expr.args[i] for i in inst.value_opnums]:
            assert isinstance(v, Var)
            actual_tvs.append(var_types[v])

        # Accumulate all formal TVs for value defs/opnums in actual_tvs
        formal_tvs = [inst.outs[i].typevar for i in inst.value_results] +\
                     [inst.ins[i].typevar for i in inst.value_opnums]
        m = {}  # type: TypeMap

        # For each actual/formal pair check that they agree
        for (actual_tv, formal_tv) in zip(actual_tvs, formal_tvs):
            # actual should be a singleton
            assert actual_tv.singleton_type() is not None
            formal_tv = subst(formal_tv, m)
            # actual should agree with the concretized formal
            assert actual_tv.get_typeset().issubset(formal_tv.get_typeset())

            if formal_tv not in m and not formal_tv.is_derived:
                m[formal_tv] = actual_tv


def check_concrete_typing_xform(var_types, xform):
    # type: (VarMap, XForm) -> None
    """
    Check a concrete type assignment var_types for an XForm xform
    """
    check_concrete_typing_rtl(var_types, xform.src)
    check_concrete_typing_rtl(var_types, xform.dst)


class TypeCheckingBaseTest(TestCase):
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


class TestRTL(TypeCheckingBaseTest):
    def test_bad_rtl1(self):
        # type: () -> None
        r = Rtl(
                (self.v0, self.v1) << vsplit(self.v2),
                self.v3 << vconcat(self.v0, self.v2),
        )
        ti = TypeEnv()
        self.assertEqual(ti_rtl(r, ti),
                         "On line 1: fail ti on `typeof_v2` <: `1`: " +
                         "Error: empty type created when unifying " +
                         "`typeof_v2` and `half_vector(typeof_v2)`")

    def test_vselect(self):
        # type: () -> None
        r = Rtl(
                self.v0 << vselect(self.v1, self.v2, self.v3),
        )
        ti = TypeEnv()
        typing = ti_rtl(r, ti)
        txn = self.TxN.get_fresh_copy("TxN1")
        check_typing(typing, ({
            self.v0: txn,
            self.v1: txn.as_bool(),
            self.v2: txn,
            self.v3: txn
        }, []))

    def test_vselect_icmpimm(self):
        # type: () -> None
        r = Rtl(
                self.v0 << iconst(self.imm0),
                self.v1 << icmp(intcc.eq, self.v2, self.v0),
                self.v5 << vselect(self.v1, self.v3, self.v4),
        )
        ti = TypeEnv()
        typing = ti_rtl(r, ti)
        ixn = self.IxN_nonscalar.get_fresh_copy("IxN1")
        txn = self.TxN.get_fresh_copy("TxN1")
        check_typing(typing, ({
            self.v0: ixn,
            self.v1: ixn.as_bool(),
            self.v2: ixn,
            self.v3: txn,
            self.v4: txn,
            self.v5: txn,
        }, [TypesEqual(ixn.as_bool(), txn.as_bool())]))

    def test_vselect_vsplits(self):
        # type: () -> None
        r = Rtl(
                self.v3 << vselect(self.v0, self.v1, self.v2),
                (self.v4, self.v5) << vsplit(self.v3),
                (self.v6, self.v7) << vsplit(self.v4),
        )
        ti = TypeEnv()
        typing = ti_rtl(r, ti)
        t = TypeVar("t", "", ints=True, bools=True, floats=True,
                    simd=(4, 256))
        check_typing(typing, ({
            self.v0: t.as_bool(),
            self.v1: t,
            self.v2: t,
            self.v3: t,
            self.v4: t.half_vector(),
            self.v5: t.half_vector(),
            self.v6: t.half_vector().half_vector(),
            self.v7: t.half_vector().half_vector(),
        }, []))

    def test_vselect_vconcats(self):
        # type: () -> None
        r = Rtl(
                self.v3 << vselect(self.v0, self.v1, self.v2),
                self.v8 << vconcat(self.v3, self.v3),
                self.v9 << vconcat(self.v8, self.v8),
        )
        ti = TypeEnv()
        typing = ti_rtl(r, ti)
        t = TypeVar("t", "", ints=True, bools=True, floats=True,
                    simd=(2, 64))
        check_typing(typing, ({
            self.v0: t.as_bool(),
            self.v1: t,
            self.v2: t,
            self.v3: t,
            self.v8: t.double_vector(),
            self.v9: t.double_vector().double_vector(),
        }, []))

    def test_vselect_vsplits_vconcats(self):
        # type: () -> None
        r = Rtl(
                self.v3 << vselect(self.v0, self.v1, self.v2),
                (self.v4, self.v5) << vsplit(self.v3),
                (self.v6, self.v7) << vsplit(self.v4),
                self.v8 << vconcat(self.v3, self.v3),
                self.v9 << vconcat(self.v8, self.v8),
        )
        ti = TypeEnv()
        typing = ti_rtl(r, ti)
        t = TypeVar("t", "", ints=True, bools=True, floats=True,
                    simd=(4, 64))
        check_typing(typing, ({
            self.v0: t.as_bool(),
            self.v1: t,
            self.v2: t,
            self.v3: t,
            self.v4: t.half_vector(),
            self.v5: t.half_vector(),
            self.v6: t.half_vector().half_vector(),
            self.v7: t.half_vector().half_vector(),
            self.v8: t.double_vector(),
            self.v9: t.double_vector().double_vector(),
        }, []))

    def test_bint(self):
        # type: () -> None
        r = Rtl(
            self.v4 << iadd(self.v1, self.v2),
            self.v5 << bint(self.v3),
            self.v0 << iadd(self.v4, self.v5)
        )
        ti = TypeEnv()
        typing = ti_rtl(r, ti)
        itype = TypeVar("t", "", ints=True, simd=(1, 256))
        btype = TypeVar("b", "", bools=True, simd=True)

        # Check that self.v5 gets the same integer type as
        # the rest of them
        # TODO: Add constraint nlanes(v3) == nlanes(v1) when we
        # add that type constraint to bint
        check_typing(typing, ({
            self.v1:    itype,
            self.v2:    itype,
            self.v4:    itype,
            self.v5:    itype,
            self.v3:    btype,
            self.v0:    itype,
        }, []))

    def test_fully_bound_inst_inference_bad(self):
        # type: () -> None
        # Incompatible bound instructions fail accordingly
        r = Rtl(
                self.v3 << uextend.i32(self.v1),
                self.v4 << uextend.i16(self.v2),
                self.v5 << iadd(self.v3, self.v4),
            )
        ti = TypeEnv()
        typing = ti_rtl(r, ti)

        self.assertEqual(typing,
                         "On line 2: fail ti on `typeof_v4` <: `4`: " +
                         "Error: empty type created when unifying " +
                         "`typeof_v4` and `typeof_v5`")

    def test_extend_reduce(self):
        # type: () -> None
        r = Rtl(
            self.v1 << uextend(self.v0),
            self.v2 << ireduce(self.v1),
            self.v3 << sextend(self.v2),
        )
        ti = TypeEnv()
        typing = get_type_env(ti_rtl(r, ti))
        typing = typing.extract()

        itype0 = TypeVar("t", "", ints=True, simd=(1, 256))
        itype1 = TypeVar("t1", "", ints=True, simd=(1, 256))
        itype2 = TypeVar("t2", "", ints=True, simd=(1, 256))
        itype3 = TypeVar("t3", "", ints=True, simd=(1, 256))

        check_typing(typing, ({
            self.v0:    itype0,
            self.v1:    itype1,
            self.v2:    itype2,
            self.v3:    itype3,
        }, [WiderOrEq(itype1, itype0),
            WiderOrEq(itype1, itype2),
            WiderOrEq(itype3, itype2)]))

    def test_extend_reduce_enumeration(self):
        # type: () -> None
        for op in (uextend, sextend, ireduce):
            r = Rtl(
                self.v1 << op(self.v0),
            )
            ti = TypeEnv()
            typing = get_type_env(ti_rtl(r, ti)).extract()

            # The number of possible typings is 9 * (3+ 2*2 + 3) = 90
            l = [(t[self.v0], t[self.v1]) for t in typing.concrete_typings()]
            assert (len(l) == len(set(l)) and len(l) == 90)
            for (tv0, tv1) in l:
                typ0, typ1 = (tv0.singleton_type(), tv1.singleton_type())
                if (op == ireduce):
                    assert typ0.wider_or_equal(typ1)
                else:
                    assert typ1.wider_or_equal(typ0)

    def test_fpromote_fdemote(self):
        # type: () -> None
        r = Rtl(
            self.v1 << fpromote(self.v0),
            self.v2 << fdemote(self.v1),
        )
        ti = TypeEnv()
        typing = ti_rtl(r, ti)
        typing = get_type_env(typing).extract()

        ftype0 = TypeVar("t", "", floats=True, simd=(1, 256))
        ftype1 = TypeVar("t1", "", floats=True, simd=(1, 256))
        ftype2 = TypeVar("t2", "", floats=True, simd=(1, 256))

        check_typing(typing, ({
            self.v0:    ftype0,
            self.v1:    ftype1,
            self.v2:    ftype2,
        }, [WiderOrEq(ftype1, ftype0),
            WiderOrEq(ftype1, ftype2)]))

    def test_fpromote_fdemote_enumeration(self):
        # type: () -> None
        for op in (fpromote, fdemote):
            r = Rtl(
                self.v1 << op(self.v0),
            )
            ti = TypeEnv()
            typing = get_type_env(ti_rtl(r, ti)).extract()

            # The number of possible typings is 9*(2 + 1) = 27
            l = [(t[self.v0], t[self.v1]) for t in typing.concrete_typings()]
            assert (len(l) == len(set(l)) and len(l) == 27)
            for (tv0, tv1) in l:
                (typ0, typ1) = (tv0.singleton_type(), tv1.singleton_type())
                if (op == fdemote):
                    assert typ0.wider_or_equal(typ1)
                else:
                    assert typ1.wider_or_equal(typ0)


class TestXForm(TypeCheckingBaseTest):
    def test_iadd_cout(self):
        # type: () -> None
        x = XForm(Rtl((self.v0, self.v1) << iadd_cout(self.v2, self.v3),),
                  Rtl(
                      self.v0 << iadd(self.v2, self.v3),
                      self.v1 << icmp(intcc.ult, self.v0, self.v2)
                  ))
        itype = TypeVar("t", "", ints=True, simd=(1, 1))

        check_typing(x.ti, ({
            self.v0:    itype,
            self.v2:    itype,
            self.v3:    itype,
            self.v1:    itype.as_bool(),
        }, []), x.symtab)

    def test_iadd_cin(self):
        # type: () -> None
        x = XForm(Rtl(self.v0 << iadd_cin(self.v1, self.v2, self.v3)),
                  Rtl(
                      self.v4 << iadd(self.v1, self.v2),
                      self.v5 << bint(self.v3),
                      self.v0 << iadd(self.v4, self.v5)
                  ))
        itype = TypeVar("t", "", ints=True, simd=(1, 1))

        check_typing(x.ti, ({
            self.v0:    itype,
            self.v1:    itype,
            self.v2:    itype,
            self.v3:    self.b1,
            self.v4:    itype,
            self.v5:    itype,
        }, []), x.symtab)

    def test_enumeration_with_constraints(self):
        # type: () -> None
        xform = XForm(
            Rtl(
                self.v0 << iconst(self.imm0),
                self.v1 << icmp(intcc.eq, self.v2, self.v0),
                self.v5 << vselect(self.v1, self.v3, self.v4)
            ),
            Rtl(
                self.v0 << iconst(self.imm0),
                self.v1 << icmp(intcc.eq, self.v2, self.v0),
                self.v5 << vselect(self.v1, self.v3, self.v4)
            ))

        # Check all var assigns are correct
        assert len(xform.ti.constraints) > 0
        concrete_var_assigns = list(xform.ti.concrete_typings())

        v0 = xform.symtab[str(self.v0)]
        v1 = xform.symtab[str(self.v1)]
        v2 = xform.symtab[str(self.v2)]
        v3 = xform.symtab[str(self.v3)]
        v4 = xform.symtab[str(self.v4)]
        v5 = xform.symtab[str(self.v5)]

        for var_m in concrete_var_assigns:
            assert var_m[v0] == var_m[v2] and \
                   var_m[v3] == var_m[v4] and\
                   var_m[v5] == var_m[v3] and\
                   var_m[v1] == var_m[v2].as_bool() and\
                   var_m[v1].get_typeset() == var_m[v3].as_bool().get_typeset()
            check_concrete_typing_xform(var_m, xform)

        # The number of possible typings here is:
        # 8 cases for v0 = i8xN times 2 options for v3 - i8, b8 = 16
        # 8 cases for v0 = i16xN times 2 options for v3 - i16, b16 = 16
        # 8 cases for v0 = i32xN times 3 options for v3 - i32, b32, f32 = 24
        # 8 cases for v0 = i64xN times 3 options for v3 - i64, b64, f64 = 24
        #
        # (Note we have 8 cases for lanes since vselect prevents scalars)
        # Total: 2*16 + 2*24 = 80
        assert len(concrete_var_assigns) == 80

    def test_base_legalizations_enumeration(self):
        # type: () -> None
        for xform in narrow.xforms + expand.xforms:
            # Any legalization patterns we defined should have at least 1
            # concrete typing
            concrete_typings_list = list(xform.ti.concrete_typings())
            assert len(concrete_typings_list) > 0

            # If there are no free_typevars, this is a non-polymorphic pattern.
            # There should be only one possible concrete typing.
            if (len(xform.ti.free_typevars()) == 0):
                assert len(concrete_typings_list) == 1
                continue

            # For any patterns where the type env includes constraints, at
            # least one of the "theoretically possible" concrete typings must
            # be prevented by the constraints. (i.e. we are not emitting
            # unneccessary constraints).
            # We check that by asserting that the number of concrete typings is
            # less than the number of all possible free typevar assignments
            if (len(xform.ti.constraints) > 0):
                theoretical_num_typings =\
                    reduce(lambda x, y:    x*y,
                           [tv.get_typeset().size()
                            for tv in xform.ti.free_typevars()], 1)
                assert len(concrete_typings_list) < theoretical_num_typings

            # Check the validity of each individual concrete typing against the
            # xform
            for concrete_typing in concrete_typings_list:
                check_concrete_typing_xform(concrete_typing, xform)

    def test_bound_inst_inference(self):
        # type: () -> None
        # First example from issue #26
        x = XForm(
            Rtl(
                self.v0 << iadd(self.v1, self.v2),
            ),
            Rtl(
                self.v3 << uextend.i32(self.v1),
                self.v4 << uextend.i32(self.v2),
                self.v5 << iadd(self.v3, self.v4),
                self.v0 << ireduce(self.v5)
            ))
        itype = TypeVar("t", "", ints=True, simd=True)
        i32t = TypeVar.singleton(i32)

        check_typing(x.ti, ({
            self.v0:    itype,
            self.v1:    itype,
            self.v2:    itype,
            self.v3:    i32t,
            self.v4:    i32t,
            self.v5:    i32t,
        }, [WiderOrEq(i32t, itype)]), x.symtab)

    def test_bound_inst_inference1(self):
        # type: () -> None
        # Second example taken from issue #26
        x = XForm(
            Rtl(
                self.v0 << iadd(self.v1, self.v2),
            ),
            Rtl(
                self.v3 << uextend(self.v1),
                self.v4 << uextend(self.v2),
                self.v5 << iadd.i32(self.v3, self.v4),
                self.v0 << ireduce(self.v5)
            ))
        itype = TypeVar("t", "", ints=True, simd=True)
        i32t = TypeVar.singleton(i32)

        check_typing(x.ti, ({
            self.v0:    itype,
            self.v1:    itype,
            self.v2:    itype,
            self.v3:    i32t,
            self.v4:    i32t,
            self.v5:    i32t,
        }, [WiderOrEq(i32t, itype)]), x.symtab)

    def test_fully_bound_inst_inference(self):
        # type: () -> None
        # Second example taken from issue #26 with complete bounds
        x = XForm(
            Rtl(
                self.v0 << iadd(self.v1, self.v2),
            ),
            Rtl(
                self.v3 << uextend.i32.i8(self.v1),
                self.v4 << uextend.i32.i8(self.v2),
                self.v5 << iadd(self.v3, self.v4),
                self.v0 << ireduce(self.v5)
            ))
        i8t = TypeVar.singleton(i8)
        i32t = TypeVar.singleton(i32)

        # Note no constraints here since they are all trivial
        check_typing(x.ti, ({
            self.v0:    i8t,
            self.v1:    i8t,
            self.v2:    i8t,
            self.v3:    i32t,
            self.v4:    i32t,
            self.v5:    i32t,
        }, []), x.symtab)

    def test_fully_bound_inst_inference_bad(self):
        # type: () -> None
        # Can't force a mistyped XForm using bound instructions
        with self.assertRaises(AssertionError):
            XForm(
                Rtl(
                    self.v0 << iadd(self.v1, self.v2),
                ),
                Rtl(
                    self.v3 << uextend.i32.i8(self.v1),
                    self.v4 << uextend.i32.i16(self.v2),
                    self.v5 << iadd(self.v3, self.v4),
                    self.v0 << ireduce(self.v5)
                ))

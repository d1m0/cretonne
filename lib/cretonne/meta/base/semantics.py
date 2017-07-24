from __future__ import absolute_import
from semantics.primitives import prim_to_bv, prim_from_bv, bvsplit, bvconcat,\
    bvadd, prim_int2bv
from .instructions import vsplit, vconcat, iadd, iconst
from cdsl.xform import Rtl
from cdsl.ast import Var
from cdsl.typevar import TypeSet
from cdsl.ti import InTypeset

x = Var('x')
y = Var('y')
a = Var('a')
xhi = Var('xhi')
yhi = Var('yhi')
ahi = Var('ahi')
xlo = Var('xlo')
ylo = Var('ylo')
alo = Var('alo')
lo = Var('lo')
hi = Var('hi')
bvx = Var('bvx')
bvy = Var('bvy')
bva = Var('bva')
bvlo = Var('bvlo')
bvhi = Var('bvhi')

ScalarTS = TypeSet(lanes=(1, 1), ints=True, floats=True, bools=True)

iconst.set_semantics(
    a << iconst(x),
    Rtl(
        bva << prim_int2bv(x),
        a << prim_from_bv(bva)
    ))

vsplit.set_semantics(
    (lo, hi) << vsplit(x),
    Rtl(
        bvx << prim_to_bv(x),
        (bvlo, bvhi) << bvsplit(bvx),
        lo << prim_from_bv(bvlo),
        hi << prim_from_bv(bvhi)
    ))

vconcat.set_semantics(
    x << vconcat(lo, hi),
    Rtl(
        bvlo << prim_to_bv(lo),
        bvhi << prim_to_bv(hi),
        bvx << bvconcat(bvlo, bvhi),
        x << prim_from_bv(bvx)
    ))

iadd.set_semantics(
    a << iadd(x, y),
    (Rtl(bvx << prim_to_bv(x),
         bvy << prim_to_bv(y),
         bva << bvadd(bvx, bvy),
         a << prim_from_bv(bva)),
     [InTypeset(x.get_typevar(), ScalarTS)]),
    Rtl((xlo, xhi) << vsplit(x),
        (ylo, yhi) << vsplit(y),
        alo << iadd(xlo, ylo),
        ahi << iadd(xhi, yhi),
        a << vconcat(alo, ahi)))

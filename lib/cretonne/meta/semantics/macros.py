"""
Useful semantics "macro" instructions built on top of
the primitives.
"""
from __future__ import absolute_import
from cdsl.operands import Operand
from cdsl.typevar import TypeVar, MAX_BITVEC
from cdsl.instructions import Instruction, InstructionGroup
from base.types import b1, MemType
from base.immediates import imm64
from cdsl.ast import Var
from cdsl.ti import TypesEqual, WiderOrEq
from cdsl.xform import Rtl, XForm
from cdsl.formats import InstructionFormat
from cdsl.operands import VALUE
from semantics.primitives import bv_from_imm64, bvite, bvselect, bvadd, \
        bvconcat, bvurem, bveq, bvwidth, bvcontains, prim_and, bvsplit
import base.formats # noqa
import semantics.formats # noqa

GROUP = InstructionGroup("primitive_macros", "Semantic macros instruction set")
AnyBV = TypeVar('AnyBV', bitvecs=True, doc="")
x = Var('x')
y = Var('y')
imm = Var('imm')
a = Var('a')

#
# Immediate arithmetic
#
x_op = Operand('x_op', AnyBV, doc="")
a_op = Operand('a_op', AnyBV, doc="")
imm_op = Operand('imm_op', imm64)
cond_op = Operand("cond", b1, doc="")

bvadd_imm = Instruction(
        'bvadd_imm', r"""
        Macro around bvadd where the 2nd argument is anoimmediate.""",
        ins=(x_op, imm_op), outs=a_op)

bvadd_imm.set_semantics(
        a << bvadd_imm(x, imm),
        Rtl(
            y << bv_from_imm64(imm),
            a << bvadd(x, y)
        )
)

bvurem_imm = Instruction(
        'bvmod_imm', r"""
        Macro around bvmod where the 2nd argument is anoimmediate.""",
        ins=(x_op, imm_op), outs=a_op)

bvurem_imm.set_semantics(
        a << bvurem_imm(x, imm),
        Rtl(
            y << bv_from_imm64(imm),
            a << bvurem(x, y)
        )
)

bveq_imm = Instruction(
        'bveq_imm', r"""
        Macro around bveq where the 2nd argument is anoimmediate.""",
        ins=(x_op, imm_op), outs=cond_op)

bveq_imm.set_semantics(
        a << bveq_imm(x, imm),
        Rtl(
            y << bv_from_imm64(imm),
            a << bveq(x, y)
        )
)

#
# Bool-to-bv1
#
BV1 = TypeVar("BV1", bitvecs=(1, 1), doc="")
bv1_op = Operand('bv1_op', BV1, doc="")
cond_op = Operand("cond", b1, doc="")
bool2bv = Instruction(
        'bool2bv', r"""Convert a b1 value to a 1-bit BV""",
        ins=cond_op, outs=bv1_op)

v1 = Var('v1')
v2 = Var('v2')
bvone = Var('bvone')
bvzero = Var('bvzero')
bool2bv.set_semantics(
        v1 << bool2bv(v2),
        Rtl(
            bvone << bv_from_imm64(imm64(1)),
            bvzero << bv_from_imm64(imm64(0)),
            v1 << bvite(v2, bvone, bvzero)
        ))


#
# Memory macros (load/store multibyte bitvectors)
#


AnyMem = TypeVar('AnyMem', '', memories=True)
# TODO: Remove Mem and generalize
Mem = TypeVar.singleton(MemType.with_bits(64, 8))
MemBV = TypeVar('MemBV', bitvecs=(8, MAX_BITVEC), doc="")

mem_op = Operand('mem_op', Mem, doc="A semantic value representing a memory")
addr_op = Operand('addr_op', Mem.domain(), doc="A semantic value addr")
val_op = Operand('val_op', MemBV, doc="A semantic value for a memory cell")

bvselect_wide = Instruction(
        'bvselect_wide', r""" """, ins=(mem_op, addr_op),
        outs=val_op, constraints=WiderOrEq(MemBV, Mem.range()))


addr = Var('addr')
mem = Var('mem', AnyMem)
val = Var('val')
lo = Var('lo')
hi = Var('hi')
hi_addr = Var('hi_addr')
half_width = Var('half_width')

bvselect_wide.set_semantics(
        val << bvselect_wide(mem, addr),
        XForm(
            Rtl(
                val << bvselect_wide(mem, addr),
            ),
            Rtl(
                val << bvselect(mem, addr),
            ), constraints=[TypesEqual(val.get_typevar(),
                                       mem.get_typevar().range())]),
        XForm(
            Rtl(
                val << bvselect_wide(mem, addr),
            ),
            Rtl(
                lo << bvselect_wide(mem, addr),
                half_width << bvwidth(lo),
                hi_addr << bvadd(addr, half_width),
                hi << bvselect_wide(mem, hi_addr),
                val << bvconcat(lo, hi)
            )
        ))


bvaligned = Instruction(
        'bvaligned', r""" """, ins=(x_op, val_op),
        outs=cond_op)

cond = Var('cond')
t = Var('t')
width = Var('width')

bvaligned.set_semantics(
    cond << bvaligned(addr, val),
    Rtl(
        width << bvwidth(val),
        t << bvurem(addr, width),
        cond << bveq_imm(t, imm64(0))
    ))


# Add dummy op to distinguish from ternary instructions where
# the second argument is alway assumed to be ctrl typevar
bvcontains_fmt = InstructionFormat(VALUE, VALUE, VALUE, imm64)
dummy_op = Operand('dummy_op', imm64)
bvcontains_wide = Instruction(
        'bvcontains_wide', r""" """, ins=(mem_op, addr_op, val_op, dummy_op),
        outs=cond_op, constraints=WiderOrEq(MemBV, Mem.range()))


addr = Var('addr')
mem = Var('mem', AnyMem)
val = Var('val')
contains = Var('contains')
lo = Var('lo')
hi = Var('hi')
hi_addr = Var('hi_addr')
half_width = Var('half_width')
dummy = Var('dummy')

lo_val = Var('lo_val')
hi_val = Var('hi_val')

bvcontains_wide.set_semantics(
        contains << bvcontains_wide(mem, addr, val, dummy),
        XForm(
            Rtl(
                contains << bvcontains_wide(mem, addr, val, dummy),
            ),
            Rtl(
                contains << bvcontains(mem, addr),
            ), constraints=[TypesEqual(val.get_typevar(),
                                       mem.get_typevar().range())]),
        XForm(
            Rtl(
                contains << bvcontains_wide(mem, addr, val, dummy),
            ),
            Rtl(
                (lo_val, hi_val) << bvsplit(val),
                lo << bvcontains_wide(mem, addr, lo_val, dummy),
                half_width << bvwidth(lo_val),
                hi_addr << bvadd(addr, half_width),
                hi << bvcontains_wide(mem, hi_addr, hi_val, dummy),
                contains << prim_and(lo, hi)
            )
        ))

GROUP.close()

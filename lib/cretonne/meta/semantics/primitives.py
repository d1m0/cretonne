"""
Cretonne primitive instruction set.

This module defines a primitive instruction set, in terms of which the base set
is described. Most instructions in this set correspond 1-1 with an SMTLIB
bitvector function.
"""
from __future__ import absolute_import
from cdsl.operands import Operand
from cdsl.typevar import TypeVar
from cdsl.instructions import Instruction, InstructionGroup
from cdsl.ti import WiderOrEq
from cdsl.types import MemType
from base.types import b1
from base.immediates import offset32, imm64
import base.formats # noqa
import semantics.formats # noqa

GROUP = InstructionGroup("primitive", "Primitive instruction set")

BV = TypeVar('BV', 'A bitvector type.', bitvecs=True)
OtherBV = TypeVar('OtherBV', 'A bitvector type.', bitvecs=True)
BV1 = TypeVar('BV1', 'A single bit bitvector.', bitvecs=(1, 1))
Real = TypeVar('Real', 'Any real type.', ints=True, floats=True,
               bools=True, simd=True)

x = Operand('x', BV, doc="A semantic value X")
y = Operand('x', BV, doc="A semantic value Y (same width as X)")
a = Operand('a', BV, doc="A semantic value A (same width as X)")
cond = Operand('b', TypeVar.singleton(b1), doc='A b1 value')

real = Operand('real', Real, doc="A real cretonne value")
fromReal = Operand('fromReal', Real.to_bitvec(),
                   doc="A real cretonne value converted to a BV")

#
# BV Conversion/Materialization
#
prim_to_bv = Instruction(
        'prim_to_bv', r"""
        Convert an SSA Value to a flat bitvector
        """,
        ins=real, outs=fromReal)

prim_from_bv = Instruction(
        'prim_from_bv', r"""
        Convert a flat bitvector to a real SSA Value.
        """,
        ins=fromReal, outs=real)

off = Operand('off', offset32)
bv_from_offset32 = Instruction(
        'bv_from_off', r"""Materialize an offset32 as a bitvector.""",
        ins=(off), outs=a)

N = Operand('N', imm64)
bv_from_imm64 = Instruction(
        'bv_from_imm64', r"""Materialize an imm64 as a bitvector.""",
        ins=(N), outs=a)

#
# Generics
#
bvite = Instruction(
        'bvite', r"""Bitvector ternary operator""",
        ins=(cond, x, y), outs=a)

#
# BV Memory Ops
#
Mem = TypeVar.singleton(MemType.with_bits(64, 8))
MemTo = TypeVar.singleton(MemType.with_bits(64, 8))

mem = Operand('mem', Mem, doc="A semantic value representing a memory")
memTo = Operand('memTo', MemTo, doc="A semantic value representing a memory")
addr = Operand('addr', Mem.domain(), doc="A semantic value addr")
val = Operand('val', Mem.range(), doc="A semantic value for a memory cell")

bvselect = Instruction(
        'bvselect', r"""Lookup in an array """,
        ins=(mem, addr), outs=val)

bvstore = Instruction(
        'bvstore', r"""Store in an array """,
        ins=(mem, addr, val), outs=memTo)

bvcontains = Instruction(
        'bvcontains', r"""
        Predicate asserting that a value is 'mapped' in an array""",
        ins=(mem, addr), outs=cond)

N = Operand('N', imm64)
bv_from_imm64 = Instruction(
        'bv_from_imm64', r"""Materialize an imm64 as a bitvector.""",
        ins=(N), outs=a)

#
# Generics
#
bvite = Instruction(
        'bvite', r"""Bitvector ternary operator""",
        ins=(cond, x, y), outs=a)


xh = Operand('xh', BV.half_width(),
             doc="A semantic value representing the upper half of X")
xl = Operand('xl', BV.half_width(),
             doc="A semantic value representing the lower half of X")

#
# Width manipulation bvs
#
bvsplit = Instruction(
        'bvsplit', r"""
        """,
        ins=(x), outs=(xh, xl))

xy = Operand('xy', BV.double_width(),
             doc="A semantic value representing the concatenation of X and Y")
bvconcat = Instruction(
        'bvconcat', r"""
        """,
        ins=(x, y), outs=xy)
#
# Extensions
#
ToBV = TypeVar('ToBV', 'A bitvector type.', bitvecs=True)
x1 = Operand('x1', ToBV, doc="")

bvzeroext = Instruction(
        'bvzeroext', r"""Unsigned bitvector extension""",
        ins=x, outs=x1, constraints=WiderOrEq(ToBV, BV))

bvsignext = Instruction(
        'bvsignext', r"""Signed bitvector extension""",
        ins=x, outs=x1, constraints=WiderOrEq(ToBV, BV))


#
# Arithmetic ops
#
bvadd = Instruction(
        'bvadd', r"""
        Standard 2's complement addition. Equivalent to wrapping integer
        addition: :math:`a := x + y \pmod{2^B}`.

        This instruction does not depend on the signed/unsigned interpretation
        of the operands.
        """,
        ins=(x, y), outs=a)

bvsub = Instruction(
        'bvadd', r""" Standard 2's complement subtraction.  """,
        ins=(x, y), outs=a)

bvurem = Instruction(
        'bvurem', r"""Usigned division reminder.  """, ins=(x, y), outs=a)
#
# Bitwise ops
#

bvand = Instruction('bvand', r""" Bitwise and""", ins=(x, y), outs=a)

#
# Logical ops. Unlike the corresponding boolean ops in base/instructions.py
# these only operate over b1.
#
c1 = Operand('c1', TypeVar.singleton(b1), doc='A b1 value')
c2 = Operand('c2', TypeVar.singleton(b1), doc='A b1 value')
c3 = Operand('c3', TypeVar.singleton(b1), doc='A b1 value')
prim_and = Instruction('prim_and', r"""Logical and""", ins=(c1, c2), outs=c3)
prim_or = Instruction('prim_or', r"""Logical or""", ins=(c1, c2), outs=c3)
prim_not = Instruction('prim_not', r"""Logical not""", ins=(c1), outs=c3)

#
# Bitvector comparisons
#

bveq = Instruction(
        'bveq', r"""Unsigned bitvector equality""",
        ins=(x, y), outs=cond)
bvne = Instruction(
        'bveq', r"""Unsigned bitvector inequality""",
        ins=(x, y), outs=cond)
bvsge = Instruction(
        'bvsge', r"""Signed bitvector greater or equal""",
        ins=(x, y), outs=cond)
bvsgt = Instruction(
        'bvsgt', r"""Signed bitvector greater than""",
        ins=(x, y), outs=cond)
bvsle = Instruction(
        'bvsle', r"""Signed bitvector less than or equal""",
        ins=(x, y), outs=cond)
bvslt = Instruction(
        'bvslt', r"""Signed bitvector less than""",
        ins=(x, y), outs=cond)
bvuge = Instruction(
        'bvuge', r"""Unsigned bitvector greater or equal""",
        ins=(x, y), outs=cond)
bvugt = Instruction(
        'bvugt', r"""Unsigned bitvector greater than""",
        ins=(x, y), outs=cond)
bvule = Instruction(
        'bvule', r"""Unsigned bitvector less than or equal""",
        ins=(x, y), outs=cond)
bvult = Instruction(
        'bvult', r"""Unsigned bitvector less than""",
        ins=(x, y), outs=cond)

<<<<<<< HEAD
# Extensions
ToBV = TypeVar('ToBV', 'A bitvector type.', bitvecs=True)
x1 = Operand('x1', ToBV, doc="")

bvzeroext = Instruction(
        'bvzeroext', r"""Unsigned bitvector extension""",
        ins=x, outs=x1, constraints=WiderOrEq(ToBV, BV))

bvsignext = Instruction(
        'bvsignext', r"""Signed bitvector extension""",
        ins=x, outs=x1, constraints=WiderOrEq(ToBV, BV))
=======
#
# Other
#
b = Operand('b', OtherBV, doc="A semantic value Z (different type from X)")
bvwidth = Instruction(
        'bvwidth', r"""
        This is a compile-time constant corresponding to the width of the input
        BV type. Similar to iconst.
        """,
        ins=x, outs=b)

bvrand = Instruction(
        'bvrand', r"""
        Returns a random bitvector. Used to modle unspecified behavior.
        """,
        outs=b)
>>>>>>> Kinda-sorta getting ld-ld comparison queries
GROUP.close()

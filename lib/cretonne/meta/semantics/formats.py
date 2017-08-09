"""
The semantics.formats defines all instruction formats that appear only in
semantics definitions. The correpsonding instructions are not emittable (unlike
base.formats).
"""
from __future__ import absolute_import
from cdsl.formats import InstructionFormat
from base.immediates import offset32

UnaryOffset32 = InstructionFormat(offset32)

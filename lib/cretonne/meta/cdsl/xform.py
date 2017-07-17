"""
Instruction transformations.
"""
from __future__ import absolute_import
from .ast import Def, Var, Apply
from .ti import ti_xform, TypeEnv, get_type_env

try:
    from typing import Union, Iterator, Sequence, Iterable, List, Dict  # noqa
    from typing import Optional  # noqa
    from .ast import Expr  # noqa
    from .ti import TypeConstraint  # noqa
    from .typevar import TypeVar  # noqa
    DefApply = Union[Def, Apply]
except ImportError:
    pass


def canonicalize_defapply(node):
    # type: (DefApply) -> Def
    """
    Canonicalize a `Def` or `Apply` node into a `Def`.

    An `Apply` becomes a `Def` with an empty list of defs.
    """
    if isinstance(node, Apply):
        return Def((), node)
    else:
        return node


class Rtl(object):
    """
    Register Transfer Language list.

    An RTL object contains a list of register assignments in the form of `Def`
    objects.

    An RTL list can represent both a source pattern to be matched, or a
    destination pattern to be inserted.
    """

    def __init__(self, *args):
        # type: (*DefApply) -> None
        self.rtl = tuple(map(canonicalize_defapply, args))

    def copy(self, m):
        # type: (Dict[Var, Var]) -> Rtl
        """
        Return a copy of this rtl with all Vars substituted with copies or
        according to m. Update m as neccessary.
        """
        return Rtl(*[d.copy(m) for d in self.rtl])


class XForm(object):
    """
    An instruction transformation consists of a source and destination pattern.

    Patterns are expressed in *register transfer language* as tuples of
    `ast.Def` or `ast.Expr` nodes.

    A legalization pattern must have a source pattern containing only a single
    instruction.

    >>> from base.instructions import iconst, iadd, iadd_imm
    >>> a = Var('a')
    >>> c = Var('c')
    >>> v = Var('v')
    >>> x = Var('x')
    >>> XForm(
    ...     Rtl(c << iconst(v),
    ...         a << iadd(x, c)),
    ...     Rtl(a << iadd_imm(x, v)))
    XForm(inputs=[Var(v), Var(x)], defs=[Var(c, src), Var(a, src, dst)],
      c << iconst(v)
      a << iadd(x, c)
    =>
      a << iadd_imm(x, v)
    )
    """

    def __init__(self, src, dst, constraints=None):
        # type: (Rtl, Rtl, Optional[Sequence[TypeConstraint]]) -> None
        self.src = src
        self.dst = dst
        # Variables that are inputs to the source pattern.
        self.inputs = list()  # type: List[Var]
        # Variables defined in either src or dst.
        self.defs = list()  # type: List[Var]

        # Rewrite variables in src and dst RTL lists to our own copies.
        # Map name -> private Var.
        symtab = dict()  # type: Dict[str, Var]
        self._rewrite_rtl(src, symtab, Var.SRCCTX)
        num_src_inputs = len(self.inputs)
        self._rewrite_rtl(dst, symtab, Var.DSTCTX)
        # Needed for testing type inference on XForms
        self.symtab = symtab

        # Check for inconsistently used inputs.
        for i in self.inputs:
            if not i.is_input():
                raise AssertionError(
                        "'{}' used as both input and def".format(i))

        # Check for spurious inputs in dst.
        if len(self.inputs) > num_src_inputs:
            raise AssertionError(
                    "extra inputs in dst RTL: {}".format(
                        self.inputs[num_src_inputs:]))

        # Perform type inference and cleanup
        raw_ti = get_type_env(ti_xform(self, TypeEnv()))
        raw_ti.normalize()
        self.ti = raw_ti.extract()

        def interp_tv(tv):
            # type: (TypeVar) -> TypeVar
            """ Convert typevars according to symtab """
            if not tv.name.startswith("typeof_"):
                return tv
            return symtab[tv.name[len("typeof_"):]].get_typevar()

        if constraints is not None:
            for c in constraints:
                type_m = {tv: interp_tv(tv) for tv in c.tvs()}
                self.ti.add_constraint(c.translate(type_m).translate(self.ti))

        # Sanity: The set of inferred free typevars should be a subset of the
        # TVs corresponding to Vars appearing in src
        free_typevars = set(self.ti.free_typevars())
        src_vars = set(self.inputs).union(
            [x for x in self.defs if not x.is_temp()])
        src_tvs = set([v.get_typevar() for v in src_vars])
        if (not free_typevars.issubset(src_tvs)):
            raise AssertionError(
                "Some free vars don't appear in src - {}"
                .format(free_typevars.difference(src_tvs)))

        # Update the type vars for each Var to their inferred values
        for v in self.inputs + self.defs:
            v.set_typevar(self.ti[v.get_typevar()])

    def __repr__(self):
        # type: () -> str
        s = "XForm(inputs={}, defs={},\n  ".format(self.inputs, self.defs)
        s += '\n  '.join(str(n) for n in self.src.rtl)
        s += '\n=>\n  '
        s += '\n  '.join(str(n) for n in self.dst.rtl)
        s += '\n)'
        return s

    def _rewrite_rtl(self, rtl, symtab, context):
        # type: (Rtl, Dict[str, Var], int) -> None
        for line in rtl.rtl:
            if isinstance(line, Def):
                line.defs = tuple(
                        self._rewrite_defs(line, symtab, context))
                expr = line.expr
            else:
                expr = line
            self._rewrite_expr(expr, symtab, context)

    def _rewrite_expr(self, expr, symtab, context):
        # type: (Apply, Dict[str, Var], int) -> None
        """
        Find all uses of variables in `expr` and replace them with our own
        local symbols.
        """

        # Accept a whole expression tree.
        stack = [expr]
        while len(stack) > 0:
            expr = stack.pop()
            expr.args = tuple(
                    self._rewrite_uses(expr, stack, symtab, context))

    def _rewrite_defs(self, line, symtab, context):
        # type: (Def, Dict[str, Var], int) -> Iterable[Var]
        """
        Given a tuple of symbols defined in a Def, rewrite them to local
        symbols. Yield the new locals.
        """
        for sym in line.defs:
            name = str(sym)
            if name in symtab:
                var = symtab[name]
                if var.get_def(context):
                    raise AssertionError("'{}' multiply defined".format(name))
            else:
                var = Var(name)
                symtab[name] = var
                self.defs.append(var)
            var.set_def(context, line)
            yield var

    def _rewrite_uses(self, expr, stack, symtab, context):
        # type: (Apply, List[Apply], Dict[str, Var], int) -> Iterable[Expr]
        """
        Given an `Apply` expr, rewrite all uses in its arguments to local
        variables. Yield a sequence of new arguments.

        Append any `Apply` arguments to `stack`.
        """
        for arg, operand in zip(expr.args, expr.inst.ins):
            # Nested instructions are allowed. Visit recursively.
            if isinstance(arg, Apply):
                stack.append(arg)
                yield arg
                continue
            if not isinstance(arg, Var):
                assert not operand.is_value(), "Value arg must be `Var`"
                yield arg
                continue
            # This is supposed to be a symbolic value reference.
            name = str(arg)
            if name in symtab:
                var = symtab[name]
                # The variable must be used consistently as a def or input.
                if not var.is_input() and not var.get_def(context):
                    raise AssertionError(
                            "'{}' used as both input and def"
                            .format(name))
            else:
                # First time use of variable.
                var = Var(name)
                symtab[name] = var
                self.inputs.append(var)
            yield var

    def verify_legalize(self):
        # type: () -> None
        """
        Verify that this is a valid legalization XForm.

        - The source pattern must describe a single instruction.
        - All values defined in the output pattern must be defined in the
          destination pattern.
        """
        assert len(self.src.rtl) == 1, "Legalize needs single instruction."
        for d in self.src.rtl[0].defs:
            if not d.is_output():
                raise AssertionError(
                        '{} not defined in dest pattern'.format(d))


class XFormGroup(object):
    """
    A group of related transformations.
    """

    def __init__(self, name, doc):
        # type: (str, str) -> None
        self.xforms = list()  # type: List[XForm]
        self.name = name
        self.__doc__ = doc

    def legalize(self, src, dst):
        # type: (Union[Def, Apply], Rtl) -> None
        """
        Add a legalization pattern to this group.

        :param src: Single `Def` or `Apply` to be legalized.
        :param dst: `Rtl` list of replacement instructions.
        """
        xform = XForm(Rtl(src), dst)
        xform.verify_legalize()
        self.xforms.append(xform)

"""Throwaway: gpu-module-to-binary{format=fatbin} on a re-parsed lowered module; ELF sha."""
import sys, hashlib
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler.jit_function import _create_mlir_context
src = open(sys.argv[1]).read()
for i in range(int(sys.argv[2])):
    with _create_mlir_context():
        m = ir.Module.parse(src)
        PassManager.parse('builtin.module(gpu-module-to-binary{format=fatbin opts=""})').run(m.operation)
        objs = []
        def visit(op):
            if op.name == "gpu.binary":
                for o in ir.ArrayAttr(op.attributes["objects"]):
                    objs.append(bytes(gpu.ObjectAttr(o).object))
            return ir.WalkResult.ADVANCE
        m.operation.walk(visit)
    print("REBIN", i, hashlib.sha256(objs[0]).hexdigest()[:16], len(objs[0]), flush=True)

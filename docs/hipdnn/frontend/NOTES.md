# hipDNN megakernel frontend validation

- [Frontend draft PR](https://github.com/ROCm/rocm-libraries/pull/16903), based on `develop` at `8e0433e4829cafa05ed6d425413e863aced7e9f3`.
- [Engine draft PR](https://github.com/ROCm/rocm-libraries/pull/15686). The engine is needed for launch, not for frontend build or tests.
- The C++ contract test takes the binary key from the first [Gemma4 manifest entry](https://github.com/ROCm/rocm-libraries/blob/bf3255d71eb/dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel/manifest.json#L2-L23). `schema` identifies the matcher schema. `nlayers` is a launch argument, not a manifest key.

## CPU validation

The CPU run builds the frontend and bindings without the engine. It packages and installs the wheel before it runs pytest with the GPU hidden. The [CPU test runner](scripts/run-cpu-tests.sh) prints the frontend, wheel build, and pytest exit codes. The [configure and build logs](logs/5449aeba62a/) contain the exact options and source revision. Local ROCm Clang 18 with libstdc++ 12 reports a deprecation from `std::stable_sort` in the existing autotune header. The local build uses `ENABLE_CLANG_TIDY=OFF` and keeps that diagnostic as a warning. The source does not change to work around it.

The [installed wheel suite](logs/5449aeba62a/wheel-tests.log) passed: 697 tests passed and 139 skipped. The [first C++ run](logs/5449aeba62a/frontend-tests.log) found that the test backend mock did not serialize graph JSON. The contract test now reads the constructed custom-op node directly. The Python test still checks serialized graph JSON with the real backend. The correction changes only the C++ test.

The [final C++ run](logs/1aa7c7b1cd/frontend-tests.log) passed all 2,667 tests, including both megakernel tests. [Changed-file pre-commit hooks](logs/1aa7c7b1cd/pre-commit.log) passed at the same revision.

## gfx942 validation

The GPU run stages a local three-way tree with the `develop` base, frontend, and engine. It does not push a merged branch. [The launcher](scripts/launch.sh) runs under tmux on the login node. It streams the source and the locked code-object archive to the worker. The [GPU runner](scripts/run-gpu.sh) configures `ENABLE_MEGAKERNEL_ENGINE=ON`, builds the provider and bindings, then starts [the Python fixture](scripts/run_standin.py).

[`IntegrationGpuMegakernel`](https://github.com/ROCm/rocm-libraries/blob/3bb8d27b9db/dnn-providers/hip-kernel-provider/src/integration_tests/kernel_ingestor_engine/IntegrationGpuMegakernel.cpp#L21-L30) defines a stand-in, not a model logits oracle. The Python fixture uses the same input table marker and expected output words. It compares every byte and prints SHA-256 of both buffers. This verifies the graph, plan, launch geometry, and kernel arguments. It does not verify model logits. The engine integration test has no model weights or expected logits hash.

Job `68605498` [passed](logs/390d38cee69/gfx942-68605498/stand-in-oracle.log) with frontend revision `390d38cee69f86af96dda7f4b92b39ff38b4a637` and engine revision `3bb8d27b9db9186e68281fd14c0dc4dc0c063903`. Actual and expected output were both 622,600 bytes with SHA-256 `19c61f5fddd3f9e4b3938147e4dd7e9d3cc246fa5703cad06ca8528d1333f817`. Later frontend commits change a Doxygen comment and the C++ contract test, but no execution code. The [first configure attempt](logs/390d38cee69/gfx942-68605470/configure-gpu.log) lacked `dnn-providers/cmake` in the archive. The passing job included that directory.
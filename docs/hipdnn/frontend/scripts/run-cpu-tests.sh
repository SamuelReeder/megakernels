#!/bin/bash
set -uo pipefail
build=$1
src=$2
logs=$build/evidence
venv=$build/venv
wheel=$src/projects/hipdnn/python/frontend_wheel_package
export LD_LIBRARY_PATH="$build/build/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=-1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HIPDNN_TEST_GOOD_PLUGIN_PATH="$build/build/lib/test_plugins/custom/libtest_good_plugin.so"
export UV_CACHE_DIR="$build/uv-cache"
export TMPDIR="$build/tmp"
mkdir -p "$TMPDIR"
(
    cd "$build/build/bin" || exit
    ./hipdnn_frontend_tests --gtest_output="json:$logs/frontend-tests.json"
) > "$logs/frontend-tests.log" 2>&1
cpp_rc=$?
(
    set -e
    "$venv/bin/python" "$wheel/pack_frontend_wheel.py" --build-dir "$build/build" --wheel-dir "$build/wheel_package"
    uv pip install --python "$venv/bin/python" "$build/wheel_package/"hipdnn_frontend-*.whl
) > "$logs/wheel-build.log" 2>&1
wheel_rc=$?
py_rc=1
if [ "$wheel_rc" -eq 0 ]; then
    "$venv/bin/python" -m pytest -q -ra --junitxml="$logs/wheel-tests.xml" "$wheel/tests" > "$logs/wheel-tests.log" 2>&1
    py_rc=$?
fi
printf 'frontend tests exit=%d\nwheel build/install exit=%d\nwheel tests exit=%d\n' "$cpp_rc" "$wheel_rc" "$py_rc"
tail -14 "$logs/frontend-tests.log"
tail -14 "$logs/wheel-build.log"
if [ -f "$logs/wheel-tests.log" ]; then tail -24 "$logs/wheel-tests.log"; fi
exit $(( cpp_rc || wheel_rc || py_rc ))

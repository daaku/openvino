# openvino

cgo bindings for the OpenVINO C API. `ov.go` is the only cgo file. It wraps
just enough of the C API to load models, run inference and move tensor data,
for whispy (Parakeet speech to text) and eg2npu (EmbeddingGemma 2 embeddings).
Tensor data is copied in and out of OpenVINO owned memory: never hand C a
pointer to Go memory that outlives the call, and prefer Go over C for anything
that is not an API call.

- OpenVINO's core must never be freed: `ov_core_free` unloads the plugins
  while OpenVINO's static state and worker threads are still around, which
  segfaults. `ov_shutdown()` only moves that crash to the next create/teardown
  cycle. `SharedCore()` keeps one core per process, and each library's `Close`
  releases only its compiled models and infer requests.
- `CompileWith` passes plugin properties through
  `ov_core_compile_model_from_file_props`, the array taking form, because cgo
  cannot call the variadic entry point at all. The array form has no limit on
  how many properties a compile carries (the NPU language model path needs six)
  and its size argument is the number of properties rather than arguments, so
  the `property_args_size` trap the variadic form has is gone. Property names
  are the C API aliases (`NPU_COMPILER_TYPE`, `CACHE_DIR`), not `ov::` names.
- `Tensor.F32` and `Tensor.I64` alias OpenVINO owned memory: the slice is only
  valid while the tensor is alive, and writing through it is how inputs are
  filled. They refuse a tensor of another element type, whose buffer the slice
  would overrun.
- Every C call goes through `status`, which takes it as a closure and runs it
  and the read of `ov_get_last_err_msg` on one locked OS thread: the message
  is thread local, and a goroutine can change threads between two cgo calls.
  Scalars go through `SetInt`, which handles both i32 and i64.
- `ReadModel` and `CompileModel` split reading an IR from compiling it, which
  is the only way to compile a graph that has been changed: the C API can
  compile a model object but cannot set an op attribute. `CompileModel` uses
  the array form too (`ov_core_compile_model_props`). Given separately, the
  weights file need not be named after the XML, so several graphs can share
  one.
- `Version()` and `Core.AvailableDevices()` exist to diagnose installs: a
  binary can link one install's core and load another's plugins, and a device
  missing from the list is a missing plugin, not a compile failure.
- `ov_shape_create` rejects a zero rank shape, so scalar tensors cannot be
  allocated. Use the tensor an infer request already owns instead.
- `PortShape` reports dynamic dimensions as -1 and `Dim` as 0, so callers pick
  their own fallback instead of tripping over OpenVINO's "to_shape was called
  on a dynamic shape" error.
- The bindings cannot build, reshape or export a model, or import a compiled
  one. Callers that need a graph of their own write it as IR (eg2npu does) and
  rely on `CACHE_DIR` to skip compiling it again.

## Build and test

- cgo links `libopenvino_c` (Arch: `openvino`). `go test ./...` needs only the
  CPU plugin and no model files.
- Releases are tags (`v0.1.0`); whispy and eg2npu require a tagged version.

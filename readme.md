# openvino

Minimal Go bindings for the [OpenVINO](https://docs.openvino.ai) C API: read
and compile models, run inference and move tensor data in and out.

```go
import "github.com/daaku/openvino"

core, err := openvino.SharedCore()
model, err := core.CompileWith("model.xml", "NPU", map[string]string{"CACHE_DIR": dir})
req, err := model.Request()
in, err := req.Get("input")
data, err := in.F32() // fill the input in place
err = req.Infer()
```

It needs `libopenvino_c` at build and run time, plus the plugin of each device
used (`openvino-intel-npu-plugin` for the NPU on Arch).

Used by [whispy](https://github.com/daaku/whispy) and
[eg2npu](https://github.com/daaku/eg2npu).

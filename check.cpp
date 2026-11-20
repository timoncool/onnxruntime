#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <array>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto* base = OrtGetApiBase();
    std::cout << "Runtime " << base->GetVersionString() << std::endl;
    if (std::string(base->GetVersionString()) != argv[1]) return 3;
    const auto* api = base->GetApi(31);
    if (!api) return 4;
    Ort::InitApi(api);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "old-cpu");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    Ort::Session session(env, "matmul.onnx", options);
    std::array<float,256> input{};
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 13) - 6);
    const std::array<int64_t,2> shape{16,16};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape.data(), shape.size());
    const char* inputs[] = {"x"};
    const char* outputs[] = {"y"};
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    const auto* output = result.at(0).GetTensorData<float>();
    for (size_t row=0; row<16; ++row) for (size_t col=0; col<16; ++col) {
        float expected = 0;
        for (size_t k=0; k<16; ++k) expected += input[row*16+k] * input[k*16+col];
        if (output[row*16+col] != expected) return 5;
    }
    std::cout << "CPU MatMul: all 256 values match" << std::endl;
}

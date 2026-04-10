#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"
#include "AudioFile.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

struct Destroyer {
  template <typename T> void operator()(T* obj) const {
    if (obj) {
      obj->Destroy();
    }
  }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, Destroyer>;

template <typename T>
UniquePtr<T> ToUniquePtr(T* ptr) {
  return UniquePtr<T>(ptr);
}

struct Args {
  std::string model_json;
  std::string audio_wav;
  std::string output_json;
  int frame_rate = 30;
  int device_id = 0;
  bool use_gpu_solver = false;
  bool constant_noise = false;
  int identity_index = 0;
  std::string mode = "regression";
};

struct FrameRecord {
  double time_sec = 0.0;
  std::vector<float> weights;
};

struct CallbackData {
  std::vector<FrameRecord> frames;
};

[[noreturn]] void Die(const std::string& message) {
  std::cerr << message << std::endl;
  std::exit(1);
}

void Check(const std::error_code& error, const std::string& context) {
  if (error) {
    Die(context + ": " + error.message());
  }
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        Die("Missing value for " + name);
      }
      return argv[++i];
    };

    if (flag == "--model-json") {
      args.model_json = require_value(flag);
    } else if (flag == "--audio-wav") {
      args.audio_wav = require_value(flag);
    } else if (flag == "--output-json") {
      args.output_json = require_value(flag);
    } else if (flag == "--frame-rate") {
      args.frame_rate = std::stoi(require_value(flag));
    } else if (flag == "--device-id") {
      args.device_id = std::stoi(require_value(flag));
    } else if (flag == "--mode") {
      args.mode = require_value(flag);
    } else if (flag == "--identity-index") {
      args.identity_index = std::stoi(require_value(flag));
    } else if (flag == "--constant-noise") {
      args.constant_noise = true;
    } else if (flag == "--gpu-solver") {
      args.use_gpu_solver = true;
    } else {
      Die("Unknown argument: " + flag);
    }
  }

  if (args.model_json.empty() || args.audio_wav.empty() || args.output_json.empty()) {
    Die("Required arguments: --model-json --audio-wav --output-json");
  }
  if (args.mode != "regression" && args.mode != "diffusion") {
    Die("Unsupported mode: " + args.mode);
  }
  return args;
}

std::vector<float> ReadAudio(const std::string& audio_path) {
  AudioFile<float> audio_file;
  if (!audio_file.load(audio_path)) {
    Die("Unable to load audio wav: " + audio_path);
  }
  if (audio_file.getNumChannels() != 1) {
    Die("Expected mono wav for A2F export");
  }
  if (audio_file.getSampleRate() != 16000) {
    Die("Expected 16kHz wav for A2F export");
  }
  return audio_file.samples[0];
}

std::vector<std::string> CollectPoseNames(nva2f::IBlendshapeExecutor& executor) {
  nva2f::IBlendshapeSolver* skin_solver = nullptr;
  nva2f::IBlendshapeSolver* tongue_solver = nullptr;
  Check(nva2f::GetExecutorSkinSolver(executor, 0, &skin_solver), "GetExecutorSkinSolver");
  Check(nva2f::GetExecutorTongueSolver(executor, 0, &tongue_solver), "GetExecutorTongueSolver");
  if (skin_solver == nullptr) {
    Die("Skin blendshape solver is null");
  }

  std::vector<std::string> names;
  names.reserve(executor.GetWeightCount());
  for (int index = 0; index < skin_solver->NumBlendshapePoses(); ++index) {
    names.emplace_back(skin_solver->GetPoseName(index));
  }
  if (tongue_solver != nullptr) {
    for (int index = 0; index < tongue_solver->NumBlendshapePoses(); ++index) {
      names.emplace_back(tongue_solver->GetPoseName(index));
    }
  }
  if (names.size() != executor.GetWeightCount()) {
    Die("Blendshape pose name count does not match executor weight count");
  }
  return names;
}

UniquePtr<nva2f::IBlendshapeExecutorBundle> CreateBundle(const Args& args) {
  if (args.mode == "regression") {
    return ToUniquePtr(nva2f::ReadRegressionBlendshapeSolveExecutorBundle(
        1,
        args.model_json.c_str(),
        nva2f::IGeometryExecutor::ExecutionOption::SkinTongue,
        args.use_gpu_solver,
        static_cast<std::size_t>(args.frame_rate),
        1,
        nullptr,
        nullptr));
  }

  return ToUniquePtr(nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
      1,
      args.model_json.c_str(),
      nva2f::IGeometryExecutor::ExecutionOption::SkinTongue,
      args.use_gpu_solver,
      static_cast<std::size_t>(args.identity_index),
      args.constant_noise,
      nullptr,
      nullptr));
}

void AccumulateNeutralEmotion(nva2f::IBlendshapeExecutorBundle& bundle) {
  const auto emotion_size = bundle.GetEmotionAccumulator(0).GetEmotionSize();
  std::vector<float> zeros(emotion_size, 0.0f);
  Check(
      bundle.GetEmotionAccumulator(0).Accumulate(
          0, nva2x::HostTensorFloatConstView{zeros.data(), zeros.size()}, bundle.GetCudaStream().Data()),
      "Accumulate neutral emotion");
  Check(bundle.GetEmotionAccumulator(0).Close(), "Close emotion accumulator");
}

void AccumulateAudio(nva2f::IBlendshapeExecutorBundle& bundle, const std::vector<float>& audio_samples) {
  Check(
      bundle.GetAudioAccumulator(0).Accumulate(
          nva2x::HostTensorFloatConstView{audio_samples.data(), audio_samples.size()}, bundle.GetCudaStream().Data()),
      "Accumulate audio");
  Check(bundle.GetAudioAccumulator(0).Close(), "Close audio accumulator");
}

void HostCallback(void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code error) {
  if (error) {
    std::cerr << "Blendshape callback error: " << error.message() << std::endl;
    return;
  }
  auto* callback_data = static_cast<CallbackData*>(userdata);
  FrameRecord record;
  record.time_sec = static_cast<double>(results.timeStampCurrentFrame) / 16000.0;
  record.weights.assign(results.weights.Data(), results.weights.Data() + results.weights.Size());
  callback_data->frames.push_back(std::move(record));
}

void RunExecutor(nva2f::IBlendshapeExecutorBundle& bundle, CallbackData& callback_data) {
  Check(bundle.GetExecutor().SetResultsCallback(HostCallback, &callback_data), "SetResultsCallback");
  while (nva2x::GetNbReadyTracks(bundle.GetExecutor()) > 0) {
    Check(bundle.GetExecutor().Execute(nullptr), "Execute");
  }
  Check(bundle.GetExecutor().Wait(0), "Wait");
}

std::string EscapeJson(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      default:
        escaped += ch;
        break;
    }
  }
  return escaped;
}

void WriteJson(const Args& args, double actual_fps, const std::vector<std::string>& names, const CallbackData& callback_data) {
  std::ofstream output(args.output_json);
  if (!output) {
    Die("Unable to open output json: " + args.output_json);
  }

  output << "{\n";
  output << "  \"provider\": \"a2f-3d-sdk\",\n";
  output << "  \"metadata\": {\n";
  output << "    \"fps\": " << actual_fps << ",\n";
  output << "    \"frame_count\": " << callback_data.frames.size() << ",\n";
  output << "    \"blendshape_names\": [";
  for (std::size_t index = 0; index < names.size(); ++index) {
    if (index > 0) {
      output << ", ";
    }
    output << "\"" << EscapeJson(names[index]) << "\"";
  }
  output << "],\n";
  output << "    \"normalization_policy\": \"raw-a2f\",\n";
  output << "    \"notes\": [";
  output << "\"Provider-native A2F SDK blendshape weights.\", ";
  output << "\"Timestamps were converted from 16kHz audio sample indices to seconds.\", ";
  output << "\"No repository-level MouthClose or eye compensation was applied.\"";
  output << "]\n";
  output << "  },\n";
  output << "  \"frames\": [\n";
  for (std::size_t frame_index = 0; frame_index < callback_data.frames.size(); ++frame_index) {
    const auto& frame = callback_data.frames[frame_index];
    output << "    {\n";
    output << "      \"time_sec\": " << frame.time_sec << ",\n";
    output << "      \"blendshapes\": {";
    for (std::size_t weight_index = 0; weight_index < frame.weights.size(); ++weight_index) {
      if (weight_index > 0) {
        output << ", ";
      }
      output << "\"" << EscapeJson(names[weight_index]) << "\": " << frame.weights[weight_index];
    }
    output << "}\n";
    output << "    }";
    if (frame_index + 1 < callback_data.frames.size()) {
      output << ",";
    }
    output << "\n";
  }
  output << "  ]\n";
  output << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = ParseArgs(argc, argv);
  Check(nva2x::SetCudaDeviceIfNeeded(args.device_id), "SetCudaDeviceIfNeeded");

  auto bundle = CreateBundle(args);
  if (!bundle) {
    Die("Blendshape executor bundle is null");
  }

  std::size_t frame_rate_numerator = 0;
  std::size_t frame_rate_denominator = 1;
  bundle->GetExecutor().GetFrameRate(frame_rate_numerator, frame_rate_denominator);
  const double actual_fps = frame_rate_denominator == 0
      ? static_cast<double>(args.frame_rate)
      : static_cast<double>(frame_rate_numerator) / static_cast<double>(frame_rate_denominator);

  const auto pose_names = CollectPoseNames(bundle->GetExecutor());
  const auto audio_samples = ReadAudio(args.audio_wav);
  AccumulateNeutralEmotion(*bundle);
  AccumulateAudio(*bundle, audio_samples);

  CallbackData callback_data;
  RunExecutor(*bundle, callback_data);
  WriteJson(args, actual_fps, pose_names, callback_data);
  return 0;
}

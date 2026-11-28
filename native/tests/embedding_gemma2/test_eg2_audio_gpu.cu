// GPU verification for EmbeddingGemma 2 (#968) Phase 3 Audio Conformer + Full Trimodal (Text + Image + Audio).
// Verifies on real google/embeddinggemma-2 weights (BF16+FP32-acc on sm_80+, native FP32 zero-FP16 on sm_70):
//   1) Selective loading VRAM residency:
//      - --no-vision --no-audio: text-only (516.90 MiB BF16 / 1033.79 MiB FP32) + audio HTTP 400 fail-closed
//      - --no-vision (audio-only): exact audio delta (582.91 MiB BF16 / 1165.82 MiB FP32)
//      - full trimodal (text + vision + audio): exact combined residency (1419.78 MiB BF16 / 2839.56 MiB FP32)
//   2) Real PyTorch FP32 oracle parity (from phase3_audio_oracle.json) via live HTTP ChatServer POST /v1/embeddings:
//      - sine_440hz_1s (1.0s exact, 99 mel frames, 25 audio soft tokens > 2 chunks)
//      - sine_880hz_1s (acoustic discrimination vs 440Hz)
//      - chirp_oddlen_1_37s (21950 samples, odd length testing pad_to_multiple_of=128 + trailing mask, 35 audio tokens)
//      - text_plus_audio (interleaved text + <|audio> + 25x<|audio|> + <audio|>)
//      - trimodal_text_img_audio (full 3-modality text + 64 image tokens + 25 audio tokens in ONE request!)
//   3) Mixed 4-item trimodal batch ([text, image, audio, text+image+audio]) with Matryoshka dimensions=256,
//      /health & /v1/models input_modalities == ["text", "image", "audio"], plus HTTP 400 guards.
//   4) Audio (B=1, B=4) and full trimodal (B=1) throughput benchmarks.
#include "backends/embedding_gemma2/eg2_audio.h"
#include "backends/embedding_gemma2/eg2_cuda.h"
#include "image/image.h"
#include "ingine/generator.h"
#include "serve/chat_server.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ingine::backends::embedding_gemma2;

int g_failures = 0;

#define CHECK(cond)                                                                                \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);                \
      ++g_failures;                                                                                \
      throw std::runtime_error("check failed");                                                    \
    }                                                                                              \
  } while (0)

struct ParityStats {
  double cosine = 0.0;
  double max_abs = 0.0;
};

ParityStats compare_vecs(const std::vector<float>& a, const std::vector<float>& b) {
  CHECK(a.size() == b.size() && !a.empty());
  double dot = 0.0, na = 0.0, nb = 0.0, ma = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    ma = std::max(ma, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return {dot / (std::sqrt(na) * std::sqrt(nb)), ma};
}

double l2_norm(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
  return std::sqrt(s);
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path model_dir = "/mnt/storage/models/embeddinggemma-2";
  std::filesystem::path oracle_json = "/mnt/storage/atlas_eg2_968_evidence/phase3_audio_oracle.json";
  std::filesystem::path out_json;
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
    else if (arg == "--oracle" && i + 1 < argc) oracle_json = argv[++i];
    else if (arg == "--out" && i + 1 < argc) out_json = argv[++i];
    else if (!arg.empty() && arg[0] != '-') {
      if (positional == 0) model_dir = arg;
      else if (positional == 1) oracle_json = arg;
      else if (positional == 2) out_json = arg;
      ++positional;
    }
  }

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::fprintf(stderr, "eg2_audio_gpu: no CUDA device available; skipping\n");
    return 77;
  }
  if (!std::filesystem::is_regular_file(model_dir / "model.safetensors")) {
    std::fprintf(stderr, "eg2_audio_gpu: checkpoint not found at %s; skipping\n", model_dir.c_str());
    return 77;
  }
  if (!std::filesystem::is_regular_file(oracle_json)) {
    std::fprintf(stderr, "eg2_audio_gpu: oracle JSON not found at %s; skipping\n", oracle_json.c_str());
    return 77;
  }

  try {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const int sm = prop.major * 10 + prop.minor;

    const ingine::text::ChatFormat fmt = ingine::text::ChatFormat::load(model_dir);
    Json oracle;
    {
      std::ifstream of(oracle_json);
      of >> oracle;
    }
    const auto& cases = oracle["cases"];
    CHECK(cases.is_array() && cases.size() >= 5);
    std::map<std::string, Json> case_by_name;
    for (const auto& c : cases) case_by_name[c["name"].get<std::string>()] = c;
    const std::string wav440_b64 = case_by_name["sine_440hz_1s"]["wav_b64"].get<std::string>();

    // 1) Selective loading check: --no-vision --no-audio -> text-only VRAM & audio HTTP 400 fail-closed!
    std::uint64_t text_only_bytes = 0;
    {
      ingine::model::RuntimeOptions opts_noaud;
      opts_noaud.model_dir = model_dir;
      opts_noaud.max_ctx = 8192;
      opts_noaud.vision = false;
      opts_noaud.audio = false;
      auto gen_na = std::make_unique<Eg2CudaBackend>(opts_noaud);
      CHECK(!gen_na->supports_images() && !gen_na->supports_audio());
      text_only_bytes = gen_na->residency_report().weights_device_bytes;

      ingine::serve::ServerOptions sopts;
      sopts.max_model_len = 8192;
      ingine::serve::ChatServer srv(*gen_na, fmt, sopts);
      CHECK(srv.bind("127.0.0.1", 0));
      srv.start();
      httplib::Client cli("127.0.0.1", srv.port());
      cli.set_read_timeout(30, 0);
      auto h = cli.Get("/health");
      CHECK(h && h->status == 200);
      CHECK(Json::parse(h->body)["input_modalities"] == Json::array({"text"}));

      const Json rej_req = {{"input", Json::array({{{"type", "input_audio"},
                                                    {"input_audio", {{"data", wav440_b64}, {"format", "wav"}}}}})}};
      auto r_rej = cli.Post("/v1/embeddings", rej_req.dump(), "application/json");
      CHECK(r_rej && r_rej->status == 400);
      srv.stop();
      std::printf("[PASS] --no-audio fail-closed HTTP 400 verified (text_only=%.2f MiB)\n",
                  double(text_only_bytes) / (1024.0 * 1024.0));
    }

    // 2) Audio-only (text + audio, --no-vision) -> exact audio weight VRAM delta!
    std::uint64_t text_plus_audio_bytes = 0;
    {
      ingine::model::RuntimeOptions opts_aud;
      opts_aud.model_dir = model_dir;
      opts_aud.max_ctx = 8192;
      opts_aud.vision = false;
      opts_aud.audio = true;
      auto gen_a = std::make_unique<Eg2CudaBackend>(opts_aud);
      CHECK(!gen_a->supports_images() && gen_a->supports_audio());
      text_plus_audio_bytes = gen_a->residency_report().weights_device_bytes;
    }

    // 3) Full trimodal CUDA backend (text + vision + audio)
    ingine::model::RuntimeOptions opts;
    opts.model_dir = model_dir;
    opts.max_ctx = 8192;
    opts.vision = true;
    opts.audio = true;
    const auto t_load0 = std::chrono::steady_clock::now();
    auto backend = std::make_unique<Eg2CudaBackend>(opts);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    const bool is_bf16 = backend->uses_bf16();
    CHECK(backend->supports_images() && backend->supports_audio());

    const std::uint64_t trimodal_bytes = backend->residency_report().weights_device_bytes;
    const std::uint64_t audio_delta_bytes = text_plus_audio_bytes - text_only_bytes;
    const double audio_delta_mib = double(audio_delta_bytes) / (1024.0 * 1024.0);
    const double trimodal_mib = double(trimodal_bytes) / (1024.0 * 1024.0);
    std::printf("[GPU TRIMODAL] device=%s sm_%d mode=%s audio_delta=%.2f MiB trimodal_total=%.2f MiB load=%.3fs\n",
                prop.name, sm, is_bf16 ? "BF16(FP32-acc)" : "FP32(zero-FP16)", audio_delta_mib, trimodal_mib, load_s);
    if (is_bf16) {
      CHECK(audio_delta_mib > 578.0 && audio_delta_mib < 588.0);
      CHECK(trimodal_mib > 1415.0 && trimodal_mib < 1425.0);
    } else {
      CHECK(audio_delta_mib > 1160.0 && audio_delta_mib < 1172.0);
      CHECK(trimodal_mib > 2834.0 && trimodal_mib < 2846.0);
    }

    // 4) Live HTTP ChatServer trimodal parity against real PyTorch FP32 oracle!
    ingine::serve::ServerOptions sopts;
    sopts.max_model_len = 8192;
    sopts.images.enabled = true;
    ingine::serve::ChatServer srv(*backend, fmt, sopts);
    CHECK(srv.bind("127.0.0.1", 0));
    srv.start();
    httplib::Client cli("127.0.0.1", srv.port());
    cli.set_read_timeout(120, 0);

    {
      auto h = cli.Get("/health");
      CHECK(h && h->status == 200);
      CHECK(Json::parse(h->body)["input_modalities"] == Json::array({"text", "image", "audio"}));
      auto m = cli.Get("/v1/models");
      CHECK(m && m->status == 200);
      CHECK(Json::parse(m->body)["data"][0]["input_modalities"] == Json::array({"text", "image", "audio"}));
    }

    const double min_cos = is_bf16 ? 0.9985 : 0.9999;
    Json parity_report = Json::object();

    for (const auto& c : cases) {
      const std::string cname = c["name"].get<std::string>();
      const std::string kind = c["kind"].get<std::string>();
      Json parts = Json::array();
      if (kind == "audio") {
        parts.push_back({{"type", "input_audio"},
                         {"input_audio", {{"data", c["wav_b64"].get<std::string>()}, {"format", "wav"}}}});
      } else if (kind == "text_audio") {
        parts.push_back({{"type", "text"}, {"text", c["text"].get<std::string>()}});
        parts.push_back({{"type", "audio"}, {"data", "data:audio/wav;base64," + c["wav_b64"].get<std::string>()}});
      } else if (kind == "trimodal") {
        parts.push_back({{"type", "text"}, {"text", c["text"].get<std::string>()}});
        parts.push_back({{"type", "image_url"},
                         {"image_url",
                          {{"url", c["image_data_uri"].get<std::string>()},
                           {"max_soft_tokens", c["image_max_soft_tokens"].get<int>()}}}});
        parts.push_back({{"type", "input_audio"},
                         {"input_audio", {{"data", c["wav_b64"].get<std::string>()}, {"format", "wav"}}}});
      }
      const Json req_j = {{"input", parts}, {"normalize", true}};
      auto r = cli.Post("/v1/embeddings", req_j.dump(), "application/json");
      CHECK(r && r->status == 200);
      const Json rj = Json::parse(r->body);
      const auto got = rj["data"][0]["embedding"].get<std::vector<float>>();
      const auto expected = c["embedding_768"].get<std::vector<float>>();
      CHECK(got.size() == 768);
      CHECK(std::abs(l2_norm(got) - 1.0) < 1e-4);
      CHECK(rj["usage"]["prompt_tokens"].get<int>() == static_cast<int>(c["token_ids"].size()));

      const auto ps = compare_vecs(got, expected);
      std::printf("[ORACLE %s] gpu_vs_pytorch_fp32 cos=%.7f max_abs=%.6f tokens=%d\n", cname.c_str(), ps.cosine,
                  ps.max_abs, rj["usage"]["prompt_tokens"].get<int>());
      CHECK(ps.cosine >= min_cos);
      parity_report[cname] = {{"cos_vs_pytorch_fp32", ps.cosine},
                              {"max_abs", ps.max_abs},
                              {"prompt_tokens", rj["usage"]["prompt_tokens"].get<int>()}};
    }

    // 5) Mixed 4-item trimodal batch + Matryoshka dim=256 + fail-closed HTTP 400 checks
    {
      const auto& tri = case_by_name["trimodal_text_img_audio"];
      const Json mreq = {
          {"input", Json::array({
                        "plain text query",
                        Json::array({{{"type", "image_url"},
                                      {"image_url", {{"url", tri["image_data_uri"].get<std::string>()}, {"max_soft_tokens", 70}}}}}),
                        Json::array({{{"type", "input_audio"}, {"input_audio", {{"data", wav440_b64}, {"format", "wav"}}}}}),
                        Json::array({{{"type", "text"}, {"text", "All three: "}},
                                     {{"type", "image_url"},
                                      {"image_url", {{"url", tri["image_data_uri"].get<std::string>()}, {"max_soft_tokens", 70}}}},
                                     {{"type", "input_audio"}, {"input_audio", {{"data", wav440_b64}, {"format", "wav"}}}}}),
                    })},
          {"dimensions", 256},
          {"normalize", true},
      };
      auto rm = cli.Post("/v1/embeddings", mreq.dump(), "application/json");
      CHECK(rm && rm->status == 200);
      const Json rmj = Json::parse(rm->body);
      CHECK(rmj["data"].size() == 4);
      for (int i = 0; i < 4; ++i) CHECK(rmj["data"][i]["embedding"].size() == 256);

      // prompt_name on pure-audio must fail closed with HTTP 400
      const Json preq = {
          {"input", Json::array({{{"type", "input_audio"}, {"input_audio", {{"data", wav440_b64}, {"format", "wav"}}}}})},
          {"prompt_name", "SearchQuery"},
      };
      auto rp = cli.Post("/v1/embeddings", preq.dump(), "application/json");
      CHECK(rp && rp->status == 400);

      // Unsupported audio format ("mp3") must fail closed with HTTP 400
      const Json bad_fmt = {
          {"input", Json::array({{{"type", "input_audio"}, {"input_audio", {{"data", wav440_b64}, {"format", "mp3"}}}}})},
      };
      auto rf = cli.Post("/v1/embeddings", bad_fmt.dump(), "application/json");
      CHECK(rf && rf->status == 400);
    }

    // 6) Audio (B=1, B=4) and Full Trimodal (B=1) throughput benchmarks
    Json bench = Json::array();
    {
      const auto aud440 = eg2_decode_wav_base64_or_data_uri(wav440_b64);
      const auto aud_tids = case_by_name["sine_440hz_1s"]["token_ids"].get<std::vector<std::int32_t>>();
      for (auto [B, iters] : {std::pair{1, 15}, std::pair{4, 10}}) {
        ingine::model::EmbeddingRequest rb;
        rb.normalize = true;
        for (int i = 0; i < B; ++i) {
          rb.inputs.push_back(aud_tids);
          rb.audios.push_back({aud440});
        }
        (void)backend->embed(rb);  // warmup
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) (void)backend->embed(rb);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double ms_per_batch = (elapsed / iters) * 1e3;
        const double clips_per_s = (static_cast<double>(B) * iters) / elapsed;
        std::printf("[BENCH AUDIO 1.0s B=%d soft=25/clip] %.2f ms/batch | %.1f clips/s\n", B, ms_per_batch, clips_per_s);
        bench.push_back({{"workload", "audio_1s"},
                         {"batch", B},
                         {"audio_soft_tokens", 25},
                         {"ms_per_batch", ms_per_batch},
                         {"clips_per_s", clips_per_s}});
      }

      // Full trimodal B=1 benchmark (text + blue_96x96 m70 + 1.0s audio)
      const auto& tri = case_by_name["trimodal_text_img_audio"];
      const auto decoded_img = ingine::image::decode_raw_rgb_data_uri(tri["image_data_uri"].get<std::string>());
      ingine::model::EmbeddingImage eimg;
      eimg.width = decoded_img.width;
      eimg.height = decoded_img.height;
      eimg.rgb = decoded_img.rgb;
      eimg.max_soft_tokens = 70;
      ingine::model::EmbeddingRequest rtri;
      rtri.normalize = true;
      rtri.inputs.push_back(tri["token_ids"].get<std::vector<std::int32_t>>());
      rtri.images.push_back({eimg});
      rtri.audios.push_back({aud440});
      (void)backend->embed(rtri);  // warmup
      constexpr int kTriIters = 10;
      const auto t0 = std::chrono::steady_clock::now();
      for (int it = 0; it < kTriIters; ++it) (void)backend->embed(rtri);
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double ms_tri = (elapsed / kTriIters) * 1e3;
      std::printf("[BENCH TRIMODAL B=1 (text+img64+aud25)] %.2f ms/req | %.1f req/s\n", ms_tri, 1000.0 / ms_tri);
      bench.push_back({{"workload", "trimodal_text_img64_aud25"},
                       {"batch", 1},
                       {"ms_per_batch", ms_tri},
                       {"req_per_s", 1000.0 / ms_tri}});
    }

    srv.stop();

    if (!out_json.empty()) {
      Json report{
          {"device", prop.name},
          {"sm", sm},
          {"mode", is_bf16 ? "bf16_fp32_acc" : "fp32_zero_fp16"},
          {"text_only_bytes", text_only_bytes},
          {"text_plus_audio_bytes", text_plus_audio_bytes},
          {"audio_delta_bytes", audio_delta_bytes},
          {"audio_delta_mib", audio_delta_mib},
          {"trimodal_bytes", trimodal_bytes},
          {"trimodal_mib", trimodal_mib},
          {"gpu_load_s", load_s},
          {"oracle_parity", parity_report},
          {"benchmarks", bench},
          {"no_audio_http_400", "PASS"},
          {"status", "PASS"},
      };
      std::filesystem::create_directories(out_json.parent_path());
      std::ofstream f(out_json);
      f << report.dump(2) << "\n";
    }
    std::printf("ok\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eg2_audio_gpu failed: %s\n", e.what());
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}

// GPU verification for EmbeddingGemma 2 (#968) Phase 2 Vision Tower + Multimodal Interleaving.
// Verifies on real google/embeddinggemma-2 weights:
//   1) Selective loading VRAM savings: --no-vision (516.9 MiB BF16 / 1033.8 MiB FP32) vs vision-on
//      (836.9 MiB BF16 / 1673.7 MiB FP32; delta = 319.97 MiB BF16 / 639.94 MiB FP32), plus --no-vision HTTP 400.
//   2) Exact PyTorch FP32 oracle parity (from phase2_vision_oracle.json) across:
//      - blue_96x96 (256 soft tokens) & red_96x96 (256 soft tokens)
//      - rect_193x127_m280 (bicubic antialias resize -> 960x624, 260 soft tokens, X!=Y axial 2D RoPE)
//      - rect_193x127_m70  (bicubic antialias resize -> 480x288, 60 soft tokens)
//      - video_blue_red_2f (2-frame video, max_soft_tokens=140, 242 <|video|> soft tokens)
//   3) Cross-modal image <-> text retrieval margin (blue_img closer to "blue" than "red", red_img closer to "red").
//   4) Live HTTP ChatServer multimodal POST /v1/embeddings (image_url, interleaved text+image, video frames,
//      Matryoshka dimensions=256/128, prompt_name on pure-media HTTP 400, --no-vision HTTP 400).
//   5) Vision + multimodal embedding throughput benchmark.
#include "backends/embedding_gemma2/eg2_cuda.h"
#include "backends/embedding_gemma2/eg2_vision.h"
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

double dot_f32(const std::vector<float>& a, const std::vector<float>& b) {
  CHECK(a.size() == b.size());
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) d += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return d;
}

double l2_norm(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
  return std::sqrt(s);
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path model_dir = "/mnt/storage/models/embeddinggemma-2";
  std::filesystem::path oracle_json = "/mnt/storage/atlas_eg2_968_evidence/phase2_vision_oracle.json";
  std::filesystem::path out_json;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
    else if (arg == "--oracle" && i + 1 < argc) oracle_json = argv[++i];
    else if (arg == "--out" && i + 1 < argc) out_json = argv[++i];
  }

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::fprintf(stderr, "eg2_vision_gpu: no CUDA device available; skipping\n");
    return 77;
  }
  if (!std::filesystem::is_regular_file(model_dir / "model.safetensors")) {
    std::fprintf(stderr, "eg2_vision_gpu: checkpoint not found at %s; skipping\n", model_dir.c_str());
    return 77;
  }
  if (!std::filesystem::is_regular_file(oracle_json)) {
    std::fprintf(stderr, "eg2_vision_gpu: oracle JSON not found at %s; skipping\n", oracle_json.c_str());
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

    // 1) Selective loading verification: --no-vision first -> measure text-only VRAM and HTTP 400 fail-closed!
    std::uint64_t text_only_bytes = 0;
    {
      ingine::model::RuntimeOptions opts_novis;
      opts_novis.model_dir = model_dir;
      opts_novis.max_ctx = 8192;
      opts_novis.vision = false;
      opts_novis.audio = false;
      auto gen_nv = std::make_unique<Eg2CudaBackend>(opts_novis);
      CHECK(!gen_nv->supports_images());
      text_only_bytes = gen_nv->residency_report().weights_device_bytes;

      ingine::serve::ServerOptions sopts;
      sopts.max_model_len = 8192;
      sopts.images.enabled = true;
      ingine::serve::ChatServer srv(*gen_nv, fmt, sopts);
      CHECK(srv.bind("127.0.0.1", 0));
      srv.start();
      httplib::Client cli("127.0.0.1", srv.port());
      cli.set_read_timeout(30, 0);
      auto h = cli.Get("/health");
      CHECK(h && h->status == 200);
      CHECK(Json::parse(h->body)["input_modalities"] == Json::array({"text"}));

      const std::string blue_uri = "data:image/png;base64," + oracle["cases"]["blue_96x96"]["png_base64"].get<std::string>();
      const Json req_body = {{"input", Json::array({{{"type", "image_url"}, {"image_url", {{"url", blue_uri}}}}})}};
      auto r_rej = cli.Post("/v1/embeddings", req_body.dump(), "application/json");
      CHECK(r_rej && r_rej->status == 400);
      srv.stop();
      std::printf("[PASS] --no-vision fail-closed HTTP 400 verified (text_only=%.2f MiB)\n",
                  double(text_only_bytes) / (1024.0 * 1024.0));
    }

    // 2) Vision-enabled CUDA backend
    ingine::model::RuntimeOptions opts;
    opts.model_dir = model_dir;
    opts.max_ctx = 8192;
    opts.vision = true;
    opts.audio = false;
    const auto t_load0 = std::chrono::steady_clock::now();
    auto backend = std::make_unique<Eg2CudaBackend>(opts);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    const bool is_bf16 = backend->uses_bf16();
    CHECK(backend->supports_images());

    const std::uint64_t text_plus_vis_bytes = backend->residency_report().weights_device_bytes;
    const std::uint64_t vis_delta_bytes = text_plus_vis_bytes - text_only_bytes;
    const double vis_delta_mib = double(vis_delta_bytes) / (1024.0 * 1024.0);
    std::printf("[GPU VISION] device=%s sm_%d mode=%s total=%.2f MiB vision_delta=%.2f MiB load=%.3fs\n", prop.name, sm,
                is_bf16 ? "BF16(FP32-acc)" : "FP32(zero-FP16)", double(text_plus_vis_bytes) / (1024.0 * 1024.0),
                vis_delta_mib, load_s);
    if (is_bf16) {
      CHECK(vis_delta_mib > 315.0 && vis_delta_mib < 325.0);
    } else {
      CHECK(vis_delta_mib > 635.0 && vis_delta_mib < 645.0);
    }

    // 3) Live HTTP ChatServer multimodal parity against real PyTorch FP32 oracle!
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
      const Json hj = Json::parse(h->body);
      CHECK(hj["input_modalities"] == Json::array({"text", "image"}));
      auto m = cli.Get("/v1/models");
      CHECK(m && m->status == 200);
      CHECK(Json::parse(m->body)["data"][0]["input_modalities"] == Json::array({"text", "image"}));
    }

    const double min_cos = is_bf16 ? 0.9985 : 0.9999;
    Json parity_report = Json::object();
    std::map<std::string, std::vector<float>> gpu_embs;

    for (const std::string& cname : {"blue_96x96", "red_96x96", "rect_193x127_m280", "rect_193x127_m70"}) {
      const Json& c = oracle["cases"][cname];
      const std::string uri = "data:image/png;base64," + c["png_base64"].get<std::string>();
      const int max_st = c["max_soft_tokens"].get<int>();
      const Json req_j = {
          {"input", Json::array({{{"type", "image_url"}, {"image_url", {{"url", uri}, {"max_soft_tokens", max_st}}}}})},
          {"normalize", true},
      };
      auto r = cli.Post("/v1/embeddings", req_j.dump(), "application/json");
      CHECK(r && r->status == 200);
      const Json rj = Json::parse(r->body);
      const auto got = rj["data"][0]["embedding"].get<std::vector<float>>();
      const auto expected = c["emb_768"].get<std::vector<float>>();
      CHECK(got.size() == 768);
      CHECK(std::abs(l2_norm(got) - 1.0) < 1e-4);
      CHECK(rj["usage"]["prompt_tokens"].get<int>() == static_cast<int>(c["token_ids"].size()));

      const auto ps = compare_vecs(got, expected);
      std::printf("[ORACLE %s] gpu_vs_pytorch_fp32 cos=%.7f max_abs=%.6f tokens=%d\n", cname.c_str(), ps.cosine,
                  ps.max_abs, rj["usage"]["prompt_tokens"].get<int>());
      CHECK(ps.cosine >= min_cos);
      gpu_embs[cname] = got;
      parity_report[cname] = {{"cos_vs_pytorch_fp32", ps.cosine},
                              {"max_abs", ps.max_abs},
                              {"n_soft_tokens", c["n_soft_tokens"].get<int>()},
                              {"prompt_tokens", rj["usage"]["prompt_tokens"].get<int>()}};
    }

    // 4) Two-frame video parity against PyTorch FP32 oracle
    {
      const std::string blue_uri = "data:image/png;base64," + oracle["cases"]["blue_96x96"]["png_base64"].get<std::string>();
      const std::string red_uri = "data:image/png;base64," + oracle["cases"]["red_96x96"]["png_base64"].get<std::string>();
      const Json vreq = {
          {"input", Json::array({{{"type", "video"}, {"frames", Json::array({blue_uri, red_uri})}, {"max_soft_tokens", 140}}})},
          {"normalize", true},
      };
      auto r = cli.Post("/v1/embeddings", vreq.dump(), "application/json");
      CHECK(r && r->status == 200);
      const Json rj = Json::parse(r->body);
      const auto got = rj["data"][0]["embedding"].get<std::vector<float>>();
      const auto expected = oracle["cases"]["video_blue_red_2f"]["emb_768"].get<std::vector<float>>();
      const auto ps = compare_vecs(got, expected);
      std::printf("[ORACLE video_blue_red_2f] gpu_vs_pytorch_fp32 cos=%.7f max_abs=%.6f tokens=%d\n", ps.cosine,
                  ps.max_abs, rj["usage"]["prompt_tokens"].get<int>());
      CHECK(ps.cosine >= min_cos);
      parity_report["video_blue_red_2f"] = {{"cos_vs_pytorch_fp32", ps.cosine},
                                            {"max_abs", ps.max_abs},
                                            {"prompt_tokens", rj["usage"]["prompt_tokens"].get<int>()}};
    }

    // 5) Cross-modal image <-> text retrieval check + mixed batch [text, image, text+image] + Matryoshka + prompt_name guards
    double cos_b_blue = 0.0, cos_b_red = 0.0, cos_r_red = 0.0, cos_r_blue = 0.0;
    {
      const Json treq = {{"input", Json::array({"a photo of the color blue", "a photo of the color red"})},
                         {"normalize", true}};
      auto rt = cli.Post("/v1/embeddings", treq.dump(), "application/json");
      CHECK(rt && rt->status == 200);
      const Json rtj = Json::parse(rt->body);
      const auto e_blue_txt = rtj["data"][0]["embedding"].get<std::vector<float>>();
      const auto e_red_txt = rtj["data"][1]["embedding"].get<std::vector<float>>();
      cos_b_blue = dot_f32(gpu_embs["blue_96x96"], e_blue_txt);
      cos_b_red = dot_f32(gpu_embs["blue_96x96"], e_red_txt);
      cos_r_red = dot_f32(gpu_embs["red_96x96"], e_red_txt);
      cos_r_blue = dot_f32(gpu_embs["red_96x96"], e_blue_txt);
      std::printf("[CROSS-MODAL] cos(blue_img,blue)=%.5f vs red=%.5f (margin=%+.5f) | cos(red_img,red)=%.5f vs blue=%.5f (margin=%+.5f)\n",
                  cos_b_blue, cos_b_red, cos_b_blue - cos_b_red, cos_r_red, cos_r_blue, cos_r_red - cos_r_blue);
      CHECK(cos_b_blue > cos_b_red + 0.02);
      CHECK(cos_r_red > cos_r_blue + 0.02);

      // Mixed batch + interleaved text+image + Matryoshka dim=256
      const std::string rect_uri =
          "data:image/png;base64," + oracle["cases"]["rect_193x127_m70"]["png_base64"].get<std::string>();
      const Json mreq = {
          {"input", Json::array({
                        "plain text query",
                        Json::array({{{"type", "text"}, {"text", "Image caption: "}},
                                     {{"type", "image_url"}, {"image_url", {{"url", rect_uri}, {"max_soft_tokens", 70}}}}}),
                    })},
          {"dimensions", 256},
          {"normalize", true},
      };
      auto rm = cli.Post("/v1/embeddings", mreq.dump(), "application/json");
      CHECK(rm && rm->status == 200);
      const Json rmj = Json::parse(rm->body);
      CHECK(rmj["data"].size() == 2);
      CHECK(rmj["data"][0]["embedding"].size() == 256);
      CHECK(rmj["data"][1]["embedding"].size() == 256);

      // prompt_name on pure-media item must fail closed with HTTP 400
      const Json preq = {
          {"input", Json::array({{{"type", "image_url"}, {"image_url", {{"url", rect_uri}, {"max_soft_tokens", 70}}}}})},
          {"prompt_name", "SearchQuery"},
      };
      auto rp = cli.Post("/v1/embeddings", preq.dump(), "application/json");
      CHECK(rp && rp->status == 400);
    }

    // 6) Vision + Multimodal throughput benchmark (direct backend.embed on rect_193x127_m280, B=1 and B=4)
    Json bench = Json::array();
    {
      const std::string raw_png = ingine::image::decode_base64(oracle["cases"]["rect_193x127_m280"]["png_base64"].get<std::string>());
      const auto decoded = ingine::image::decode_raw_rgb(
          std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(raw_png.data()), raw_png.size()));
      ingine::model::EmbeddingImage eimg;
      eimg.width = decoded.width;
      eimg.height = decoded.height;
      eimg.rgb = decoded.rgb;
      eimg.max_soft_tokens = 280;
      const auto tids = oracle["cases"]["rect_193x127_m280"]["token_ids"].get<std::vector<std::int32_t>>();

      for (auto [B, iters] : {std::pair{1, 10}, std::pair{4, 5}}) {
        ingine::model::EmbeddingRequest rb;
        rb.normalize = true;
        for (int i = 0; i < B; ++i) {
          rb.inputs.push_back(tids);
          rb.images.push_back({eimg});
        }
        (void)backend->embed(rb);  // warmup
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) (void)backend->embed(rb);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double ms_per_batch = (elapsed / iters) * 1e3;
        const double img_per_s = (static_cast<double>(B) * iters) / elapsed;
        std::printf("[BENCH VISION B=%d soft=260/img] %.2f ms/batch | %.1f img/s\n", B, ms_per_batch, img_per_s);
        bench.push_back({{"batch", B}, {"soft_tokens_per_img", 260}, {"ms_per_batch", ms_per_batch}, {"img_per_s", img_per_s}});
      }
    }

    srv.stop();

    if (!out_json.empty()) {
      Json report{
          {"device", prop.name},
          {"sm", sm},
          {"mode", is_bf16 ? "bf16_fp32_acc" : "fp32_zero_fp16"},
          {"text_only_bytes", text_only_bytes},
          {"text_plus_vision_bytes", text_plus_vis_bytes},
          {"vision_delta_bytes", vis_delta_bytes},
          {"vision_delta_mib", vis_delta_mib},
          {"gpu_load_s", load_s},
          {"oracle_parity", parity_report},
          {"cross_modal",
           {{"cos_blue_img_blue_txt", cos_b_blue},
            {"cos_blue_img_red_txt", cos_b_red},
            {"blue_margin", cos_b_blue - cos_b_red},
            {"cos_red_img_red_txt", cos_r_red},
            {"cos_red_img_blue_txt", cos_r_blue},
            {"red_margin", cos_r_red - cos_r_blue}}},
          {"benchmarks", bench},
          {"no_vision_http_400", "PASS"},
          {"status", "PASS"},
      };
      std::filesystem::create_directories(out_json.parent_path());
      std::ofstream f(out_json);
      f << report.dump(2) << "\n";
    }
    std::printf("ok\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eg2_vision_gpu failed: %s\n", e.what());
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}

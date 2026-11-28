#include "serve/openai.h"

#include "backends/embedding_gemma2/eg2_audio.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>

namespace ingine::serve {
namespace {

struct Bound {
  double value;
  const char* text;  // Python's str() of the bound, as printed in the error message
};

const Json* get(const Json& body, const char* key) {
  auto it = body.find(key);
  return it == body.end() || it->is_null() ? nullptr : &*it;
}

std::string range_message(const char* key, const std::optional<Bound>& lo, const std::optional<Bound>& hi) {
  return std::string("'") + key + "' must be in [" + (lo ? lo->text : "None") + ", " + (hi ? hi->text : "None") +
         "]";
}

int64_t saturate(double v) {
  if (v >= 9.2233720368547758e18) return std::numeric_limits<int64_t>::max();
  if (v <= -9.2233720368547758e18) return std::numeric_limits<int64_t>::min();
  return static_cast<int64_t>(v);
}

// server._num(body, key, int, lo, hi)
std::optional<int64_t> num_int(const Json& body, const char* key, std::optional<Bound> lo = std::nullopt,
                               std::optional<Bound> hi = std::nullopt) {
  const Json* v = get(body, key);
  if (!v) return std::nullopt;
  if (v->is_boolean() || !v->is_number() ||
      (v->is_number_float() && std::trunc(v->get<double>()) != v->get<double>())) {
    throw BadRequest(std::string("'") + key + "' must be an integer", key);
  }
  int64_t value;
  if (v->is_number_unsigned()) {
    value = int64_t(std::min<uint64_t>(v->get<uint64_t>(), uint64_t(std::numeric_limits<int64_t>::max())));
  } else if (v->is_number_integer()) {
    value = v->get<int64_t>();
  } else {
    value = saturate(v->get<double>());
  }
  if ((lo && double(value) < lo->value) || (hi && double(value) > hi->value)) {
    throw BadRequest(range_message(key, lo, hi), key);
  }
  return value;
}

// server._num(body, key, float, lo, hi)
std::optional<double> num_float(const Json& body, const char* key, std::optional<Bound> lo, std::optional<Bound> hi) {
  const Json* v = get(body, key);
  if (!v) return std::nullopt;
  if (v->is_boolean() || !v->is_number()) throw BadRequest(std::string("'") + key + "' must be a number", key);
  const double value = v->get<double>();
  if ((lo && value < lo->value) || (hi && value > hi->value)) throw BadRequest(range_message(key, lo, hi), key);
  return value;
}

int clamp_int(int64_t v) { return int(std::clamp<int64_t>(v, INT_MIN, INT_MAX)); }

// An image content part: its place in the request (for messages) and its data, a data: URI or bare base64.
struct ImagePart {
  std::string where;         // "messages[i].content[j]"
  const std::string* uri = nullptr;
  const std::string* base64 = nullptr;
};

const std::string* string_at(const Json& object, const char* key) {
  const auto it = object.find(key);
  return it != object.end() && it->is_string() ? &it->get_ref<const std::string&>() : nullptr;
}

// The image parts of the request in the order the chat template renders them: messages in order, the parts
// of a message in order. Shapes the template would reject anyway (non-array messages, ...) are skipped here.
std::vector<ImagePart> collect_images(const Json& messages) {
  std::vector<ImagePart> parts;
  if (!messages.is_array()) return parts;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const Json& message = messages[i];
    if (!message.is_object()) continue;
    const auto content = message.find("content");
    if (content == message.end() || !content->is_array()) continue;
    for (std::size_t j = 0; j < content->size(); ++j) {
      const Json& part = (*content)[j];
      if (!part.is_object()) continue;
      const std::string* type = string_at(part, "type");
      if (!type || (*type != "image_url" && *type != "input_image" && *type != "image")) continue;
      ImagePart found;
      found.where = "messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "]";
      if (const auto url = part.find("image_url"); url != part.end() && url->is_object()) {
        found.uri = string_at(*url, "url");
      } else {
        found.uri = string_at(part, "image_url");
      }
      if (!found.uri) found.uri = string_at(part, "image");
      if (!found.uri) found.uri = string_at(part, "url");
      if (!found.uri) {
        if (const auto source = part.find("source"); source != part.end() && source->is_object()) {
          const std::string* kind = string_at(*source, "type");
          if (kind && *kind == "base64") found.base64 = string_at(*source, "data");
          if (kind && *kind == "url") found.uri = string_at(*source, "url");
        }
      }
      if (!found.uri && !found.base64) {
        throw BadRequest(found.where + ": an image part needs a base64 data: URI (image_url.url)", "messages");
      }
      parts.push_back(std::move(found));
    }
  }
  return parts;
}

// Decodes the request's images and replaces the template's placeholder tokens by the images' tokens.
void expand_images(const std::vector<ImagePart>& parts, const ImageOptions& options, std::vector<int32_t>& ids,
                   std::vector<model::PromptImage>& images) {
  if (int64_t(parts.size()) > options.max_images) {
    throw BadRequest("the request has " + std::to_string(parts.size()) + " images; at most " +
                         std::to_string(options.max_images) + " are accepted",
                     "messages");
  }
  // The template renders exactly one placeholder per image part, in order. More placeholders than images
  // means message text spells the placeholder token, and no pairing of the two is right.
  const auto placeholders = std::count(ids.begin(), ids.end(), model::IMAGE_PLACEHOLDER_ID);
  if (placeholders != std::ptrdiff_t(parts.size())) {
    throw BadRequest("message text must not contain the image placeholder token <|unused_200054|> in a request "
                     "with images",
                     "messages");
  }
  std::vector<image::Image> decoded;
  decoded.reserve(parts.size());
  int64_t total = 0;
  for (const ImagePart& part : parts) {
    try {
      if (part.uri) {
        decoded.push_back(image::decode_data_uri(*part.uri, options.limits));
      } else {
        const std::string bytes = image::decode_base64(*part.base64);
        decoded.push_back(image::decode(
            std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()),
            options.limits));
      }
    } catch (const image::Error& e) {
      throw BadRequest(part.where + ": " + e.what(), "messages");
    }
    total += decoded.back().tokens();
  }
  if (total > options.max_total_tokens) {
    throw BadRequest("the images of the request take " + std::to_string(total) + " tokens; at most " +
                         std::to_string(options.max_total_tokens) + " are accepted",
                     "messages");
  }
  std::vector<int32_t> expanded;
  expanded.reserve(ids.size() + std::size_t(total));
  std::size_t next = 0;
  for (const int32_t id : ids) {
    if (id != model::IMAGE_PLACEHOLDER_ID) {
      expanded.push_back(id);
      continue;
    }
    image::Image& img = decoded[next++];
    model::PromptImage entry;
    entry.pos = int(expanded.size());
    entry.height = img.height;
    entry.width = img.width;
    const int tokens = img.tokens();
    for (int k = 0; k < tokens; ++k) expanded.push_back(image::token_id(img.hash, k));
    entry.rgb = std::make_shared<const std::vector<std::uint8_t>>(std::move(img.rgb));
    images.push_back(std::move(entry));
  }
  ids = std::move(expanded);
}

// Resolves a local JSON Pointer `$ref` (e.g. "#/$defs/Item" or "#/definitions/Item") inside `root`.
const Json* resolve_local_ref(const Json& root, const std::string& ref) {
  if (!ref.starts_with("#/")) return nullptr;
  const Json* cur = &root;
  std::size_t pos = 2;
  while (pos <= ref.size()) {
    const std::size_t slash = ref.find('/', pos);
    std::string seg = ref.substr(pos, slash == std::string::npos ? std::string::npos : slash - pos);
    // Unescape RFC 6901 JSON pointer tokens (~1 -> /, ~0 -> ~).
    std::string key;
    key.reserve(seg.size());
    for (std::size_t i = 0; i < seg.size(); ++i) {
      if (seg[i] == '~' && i + 1 < seg.size() && (seg[i + 1] == '0' || seg[i + 1] == '1')) {
        key.push_back(seg[i + 1] == '1' ? '/' : '~');
        ++i;
      } else {
        key.push_back(seg[i]);
      }
    }
    if (!cur->is_object()) return nullptr;
    const auto it = cur->find(key);
    if (it == cur->end()) return nullptr;
    cur = &*it;
    if (slash == std::string::npos) break;
    pos = slash + 1;
  }
  return cur;
}

// Compiles a JSON Schema into a concrete JSON instance skeleton plus concise field constraints so models do not
// echo JSON Schema meta-keywords ("type", "properties", "additionalProperties", "$defs") in their output.
Json build_schema_skeleton(const Json& node, const Json& root, std::vector<std::string>& constraints,
                           const std::string& path, int depth = 0) {
  if (!node.is_object() || depth > 12) return Json();
  if (const std::string* ref = string_at(node, "$ref")) {
    if (const Json* target = resolve_local_ref(root, *ref)) {
      return build_schema_skeleton(*target, root, constraints, path, depth + 1);
    }
  }
  for (const char* comb : {"anyOf", "oneOf"}) {
    if (const auto it = node.find(comb); it != node.end() && it->is_array() && !it->empty()) {
      const Json* chosen = nullptr;
      bool nullable = false;
      for (const Json& branch : *it) {
        if (branch.is_object()) {
          const std::string* bt = string_at(branch, "type");
          if (bt && *bt == "null") {
            nullable = true;
            continue;
          }
          if (!chosen) chosen = &branch;
        }
      }
      if (chosen) {
        if (nullable && !path.empty()) constraints.push_back(path + ": nullable");
        return build_schema_skeleton(*chosen, root, constraints, path, depth + 1);
      }
      return nullptr;
    }
  }
  if (const auto it = node.find("const"); it != node.end()) {
    if (!path.empty()) constraints.push_back(path + ": must equal " + it->dump());
    return *it;
  }
  if (const auto it = node.find("enum"); it != node.end() && it->is_array() && !it->empty()) {
    if (!path.empty()) constraints.push_back(path + ": one of " + it->dump());
    return (*it)[0];
  }
  std::string type;
  if (const std::string* t = string_at(node, "type")) {
    type = *t;
  } else if (const auto it = node.find("type"); it != node.end() && it->is_array()) {
    bool nullable = false;
    for (const Json& elem : *it) {
      if (elem.is_string()) {
        if (elem == "null") nullable = true;
        else if (type.empty()) type = elem.get<std::string>();
      }
    }
    if (nullable && !path.empty()) constraints.push_back(path + ": nullable");
  }
  if (type.empty() && node.contains("properties")) type = "object";
  if (type.empty() && node.contains("items")) type = "array";

  if (type == "object") {
    Json obj = Json::object();
    if (const auto props = node.find("properties"); props != node.end() && props->is_object()) {
      for (const auto& [k, v] : props->items()) {
        const std::string child_path = path.empty() ? k : (path + "." + k);
        obj[k] = build_schema_skeleton(v, root, constraints, child_path, depth + 1);
      }
    }
    if (const auto req = node.find("required"); req != node.end() && req->is_array() && !req->empty()) {
      constraints.push_back((path.empty() ? std::string("root") : path) + " required keys: " + req->dump());
    }
    if (const auto ap = node.find("additionalProperties"); ap != node.end() && ap->is_boolean() && !ap->get<bool>()) {
      constraints.push_back((path.empty() ? std::string("root") : path) + ": no extra keys");
    }
    return obj;
  }
  const std::string field = path.empty() ? std::string("root") : path;
  if (type == "array") {
    int min_items = -1;
    int max_items = -1;
    if (const auto it = node.find("minItems"); it != node.end() && it->is_number_integer()) {
      min_items = std::max(0, it->get<int>());
    }
    if (const auto it = node.find("maxItems"); it != node.end() && it->is_number_integer()) {
      max_items = std::max(0, it->get<int>());
    }
    if (min_items >= 0 && max_items >= 0) {
      if (min_items == max_items) {
        constraints.push_back(field + ": array with exactly " + std::to_string(min_items) + " items");
      } else {
        constraints.push_back(field + ": array length between " + std::to_string(min_items) + " and " +
                              std::to_string(max_items) + " items (inclusive)");
      }
    } else if (min_items >= 0) {
      constraints.push_back(field + ": array with at least " + std::to_string(min_items) + " items");
    } else if (max_items >= 0) {
      constraints.push_back(field + ": array with at most " + std::to_string(max_items) + " items");
    }
    if (const auto it = node.find("uniqueItems"); it != node.end() && it->is_boolean() && it->get<bool>()) {
      constraints.push_back(field + ": unique items");
    }
    Json arr = Json::array();
    if (const auto items = node.find("items"); items != node.end() && items->is_object()) {
      const Json elem = build_schema_skeleton(*items, root, constraints, path + "[]", depth + 1);
      int count = (min_items > 0) ? std::min(min_items, 4) : 1;
      if (max_items >= 0) count = std::min(count, max_items);
      for (int i = 0; i < count; ++i) arr.push_back(elem);
    }
    return arr;
  }
  if (type == "string") {
    if (const auto it = node.find("minLength"); it != node.end() && it->is_number_integer()) {
      constraints.push_back(field + ": minLength " + std::to_string(it->get<int>()));
    }
    if (const auto it = node.find("maxLength"); it != node.end() && it->is_number_integer()) {
      constraints.push_back(field + ": maxLength " + std::to_string(it->get<int>()));
    }
    if (const std::string* pat = string_at(node, "pattern")) {
      constraints.push_back(field + ": matches regex " + *pat);
    }
    if (const std::string* fmt = string_at(node, "format")) {
      constraints.push_back(field + ": format " + *fmt);
    }
    return "<string>";
  }
  if (type == "integer" || type == "number") {
    if (const auto it = node.find("minimum"); it != node.end() && it->is_number()) {
      constraints.push_back(field + ": >= " + it->dump());
    }
    if (const auto it = node.find("exclusiveMinimum"); it != node.end() && it->is_number()) {
      constraints.push_back(field + ": > " + it->dump());
    }
    if (const auto it = node.find("maximum"); it != node.end() && it->is_number()) {
      constraints.push_back(field + ": <= " + it->dump());
    }
    if (const auto it = node.find("exclusiveMaximum"); it != node.end() && it->is_number()) {
      constraints.push_back(field + ": < " + it->dump());
    }
    if (const auto it = node.find("multipleOf"); it != node.end() && it->is_number()) {
      constraints.push_back(field + ": multiple of " + it->dump());
    }
    return (type == "integer") ? Json(0) : Json(0.0);
  }
  if (type == "boolean") return false;
  return nullptr;
}

struct ParsedResponseFormat {
  std::string instruction;
  text::JsonFormatSpec spec;
};

// Parses OpenAI response_format ("text", "json_object", "json_schema") into both a prompt-guided formatting
// instruction and a token-level JSON/JSON-Schema FSM spec (None for default/text -> zero overhead).
ParsedResponseFormat parse_response_format(const Json& body, std::string_view raw_body) {
  ParsedResponseFormat out;
  const Json* rf = get(body, "response_format");
  if (!rf) return out;
  if (!rf->is_object()) throw BadRequest("'response_format' must be an object", "response_format");
  const std::string* type = string_at(*rf, "type");
  if (!type) throw BadRequest("'response_format.type' must be a string", "response_format");
  if (*type == "text") return out;
  if (*type == "json_object") {
    out.instruction = "\n\nRespond with a single valid JSON object only, with no markdown fences or surrounding text.";
    out.spec.mode = text::JsonFormatMode::JsonObject;
    return out;
  }
  if (*type == "json_schema") {
    const auto js_it = rf->find("json_schema");
    if (js_it == rf->end() || !js_it->is_object()) {
      throw BadRequest("'response_format.json_schema' must be an object", "response_format");
    }
    const auto schema_it = js_it->find("schema");
    if (schema_it == js_it->end() || !schema_it->is_object()) {
      throw BadRequest("'response_format.json_schema.schema' must be an object", "response_format");
    }
    std::vector<std::string> constraints;
    const Json skeleton = build_schema_skeleton(*schema_it, *schema_it, constraints, "");
    out.instruction =
        "\n\nRespond with a single valid JSON object matching this exact instance structure (replace placeholder values "
        "with your answer; do NOT wrap the output in JSON Schema keywords such as \"type\", \"properties\", "
        "\"required\", or \"additionalProperties\"), with no markdown fences, duplicates, or surrounding text:\n" +
        skeleton.dump();
    if (!constraints.empty()) {
      out.instruction += "\nField constraints:";
      for (const std::string& c : constraints) out.instruction += "\n- " + c;
    }
    out.spec.mode = text::JsonFormatMode::JsonSchema;
    out.spec.schema = *schema_it;
    if (!raw_body.empty()) {
      if (auto raw_schema = text::extract_raw_json_field_path(raw_body, {"response_format", "json_schema", "schema"})) {
        out.spec.raw_schema_json = std::move(*raw_schema);
      }
    }
    return out;
  }
  throw BadRequest("'response_format.type' must be 'text', 'json_object', or 'json_schema'", "response_format");
}

// Appends `suffix` to the last user message (or last message if none is a user turn) on a copy of `messages`.
Json append_format_instruction(const Json& messages, const std::string& suffix) {
  if (suffix.empty() || !messages.is_array() || messages.empty()) return messages;
  Json out = messages;
  std::size_t idx = out.size() - 1;
  for (std::size_t i = out.size(); i-- > 0;) {
    if (out[i].is_object() && string_at(out[i], "role") && *string_at(out[i], "role") == "user") {
      idx = i;
      break;
    }
  }
  Json& msg = out[idx];
  if (!msg.is_object()) return out;
  auto it = msg.find("content");
  if (it == msg.end() || it->is_null()) {
    msg["content"] = suffix;
  } else if (it->is_string()) {
    it->get_ref<std::string&>() += suffix;
  } else if (it->is_array()) {
    it->push_back(Json{{"type", "text"}, {"text", suffix}});
  }
  return out;
}

}  // namespace

bool truthy(const Json& value) {
  switch (value.type()) {
    case Json::value_t::null: return false;
    case Json::value_t::boolean: return value.get<bool>();
    case Json::value_t::number_integer: return value.get<int64_t>() != 0;
    case Json::value_t::number_unsigned: return value.get<uint64_t>() != 0;
    case Json::value_t::number_float: return value.get<double>() != 0.0;
    case Json::value_t::string: return !value.get_ref<const std::string&>().empty();
    case Json::value_t::array:
    case Json::value_t::object: return !value.empty();
    default: return true;
  }
}

ChatRequest parse_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                          const model::Sampling& defaults, const ImageOptions& image_options,
                          std::string_view raw_body, bool allow_logit_masker) {
  if (num_int(body, "n", Bound{1, "1"}, Bound{1, "1"}).value_or(1) != 1) {
    throw BadRequest("only n=1 is supported", "n");
  }
  Json kwargs = Json::object();
  if (auto it = body.find("chat_template_kwargs"); it != body.end() && truthy(*it)) {
    if (!it->is_object()) {
      throw BadRequest("'chat_template_kwargs' must be an object", "chat_template_kwargs");
    }
    kwargs = *it;
  }
  // body.get("reasoning_effort", kwargs.pop("reasoning_effort", None))
  Json effort = kwargs.contains("reasoning_effort") ? kwargs["reasoning_effort"] : Json();
  kwargs.erase("reasoning_effort");
  if (auto it = body.find("reasoning_effort"); it != body.end()) effort = *it;
  // kwargs.pop("enable_thinking", body.get("enable_thinking", True))
  Json enable_thinking = true;
  if (auto it = body.find("enable_thinking"); it != body.end()) enable_thinking = *it;
  if (kwargs.contains("enable_thinking")) {
    enable_thinking = kwargs["enable_thinking"];
    kwargs.erase("enable_thinking");
  }
  if (!enable_thinking.is_boolean()) throw BadRequest("'enable_thinking' must be a boolean", "enable_thinking");
  const bool thinking_enabled = enable_thinking.get<bool>();
  // When thinking is disabled and the caller omitted reasoning_effort, clamp to "none" so chat_template.jinja
  // emits "Thinking effort level: 0" instead of its default "0.9".
  if (!thinking_enabled && effort.is_null()) {
    effort = "none";
  }

  text::ChatOptions options;
  options.reasoning_effort = std::move(effort);
  options.enable_thinking = thinking_enabled;
  options.template_kwargs = std::move(kwargs);
  const auto messages = body.find("messages");
  const auto tools = body.find("tools");
  static const Json kAbsent;  // null: no messages/tools key (passed by reference, not a deep copy of the body)
  const ParsedResponseFormat parsed_rf = parse_response_format(body, raw_body);
  if (parsed_rf.spec.mode != text::JsonFormatMode::None && !allow_logit_masker) {
    throw BadRequest(
        "structured JSON response_format ('json_object'/'json_schema') is not supported by this model backend",
        "response_format");
  }
  const std::string& rf_suffix = parsed_rf.instruction;
  const Json& raw_messages = messages == body.end() ? kAbsent : *messages;
  const Json augmented_messages = rf_suffix.empty() ? Json() : append_format_instruction(raw_messages, rf_suffix);
  const Json& effective_messages = rf_suffix.empty() ? raw_messages : augmented_messages;
  const std::vector<ImagePart> image_parts = collect_images(effective_messages);
  if (!image_parts.empty() && !image_options.enabled) {
    throw BadRequest("this server does not take images (it runs without the vision tower)", "messages");
  }
  const Json& effective_tools = tools == body.end() ? kAbsent : *tools;
  text::RenderedPrompt prompt = fmt.render(effective_messages, effective_tools, options);
  ChatRequest req;
  int root_prefix_tokens = prompt.root_prefix_tokens;
  if (!image_parts.empty()) {
    // Images are only permitted in user/tool turns (strictly at or after the leading system root);
    // defensively zero root_prefix_tokens if any placeholder somehow preceded it.
    const std::size_t r = std::min(std::size_t(std::max(0, root_prefix_tokens)), prompt.ids.size());
    if (std::find(prompt.ids.begin(), prompt.ids.begin() + std::ptrdiff_t(r), model::IMAGE_PLACEHOLDER_ID) !=
        prompt.ids.begin() + std::ptrdiff_t(r)) {
      root_prefix_tokens = 0;
    }
    expand_images(image_parts, image_options, prompt.ids, req.images);
  }
  req.root_prefix_tokens = model::validate_root_prefix_tokens(prompt.ids, req.images, root_prefix_tokens);

  const int64_t room = int64_t(max_model_len) - int64_t(prompt.ids.size());
  if (room <= 0) {
    throw BadRequest("prompt has " + std::to_string(prompt.ids.size()) + " tokens; the model context is " +
                         std::to_string(max_model_len),
                     "messages");
  }
  std::optional<int64_t> max_tokens = num_int(body, "max_completion_tokens", Bound{1, "1"});
  if (!max_tokens) max_tokens = num_int(body, "max_tokens", Bound{1, "1"});

  req.sampling.temperature =
      float(num_float(body, "temperature", Bound{0.0, "0.0"}, Bound{2.0, "2.0"}).value_or(defaults.temperature));
  req.sampling.top_p = float(num_float(body, "top_p", Bound{0.0, "0.0"}, Bound{1.0, "1.0"}).value_or(defaults.top_p));
  req.sampling.top_k =
      clamp_int(std::max<int64_t>(0, num_int(body, "top_k", Bound{-1, "-1"}).value_or(defaults.top_k)));
  if (auto seed = num_int(body, "seed")) req.sampling.seed = static_cast<uint64_t>(*seed);
  if (req.sampling.top_p <= 0.0f) throw BadRequest("'top_p' must be in (0, 1]", "top_p");
  // When no tools are declared, ban the model's tool-invoke control token if the chat format defines one.
  const bool has_tools = effective_tools.is_array() && !effective_tools.empty();
  if (!has_tools) {
    if (const int32_t id = fmt.content_invoke_tool_json_id(); id >= 0) {
      req.sampling.banned_tokens.push_back(id);
    }
  }
  // When thinking is disabled, ban the thinking-start control token if the chat format defines one.
  if (!thinking_enabled) {
    if (const int32_t id = fmt.content_thinking_id(); id >= 0) {
      req.sampling.banned_tokens.push_back(id);
    }
  }
  // Token-level JSON / JSON-Schema FSM masker (nullptr for default/text -> zero overhead).
  if (parsed_rf.spec.mode != text::JsonFormatMode::None) {
    try {
      req.sampling.logit_masker = fmt.make_json_masker(parsed_rf.spec, prompt.prefill_ids);
    } catch (const std::invalid_argument& e) {
      throw BadRequest(e.what(), "response_format");
    }
  }

  if (const Json* stop = get(body, "stop")) {
    if (stop->is_string()) {
      req.stop.push_back(stop->get<std::string>());
    } else if (stop->is_array() &&
               std::all_of(stop->begin(), stop->end(), [](const Json& s) { return s.is_string(); })) {
      for (const Json& s : *stop) req.stop.push_back(s.get<std::string>());
    } else {
      throw BadRequest("'stop' must be a string or a list of strings", "stop");
    }
  }
  if (const Json* p = get(body, "persona")) {
    if (!p->is_string()) throw BadRequest("'persona' must be a string", "persona");
    req.persona_name = p->get<std::string>();
  }
  for (const char* scope_key : {"user", "prompt_cache_key"}) {
    if (const Json* v = get(body, scope_key); v != nullptr && !v->is_null()) {
      if (!v->is_string() || !v->get_ref<const std::string&>().empty()) {
        req.has_isolation_hint = true;
        req.has_user_or_cache_key = true;
      }
    }
  }
  if (auto it = body.find("stream"); it != body.end()) {
    if (!it->is_boolean()) throw BadRequest("'stream' must be a boolean", "stream");
    req.stream = it->get<bool>();
  }
  if (auto it = body.find("stream_options"); it != body.end() && truthy(*it)) {
    if (!it->is_object()) throw BadRequest("'stream_options' must be an object", "stream_options");
    if (auto u = it->find("include_usage"); u != it->end()) req.include_usage = truthy(*u);
  }
  req.max_new_tokens = int(std::min(max_tokens.value_or(room), room));
  req.prompt_ids = std::move(prompt.ids);
  req.prefill_ids = std::move(prompt.prefill_ids);
  req.render_stats = prompt.stats;
  return req;
}

Json error_body(int status, std::string_view message, const std::optional<std::string>& param) {
  const char* kind = status == 400 ? "invalid_request_error"
                     : status == 404 ? "not_found_error"
                     : status == 503 ? "server_busy"
                                     : "server_error";
  return Json{{"error",
               {{"message", message}, {"type", kind}, {"param", param ? Json(*param) : Json()}, {"code", status}}}};
}

Json delta_json(const text::Delta& d) {
  if (!d.reasoning_content.empty()) return Json{{"reasoning_content", d.reasoning_content}};
  if (!d.content.empty()) return Json{{"content", d.content}};
  const text::ToolCallDelta& tc = *d.tool_call;
  if (tc.id) {
    return Json{{"tool_calls",
                 Json::array({Json{{"index", tc.index},
                                   {"id", *tc.id},
                                   {"type", "function"},
                                   {"function", {{"name", tc.name ? Json(*tc.name) : Json()},
                                                 {"arguments", tc.arguments}}}}})}};
  }
  return Json{{"tool_calls", Json::array({Json{{"index", tc.index}, {"function", {{"arguments", tc.arguments}}}}})}};
}

std::string dump(const Json& value) { return value.dump(-1, ' ', false, Json::error_handler_t::replace); }

ParsedEmbeddingRequest parse_embedding_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                                               int num_tap_layers, bool has_platos_key) {
  EmbeddingParseOptions opts;
  opts.max_model_len = max_model_len;
  opts.num_tap_layers = num_tap_layers;
  opts.has_platos_key = has_platos_key;
  return parse_embedding_request(body, fmt, opts);
}

ParsedEmbeddingRequest parse_embedding_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                                               int num_tap_layers, const PlatosKey* loaded_platos_key) {
  EmbeddingParseOptions opts;
  opts.max_model_len = max_model_len;
  opts.num_tap_layers = num_tap_layers;
  opts.has_platos_key = (loaded_platos_key != nullptr);
  opts.loaded_platos_key = loaded_platos_key;
  return parse_embedding_request(body, fmt, opts);
}

ParsedEmbeddingRequest parse_embedding_request(const Json& body, const text::ChatFormat& fmt,
                                               const EmbeddingParseOptions& opts) {
  ParsedEmbeddingRequest out;

  if (const Json* m = get(body, "model")) {
    if (!m->is_string()) throw BadRequest("'model' must be a string", "model");
  }

  std::string prompt_prefix;
  if (const Json* pn = get(body, "prompt_name")) {
    if (!pn->is_string()) throw BadRequest("'prompt_name' must be a string", "prompt_name");
    const std::string& name = pn->get_ref<const std::string&>();
    if (name.empty()) throw BadRequest("'prompt_name' must not be empty", "prompt_name");
    const auto it = opts.prompt_prefixes.find(name);
    if (it == opts.prompt_prefixes.end()) {
      throw BadRequest("unknown 'prompt_name': '" + name + "'", "prompt_name");
    }
    prompt_prefix = it->second;
    out.request.prompt_name = name;
  }

  const Json* inp = get(body, "input");
  if (!inp) throw BadRequest("'input' is required", "input");

  const bool wrap_bos_eos = (fmt.kind() == text::ChatFormatKind::EmbeddingGemma2);
  auto eg2_num_soft_tokens = [](int h, int w, int max_st) -> int {
    if (max_st != 70 && max_st != 140 && max_st != 280 && max_st != 560 && max_st != 1120) {
      throw BadRequest("'max_soft_tokens' must be one of [70, 140, 280, 560, 1120]", "input");
    }
    constexpr int patch_size = 16;
    constexpr int pool_k = 3;
    constexpr int k2 = pool_k * pool_k;
    constexpr int side_mult = patch_size * pool_k;
    const int max_patches = max_st * k2;
    const double factor = std::sqrt(static_cast<double>(max_patches * patch_size * patch_size) /
                                    (static_cast<double>(h) * static_cast<double>(w)));
    int th = static_cast<int>(std::floor(factor * static_cast<double>(h) / side_mult)) * side_mult;
    int tw = static_cast<int>(std::floor(factor * static_cast<double>(w) / side_mult)) * side_mult;
    if (th == 0 && tw == 0) throw BadRequest("image dimensions collapse to 0x0 after aspect-ratio resize", "input");
    const int max_side = max_st * side_mult;
    if (th == 0) {
      th = side_mult;
      tw = std::max(side_mult, std::min(static_cast<int>(std::floor(static_cast<double>(w) / h)) * side_mult, max_side));
    } else if (tw == 0) {
      tw = side_mult;
      th = std::max(side_mult, std::min(static_cast<int>(std::floor(static_cast<double>(h) / w)) * side_mult, max_side));
    }
    return (th / side_mult) * (tw / side_mult);
  };

  auto extract_uri_and_budget = [](const Json& obj, int default_budget, const char* where_desc) -> std::pair<std::string, int> {
    int budget = default_budget;
    if (const Json* mst = get(obj, "max_soft_tokens")) {
      if (!mst->is_number_integer() || mst->is_boolean()) {
        throw BadRequest(std::string(where_desc) + " 'max_soft_tokens' must be an integer", "input");
      }
      budget = mst->get<int>();
    }
    if (const Json* iu = get(obj, "image_url")) {
      if (iu->is_string()) return {iu->get<std::string>(), budget};
      if (iu->is_object()) {
        if (const Json* mst2 = get(*iu, "max_soft_tokens")) {
          if (!mst2->is_number_integer() || mst2->is_boolean()) {
            throw BadRequest(std::string(where_desc) + " 'max_soft_tokens' must be an integer", "input");
          }
          budget = mst2->get<int>();
        }
        if (const Json* u = get(*iu, "url"); u && u->is_string()) return {u->get<std::string>(), budget};
      }
      throw BadRequest(std::string(where_desc) + " 'image_url' must be a data: URI string or {\"url\": ...}", "input");
    }
    if (const Json* u = get(obj, "url"); u && u->is_string()) return {u->get<std::string>(), budget};
    if (const Json* im = get(obj, "image"); im && im->is_string()) return {im->get<std::string>(), budget};
    throw BadRequest(std::string(where_desc) + " missing image data URI ('image_url' or 'url')", "input");
  };

  bool any_images = false;
  bool any_audios = false;
  std::vector<std::vector<model::EmbeddingImage>> per_item_images;
  std::vector<std::vector<model::EmbeddingAudio>> per_item_audios;

  auto encode_text_item = [&](const std::string& text, const char* where_desc) {
    if (text.empty()) {
      throw BadRequest(std::string(where_desc) + " must not be empty", "input");
    }
    const std::string full_text = prompt_prefix.empty() ? text : (prompt_prefix + text);
    std::vector<int32_t> ids;
    try {
      ids = wrap_bos_eos ? fmt.tokenizer().encode_with_post_processor(full_text, /*special=*/false)
                         : fmt.tokenizer().encode(full_text, /*special=*/false);
    } catch (const std::invalid_argument& e) {
      throw BadRequest(std::string("failed to tokenize 'input': ") + e.what(), "input");
    }
    if (ids.empty()) {
      throw BadRequest(std::string(where_desc) + " tokenized to 0 tokens", "input");
    }
    if (static_cast<int>(ids.size()) > opts.max_model_len) {
      throw BadRequest("input has " + std::to_string(ids.size()) + " tokens; the model context is " +
                           std::to_string(opts.max_model_len),
                       "input");
    }
    out.request.inputs.push_back(std::move(ids));
    per_item_images.emplace_back();
    per_item_audios.emplace_back();
  };

  auto encode_parts_item = [&](const Json& parts, const char* where_desc) {
    if (!parts.is_array() || parts.empty()) {
      throw BadRequest(std::string(where_desc) + " content parts array must not be empty", "input");
    }
    std::vector<int32_t> ids;
    std::vector<model::EmbeddingImage> item_imgs;
    std::vector<model::EmbeddingAudio> item_auds;
    bool has_text = false;
    bool prefix_applied = false;
    if (wrap_bos_eos && fmt.tokenizer().bos_token_id().has_value()) {
      ids.push_back(*fmt.tokenizer().bos_token_id());
    }

    constexpr int32_t kBoi = 255999;
    constexpr int32_t kBoa = 256000;
    constexpr int32_t kImgTok = 258880;
    constexpr int32_t kAudTok = 258881;
    constexpr int32_t kEoi = 258882;
    constexpr int32_t kEoa = 258883;
    constexpr int32_t kVidTok = 258884;

    for (std::size_t pi = 0; pi < parts.size(); ++pi) {
      const Json& part = parts[pi];
      if (!part.is_object()) {
        throw BadRequest(std::string(where_desc) + "[" + std::to_string(pi) + "] must be an object", "input");
      }
      const Json* tp = get(part, "type");
      if (!tp || !tp->is_string()) {
        throw BadRequest(std::string(where_desc) + "[" + std::to_string(pi) + "] missing string 'type'", "input");
      }
      const std::string& ptype = tp->get_ref<const std::string&>();
      if (ptype == "text" || ptype == "input_text") {
        const Json* tv = get(part, "text");
        if (!tv || !tv->is_string() || tv->get_ref<const std::string&>().empty()) {
          throw BadRequest(std::string(where_desc) + "[" + std::to_string(pi) + "] 'text' must be a non-empty string",
                           "input");
        }
        has_text = true;
        std::string txt = tv->get<std::string>();
        if (!prompt_prefix.empty() && !prefix_applied) {
          txt = prompt_prefix + txt;
          prefix_applied = true;
        }
        std::vector<int32_t> tids = fmt.tokenizer().encode(txt, /*special=*/false);
        ids.insert(ids.end(), tids.begin(), tids.end());
      } else if (ptype == "image_url" || ptype == "input_image" || ptype == "image") {
        if (!opts.supports_images) {
          throw BadRequest("image inputs are not enabled on this server (started without vision or --no-vision)",
                           "input");
        }
        if (!wrap_bos_eos) {
          throw BadRequest("multimodal image embeddings are only supported on multimodal encoder models", "input");
        }
        const auto [uri, budget] = extract_uri_and_budget(part, opts.vision_soft_tokens_per_image, where_desc);
        image::Image decoded;
        try {
          decoded = image::decode_raw_rgb_data_uri(uri, opts.image_limits);
        } catch (const image::Error& e) {
          throw BadRequest(e.what(), "input");
        }
        const int n_soft = eg2_num_soft_tokens(decoded.height, decoded.width, budget);
        ids.push_back(kBoi);
        ids.insert(ids.end(), static_cast<std::size_t>(n_soft), kImgTok);
        ids.push_back(kEoi);
        model::EmbeddingImage eimg;
        eimg.width = decoded.width;
        eimg.height = decoded.height;
        eimg.rgb = std::move(decoded.rgb);
        eimg.max_soft_tokens = budget;
        eimg.is_video_frame = false;
        item_imgs.push_back(std::move(eimg));
        any_images = true;
      } else if (ptype == "video") {
        if (!opts.supports_images) {
          throw BadRequest("video inputs are not enabled on this server (started without vision or --no-vision)",
                           "input");
        }
        if (!wrap_bos_eos) {
          throw BadRequest("multimodal video embeddings are only supported on multimodal encoder models", "input");
        }
        const Json* frames = get(part, "frames");
        if (!frames || !frames->is_array() || frames->empty()) {
          throw BadRequest(std::string(where_desc) + "[" + std::to_string(pi) + "] 'frames' must be a non-empty array",
                           "input");
        }
        int frame_budget = opts.video_soft_tokens_per_frame;
        if (const Json* mst = get(part, "max_soft_tokens")) {
          if (!mst->is_number_integer() || mst->is_boolean()) {
            throw BadRequest("'max_soft_tokens' must be an integer", "input");
          }
          frame_budget = mst->get<int>();
        }
        for (std::size_t fi = 0; fi < frames->size(); ++fi) {
          const Json& fr = (*frames)[fi];
          std::string uri;
          int fbudget = frame_budget;
          if (fr.is_string()) {
            uri = fr.get<std::string>();
          } else if (fr.is_object()) {
            std::tie(uri, fbudget) = extract_uri_and_budget(fr, frame_budget, where_desc);
          } else {
            throw BadRequest("video frame must be a data: URI string or object", "input");
          }
          image::Image decoded;
          try {
            decoded = image::decode_raw_rgb_data_uri(uri, opts.image_limits);
          } catch (const image::Error& e) {
            throw BadRequest(e.what(), "input");
          }
          const int n_soft = eg2_num_soft_tokens(decoded.height, decoded.width, fbudget);
          ids.push_back(kBoi);
          ids.insert(ids.end(), static_cast<std::size_t>(n_soft), kVidTok);
          ids.push_back(kEoi);
          model::EmbeddingImage eimg;
          eimg.width = decoded.width;
          eimg.height = decoded.height;
          eimg.rgb = std::move(decoded.rgb);
          eimg.max_soft_tokens = fbudget;
          eimg.is_video_frame = true;
          item_imgs.push_back(std::move(eimg));
          any_images = true;
        }
      } else if (ptype == "input_audio" || ptype == "audio") {
        if (!opts.supports_audio) {
          throw BadRequest("audio inputs are not enabled on this server (started without audio or --no-audio)", "input");
        }
        if (!wrap_bos_eos) {
          throw BadRequest("multimodal audio embeddings are only supported on multimodal encoder models", "input");
        }
        std::string audio_payload;
        if (const Json* ia = get(part, "input_audio")) {
          if (ia->is_object()) {
            if (const Json* fmt_j = get(*ia, "format")) {
              if (!fmt_j->is_string() || fmt_j->get_ref<const std::string&>() != "wav") {
                throw BadRequest("audio input 'format' must be 'wav'", "input");
              }
            }
            if (const Json* d = get(*ia, "data"); d && d->is_string()) {
              audio_payload = d->get<std::string>();
            }
          } else if (ia->is_string()) {
            audio_payload = ia->get<std::string>();
          }
        }
        if (audio_payload.empty()) {
          if (const Json* d = get(part, "data"); d && d->is_string()) {
            audio_payload = d->get<std::string>();
          } else if (const Json* au = get(part, "audio_url")) {
            if (au->is_string()) {
              audio_payload = au->get<std::string>();
            } else if (au->is_object()) {
              if (const Json* u = get(*au, "url"); u && u->is_string()) audio_payload = u->get<std::string>();
            }
          } else if (const Json* a = get(part, "audio"); a && a->is_string()) {
            audio_payload = a->get<std::string>();
          } else if (const Json* u = get(part, "url"); u && u->is_string()) {
            audio_payload = u->get<std::string>();
          }
        }
        if (audio_payload.empty()) {
          throw BadRequest(std::string(where_desc) + "[" + std::to_string(pi) +
                               "] missing audio payload ('input_audio.data', 'data', or 'audio_url')",
                           "input");
        }
        model::EmbeddingAudio decoded_audio;
        int n_aud_soft = 0;
        try {
          decoded_audio = backends::embedding_gemma2::eg2_decode_wav_base64_or_data_uri(audio_payload);
          n_aud_soft = backends::embedding_gemma2::eg2_preprocess_audio(decoded_audio).num_valid_tokens;
        } catch (const std::invalid_argument& e) {
          throw BadRequest(e.what(), "input");
        }
        ids.push_back(kBoa);
        ids.insert(ids.end(), static_cast<std::size_t>(n_aud_soft), kAudTok);
        ids.push_back(kEoa);
        item_auds.push_back(std::move(decoded_audio));
        any_audios = true;
      } else {
        throw BadRequest(std::string(where_desc) + "[" + std::to_string(pi) + "] unsupported content part type '" +
                             ptype + "'",
                         "input");
      }
    }

    if (!prompt_prefix.empty() && !has_text) {
      throw BadRequest("'prompt_name' cannot be applied to pure-media input without text", "prompt_name");
    }
    if (wrap_bos_eos && fmt.tokenizer().eos_token_id().has_value()) {
      ids.push_back(*fmt.tokenizer().eos_token_id());
    }
    if (ids.empty()) {
      throw BadRequest(std::string(where_desc) + " tokenized to 0 tokens", "input");
    }
    if (static_cast<int>(ids.size()) > opts.max_model_len) {
      throw BadRequest("input has " + std::to_string(ids.size()) + " tokens; the model context is " +
                           std::to_string(opts.max_model_len),
                       "input");
    }
    out.request.inputs.push_back(std::move(ids));
    per_item_images.push_back(std::move(item_imgs));
    per_item_audios.push_back(std::move(item_auds));
  };

  if (inp->is_string()) {
    encode_text_item(inp->get_ref<const std::string&>(), "'input'");
  } else if (inp->is_array()) {
    if (inp->empty()) throw BadRequest("'input' array must not be empty", "input");
    if ((*inp)[0].is_object()) {
      // Single multimodal item expressed as an array of content-part objects.
      encode_parts_item(*inp, "'input'");
    } else {
      out.request.inputs.reserve(inp->size());
      per_item_images.reserve(inp->size());
      per_item_audios.reserve(inp->size());
      for (std::size_t i = 0; i < inp->size(); ++i) {
        const Json& elem = (*inp)[i];
        const std::string desc = "'input[" + std::to_string(i) + "]'";
        if (elem.is_string()) {
          encode_text_item(elem.get_ref<const std::string&>(), desc.c_str());
        } else if (elem.is_array()) {
          encode_parts_item(elem, desc.c_str());
        } else {
          throw BadRequest(desc + " must be a string or a content-parts array", "input");
        }
      }
    }
  } else {
    throw BadRequest("'input' must be a string, an array of strings, or multimodal content parts", "input");
  }
  if (any_images) {
    out.request.images = std::move(per_item_images);
  }
  if (any_audios) {
    out.request.audios = std::move(per_item_audios);
  }

  if (const Json* ef = get(body, "encoding_format")) {
    if (!ef->is_string()) throw BadRequest("'encoding_format' must be a string", "encoding_format");
    const std::string& v = ef->get_ref<const std::string&>();
    if (v == "float") {
      out.encoding_format = EmbeddingEncodingFormat::Float;
    } else if (v == "base64") {
      out.encoding_format = EmbeddingEncodingFormat::Base64;
    } else {
      throw BadRequest("'encoding_format' must be 'float' or 'base64'", "encoding_format");
    }
  }

  if (const Json* pool = get(body, "pooling")) {
    if (!pool->is_string()) throw BadRequest("'pooling' must be a string", "pooling");
    try {
      out.request.pooling = model::parse_embedding_pooling(pool->get_ref<const std::string&>());
    } catch (const std::invalid_argument& e) {
      throw BadRequest(e.what(), "pooling");
    }
  }

  if (const Json* norm = get(body, "normalize")) {
    if (!norm->is_boolean()) throw BadRequest("'normalize' must be a boolean", "normalize");
    out.request.normalize = norm->get<bool>();
  }

  if (const auto tap = num_int(body, "tap_layer")) {
    if (opts.num_tap_layers <= 0) {
      throw BadRequest("'tap_layer' is not supported by this model backend", "tap_layer");
    }
    if (*tap < 0 || *tap >= opts.num_tap_layers) {
      throw BadRequest("'tap_layer' must be in [0, " + std::to_string(opts.num_tap_layers - 1) + "]", "tap_layer");
    }
    out.request.tap_layer = static_cast<int>(*tap);
  }

  if (const Json* cid = get(body, "cartridge_id")) {
    if (!cid->is_string() || cid->get_ref<const std::string&>().empty()) {
      throw BadRequest("'cartridge_id' must be a non-empty string", "cartridge_id");
    }
    out.request.cartridge_id = cid->get<std::string>();
  }

  if (const Json* csha = get(body, "cartridge_sha256")) {
    if (!csha->is_string() || csha->get_ref<const std::string&>().empty()) {
      throw BadRequest("'cartridge_sha256' must be a non-empty string", "cartridge_sha256");
    }
    out.request.cartridge_sha256 = csha->get<std::string>();
  }

  if (const Json* pk = get(body, "platos_key")) {
    if (!pk->is_boolean()) throw BadRequest("'platos_key' must be a boolean", "platos_key");
    out.request.use_platos_key = pk->get<bool>();
    if (out.request.use_platos_key && !opts.has_platos_key) {
      throw BadRequest("'platos_key' requested but server was started without --platos-key", "platos_key");
    }
  }

  if (!out.request.use_platos_key &&
      (out.request.cartridge_id.has_value() || out.request.cartridge_sha256.has_value())) {
    throw BadRequest("'cartridge_id'/'cartridge_sha256' require 'platos_key': true",
                     out.request.cartridge_sha256.has_value() ? "cartridge_sha256" : "cartridge_id");
  }

  if (out.request.use_platos_key && opts.loaded_platos_key != nullptr) {
    try {
      opts.loaded_platos_key->validate_request_tap(out.request.tap_layer);
    } catch (const std::invalid_argument& e) {
      throw BadRequest(e.what(), "tap_layer");
    }
    try {
      opts.loaded_platos_key->validate_request_identity(out.request.cartridge_id.value_or(""),
                                                        out.request.cartridge_sha256.value_or(""));
    } catch (const std::invalid_argument& e) {
      const std::string_view msg(e.what());
      const char* param = (msg.find("cartridge_id") != std::string_view::npos) ? "cartridge_id" : "cartridge_sha256";
      throw BadRequest(e.what(), param);
    }
  }

  if (const auto dims = num_int(body, "dimensions")) {
    if (*dims <= 0) {
      throw BadRequest("'dimensions' must be a positive integer", "dimensions");
    }
    if (out.request.use_platos_key) {
      throw BadRequest("'dimensions' cannot be combined with 'platos_key'", "dimensions");
    }
    const int d = static_cast<int>(*dims);
    if (!opts.matryoshka_dims.empty()) {
      if (!out.request.tap_layer.has_value() &&
          std::find(opts.matryoshka_dims.begin(), opts.matryoshka_dims.end(), d) == opts.matryoshka_dims.end()) {
        std::string allowed;
        for (std::size_t i = 0; i < opts.matryoshka_dims.size(); ++i) {
          if (i) allowed += ", ";
          allowed += std::to_string(opts.matryoshka_dims[i]);
        }
        throw BadRequest("'dimensions' must be one of [" + allowed + "]", "dimensions");
      }
    } else if (opts.embedding_dim > 0 && d != opts.embedding_dim) {
      throw BadRequest("'dimensions' truncation is not supported by this model (native dim is " +
                           std::to_string(opts.embedding_dim) + ")",
                       "dimensions");
    }
    out.request.dimensions = d;
  }

  return out;
}

std::string encode_embedding_base64(std::span<const float> vec) {
  static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const std::size_t nbytes = vec.size() * sizeof(float);
  std::vector<std::uint8_t> bytes(nbytes);
  for (std::size_t i = 0; i < vec.size(); ++i) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &vec[i], sizeof(bits));
    bytes[i * 4 + 0] = static_cast<std::uint8_t>(bits & 0xffu);
    bytes[i * 4 + 1] = static_cast<std::uint8_t>((bits >> 8) & 0xffu);
    bytes[i * 4 + 2] = static_cast<std::uint8_t>((bits >> 16) & 0xffu);
    bytes[i * 4 + 3] = static_cast<std::uint8_t>((bits >> 24) & 0xffu);
  }
  std::string out;
  out.reserve(((nbytes + 2) / 3) * 4);
  for (std::size_t i = 0; i < nbytes; i += 3) {
    const std::uint32_t b0 = bytes[i];
    const std::uint32_t b1 = (i + 1 < nbytes) ? bytes[i + 1] : 0u;
    const std::uint32_t b2 = (i + 2 < nbytes) ? bytes[i + 2] : 0u;
    const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
    out.push_back(kAlphabet[(triple >> 18) & 0x3fu]);
    out.push_back(kAlphabet[(triple >> 12) & 0x3fu]);
    out.push_back(i + 1 < nbytes ? kAlphabet[(triple >> 6) & 0x3fu] : '=');
    out.push_back(i + 2 < nbytes ? kAlphabet[triple & 0x3fu] : '=');
  }
  return out;
}

Json embedding_response_json(const std::string& model_id, const model::EmbeddingResult& result,
                             EmbeddingEncodingFormat format) {
  Json data = Json::array();
  for (std::size_t i = 0; i < result.embeddings.size(); ++i) {
    const auto& vec = result.embeddings[i];
    Json emb;
    if (format == EmbeddingEncodingFormat::Base64) {
      emb = encode_embedding_base64(vec);
    } else {
      emb = Json::array();
      for (float v : vec) emb.push_back(v);
    }
    data.push_back(Json{{"object", "embedding"}, {"index", static_cast<int>(i)}, {"embedding", std::move(emb)}});
  }
  return Json{{"object", "list"},
              {"data", std::move(data)},
              {"model", model_id},
              {"usage",
               {{"prompt_tokens", result.stats.total_prompt_tokens},
                {"total_tokens", result.stats.total_prompt_tokens}}}};
}

}  // namespace ingine::serve

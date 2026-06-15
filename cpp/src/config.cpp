#include "itu/config.hpp"

#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace itu {

YAML::Node deep_merge(const YAML::Node& base, const YAML::Node& override) {
  // Scalars and sequences replace; only two maps merge recursively.
  if (!base.IsMap() || !override.IsMap()) return YAML::Clone(override);

  YAML::Node out(YAML::NodeType::Map);
  std::set<std::string> base_keys;
  for (const auto& kv : base) {
    auto key = kv.first.as<std::string>();
    base_keys.insert(key);
    out[key] = YAML::Clone(kv.second);
  }
  for (const auto& kv : override) {
    auto key = kv.first.as<std::string>();
    const YAML::Node oval = kv.second;
    if (base_keys.count(key) && base[key].IsMap() && oval.IsMap()) {
      out[key] = deep_merge(base[key], oval);
    } else {
      out[key] = YAML::Clone(oval);
    }
  }
  return out;
}

YAML::Node load_config(const fs::path& path) {
  YAML::Node raw = YAML::LoadFile(path.string());
  if (!raw.IsMap()) return raw;

  std::vector<std::string> includes;
  if (raw["include"] && raw["include"].IsSequence()) {
    for (const auto& inc : raw["include"]) includes.push_back(inc.as<std::string>());
  }

  // `raw` minus its own `include:` key (Python pops it before merging).
  YAML::Node raw_no_inc(YAML::NodeType::Map);
  for (const auto& kv : raw) {
    auto key = kv.first.as<std::string>();
    if (key == "include") continue;
    raw_no_inc[key] = YAML::Clone(kv.second);
  }

  YAML::Node merged(YAML::NodeType::Map);
  for (const auto& inc : includes) {
    merged = deep_merge(merged, load_config(path.parent_path() / inc));
  }
  return deep_merge(merged, raw_no_inc);
}

}  // namespace itu

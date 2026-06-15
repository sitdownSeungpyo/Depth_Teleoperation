// Config loader with file includes + deep-merge. Mirrors core/config.py.
#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>

namespace itu {

// Load a YAML config, resolving and deep-merging any top-level `include:` files
// (paths relative to the including file). Merge rule: nested maps merge
// recursively; scalars and sequences replace. The including file's own keys are
// merged on top last, so it can override included values.
YAML::Node load_config(const std::filesystem::path& path);

// Deep-merge `override` onto `base` (returns a new node). Exposed for tests.
YAML::Node deep_merge(const YAML::Node& base, const YAML::Node& override);

}  // namespace itu

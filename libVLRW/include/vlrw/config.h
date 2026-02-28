#pragma once

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vlrw/types.h>

namespace vlrw {

// Simple INI config parser
class Config {
public:
    Config() = default;

    // Load from INI file
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        std::string line, section;
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }

            // Section header
            if (line[0] == '[' && line.back() == ']') {
                section = line.substr(1, line.size() - 2);
                continue;
            }

            // Key = value
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                
                // Trim key and value
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                std::string fullKey = section.empty() ? key : section + "." + key;
                m_data[fullKey] = value;
            }
        }

        return true;
    }

    // Get string value
    std::string getString(const std::string& key, const std::string& defaultValue = "") const {
        auto it = m_data.find(key);
        return (it != m_data.end()) ? it->second : defaultValue;
    }

    // Get int value
    int getInt(const std::string& key, int defaultValue = 0) const {
        auto it = m_data.find(key);
        if (it != m_data.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    // Get float value
    float getFloat(const std::string& key, float defaultValue = 0.0f) const {
        auto it = m_data.find(key);
        if (it != m_data.end()) {
            try {
                return std::stof(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    // Get RGB value (comma-separated: r, g, b)
    RGB getRGB(const std::string& key, const RGB& defaultValue = RGB::black()) const {
        auto it = m_data.find(key);
        if (it != m_data.end()) {
            std::istringstream ss(it->second);
            float r, g, b;
            char comma;
            if (ss >> r >> comma >> g >> comma >> b) {
                return RGB(r, g, b);
            }
        }
        return defaultValue;
    }

    // Get bool value
    bool getBool(const std::string& key, bool defaultValue = false) const {
        auto it = m_data.find(key);
        if (it != m_data.end()) {
            std::string val = it->second;
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            return (val == "true" || val == "1" || val == "yes" || val == "on");
        }
        return defaultValue;
    }

private:
    std::map<std::string, std::string> m_data;
};

} // namespace vlrw

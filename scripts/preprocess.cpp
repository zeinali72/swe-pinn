#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <cctype>

// A struct to hold the final output data.
struct FinalDataRow {
    double t, x, y, h, u, v;
};

// Helper function to trim leading/trailing whitespace and quotes.
std::string trim(const std::string& s) {
    size_t first = s.find_first_not_of(" \t\n\r\"");
    if (std::string::npos == first) {
        return s;
    }
    size_t last = s.find_last_not_of(" \t\n\r\"");
    return s.substr(first, (last - first + 1));
}

// Manually parses a CSV line to handle complex cases with quotes and whitespace.
std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> result;
    std::string current_field;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            result.push_back(trim(current_field));
            current_field.clear();
        } else {
            current_field += c;
        }
    }
    result.push_back(trim(current_field));
    return result;
}

// Function to save the final data to a binary file
void save_to_binary(const std::string& output_path, const std::vector<FinalDataRow>& data) {
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Cannot open output file " << output_path << std::endl;
        return;
    }
    for (const auto& row : data) {
        out.write(reinterpret_cast<const char*>(&row), sizeof(FinalDataRow));
    }
    out.close();
}


int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <angle_path> <depth_path> <speed_path> <output_path>" << std::endl;
        return 1;
    }

    std::string angle_path = argv[1];
    std::string depth_path = argv[2];
    std::string speed_path = argv[3];
    std::string output_path = argv[4];

    std::cout << "🚀 Starting C++ data preparation (Optimized Row-by-Row)..." << std::endl;

    std::ifstream angle_file(angle_path);
    std::ifstream depth_file(depth_path);
    std::ifstream speed_file(speed_path);

    if (!angle_file.is_open() || !depth_file.is_open() || !speed_file.is_open()) {
        std::cerr << "Error: Could not open one or more input files." << std::endl;
        return 1;
    }

    std::vector<FinalDataRow> final_data;

    // --- Read Headers ---
    std::string header_line_angle, header_line_depth, header_line_speed;
    if (!std::getline(angle_file, header_line_angle) || !std::getline(depth_file, header_line_depth) || !std::getline(speed_file, header_line_speed)) {
        std::cerr << "Error: Could not read headers from one or more files." << std::endl;
        return 1;
    }
    
    // We assume all headers are the same, so we only parse one.
    std::vector<std::string> headers = parse_csv_line(header_line_angle);

    // --- Process Data Rows Synchronously ---
    std::string line_angle, line_depth, line_speed;
    while (std::getline(angle_file, line_angle) && std::getline(depth_file, line_depth) && std::getline(speed_file, line_speed)) {
        
        // Skip empty or malformed lines
        if (line_angle.empty() || line_depth.empty() || line_speed.empty()) continue;

        std::vector<std::string> values_angle = parse_csv_line(line_angle);
        std::vector<std::string> values_depth = parse_csv_line(line_depth);
        std::vector<std::string> values_speed = parse_csv_line(line_speed);

        if (values_angle.size() < headers.size() || values_depth.size() < headers.size() || values_speed.size() < headers.size()) {
            continue; // Skip rows that don't have enough columns
        }

        double seconds;
        try {
            seconds = std::stod(values_angle[1]); // Index 1 is 'Seconds'
        } catch (const std::exception&) {
            continue; // Skip if 'Seconds' is not a valid number
        }

        // Iterate through the data columns (skip Time and Seconds)
        for (size_t i = 2; i < headers.size(); ++i) {
            try {
                // Parse coordinates from the header
                std::string coord_header = headers[i];
                std::stringstream coord_ss(coord_header);
                std::string x_str, y_str;
                double x_coord, y_coord;

                if (std::getline(coord_ss, x_str, ',') && std::getline(coord_ss, y_str)) {
                    x_coord = std::stod(trim(x_str));
                    y_coord = std::stod(trim(y_str));
                } else {
                    continue; // Skip if coordinate header is malformed
                }
                
                // Get values from each file for the current column
                double angle_rad = std::stod(values_angle[i]);
                double h = std::stod(values_depth[i]);
                double speed = std::stod(values_speed[i]);

                // Calculate u and v and add the final row
                final_data.push_back({
                    seconds,
                    x_coord,
                    y_coord,
                    h,
                    speed * std::cos(angle_rad), // u
                    speed * std::sin(angle_rad)  // v
                });

            } catch (const std::exception&) {
                // This will catch errors from stod if a value is not a number.
                // We simply skip that data point and continue.
                continue;
            }
        }
    }

    angle_file.close();
    depth_file.close();
    speed_file.close();
    
    // --- Save the final data ---
    std::cout << "Finished processing. Total data points generated: " << final_data.size() << std::endl;
    std::filesystem::path out_p(output_path);
    if (!out_p.parent_path().empty()) {
        std::filesystem::create_directories(out_p.parent_path());
    }
    save_to_binary(output_path, final_data);

    std::cout << "\n--- Data Preparation Complete ---" << std::endl;
    std::cout << "Final array shape: (" << final_data.size() << ", 6)" << std::endl;
    std::cout << "Columns in array: t, x, y, h, u, v" << std::endl;
    std::cout << "Output saved to '" << output_path << "'" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    return 0;
}


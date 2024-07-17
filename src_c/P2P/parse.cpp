#include "parse.hpp"

std::vector<BenchmarkData> parse_file(std::string &filename) {
    std::ifstream infile(filename); // Replace with your file name
    if (!infile) {
        std::cerr << "Error opening file.\n";
        return {};
    }

    std::vector<BenchmarkData> benchmarkMap;

    std::string line;
    while (std::getline(infile, line)) {
        std::regex pattern("\t*\\s*#");
        std::smatch matches;

        if (std::regex_search(line, matches, pattern)) // Skip comment lines
        {
            continue;
        }

        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue; // Skip empty lines
        }

        std::istringstream iss(line);
        int bytes, repetitions;
        double time_usec, mb_per_sec;
        long long msg_per_sec;

        if (!(iss >> bytes >> repetitions >> time_usec >> mb_per_sec >> msg_per_sec)) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }

        BenchmarkData data { bytes, repetitions, time_usec, mb_per_sec, msg_per_sec };
        benchmarkMap.push_back(data);
    }

    return benchmarkMap;
}

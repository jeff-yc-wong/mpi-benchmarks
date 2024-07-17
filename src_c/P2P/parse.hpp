#ifndef PARSE_HEADER_INCLUDED
#define PARSE_HEADER_INCLUDED

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <regex>

struct BenchmarkData {
    int bytes;
    int repetitions;
    double time_usec;
    double mb_per_sec;
    long long msg_per_sec;
};

std::vector<BenchmarkData> parse_file(std::string &filename);

#endif /* PARSE_HEADER_INCLUDED */
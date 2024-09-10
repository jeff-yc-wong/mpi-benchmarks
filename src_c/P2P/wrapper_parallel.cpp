#include <boost/format.hpp>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
// #include <boost/lexical_cast.hpp>
#include "parse.hpp"
#include "simgrid/s4u.hpp"
#include "smpi/smpi.h"
#include <boost/algorithm/string.hpp>
#include <sys/wait.h>
#include <unistd.h>

namespace sg4 = simgrid::s4u;

const int BUFFER_SIZE = 1024;

class LocalData {
public:
  double threshold; /* maximal stderr requested (if positive) */
  double relstderr; /* observed stderr so far */
  double mean; /* mean of benched times, to be used if the block is disabled */
  double sum;  /* sum of benched times (to compute the mean and stderr) */
  double sum_pow2; /* sum of the square of the benched times (to compute the
                      stderr) */
  int iters;       /* amount of requested iterations */
  int count;       /* amount of iterations done so far */
  bool benching;   /* true: we are benchmarking; false: we have enough data, no
                      bench anymore */

  bool need_more_benchs() const;
};

bool LocalData::need_more_benchs() const {
  bool res =
      (count < iters) && (threshold < 0.0 || count < 2 || // not enough data
                          relstderr >= threshold);        // stderr too high yet
  // fprintf(stderr, "\r%s (count:%d sum: %f iter:%d stderr:%f thres:%f
  // mean:%fs)",
  //         (res ? "need more data" : "enough benchs"), count, sum, iters,
  //         relstderr, threshold, mean);
  return res;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    std::cerr << "Usage: " << argv[0]
              << " <platform_file> <hostfile> <executable> <benchmark> <threshold> "
                 "<max_iters> <byte_sizes> <num_procs>"
              << std::endl;
    return 1;
  }

  // Setting num_procs to run
  int num_procs = std::stoi(argv[8]);
  std::vector<int> pipe_fds((num_procs - 1) * 2);

  std::vector<pid_t> child_pids(num_procs - 1);

  // Create pipe
  for (int i = 1; i < num_procs; i++) {
    if (pipe(&pipe_fds[(i - 1) * 2]) == -1) {
      perror("pipe");
      return 1;
    }
  }

  // Creating the Simgrid engine, Loading the platform description, and turning
  // on privatization
  sg4::Engine engine(&argc, argv);
  engine.set_config("network/model:SMPI");
  engine.set_config("smpi/privatization:ON");
  // engine.set_config("smpi/display-timing:1");
  engine.set_config("smpi/tmpdir:/home/wongy/tmp");
  engine.set_config("precision/timing:1e-9");
  engine.set_config("smpi/keep-temps:0");

  engine.load_platform(argv[1]);
  SMPI_init();

  // Getting the hosts
  std::string hostfile = argv[2];
  auto hosts = engine.get_hosts_from_MPI_hostfile(hostfile);

  const std::string executable = argv[3];
  std::string benchmark = argv[4];       // grab from cmdline
  double threshold = std::stod(argv[5]); // grab from cmdline
  int max_iters = std::stoi(argv[6]);    // grab from cmdline

  std::string byte_string = argv[7];
  std::vector<std::string> byte_sizes;

  boost::split(byte_sizes, byte_string, boost::is_any_of(","));

  std::vector<std::string> final_benchmarks;

  int local_len = byte_sizes.size() / num_procs;
  int remainder = byte_sizes.size() % num_procs;

  pid_t pid;

  for (int rank = 1; rank < num_procs; rank++) {
    pid = fork();
    if (pid == 0) {
      for (int j = 0; j < (num_procs - 1) * 2; ++j) {
        if (j != (rank - 1) * 2 +
                     1) { // Close all the read pipes except the one we are
                          // using to write which is at index (rank - 1) * 2 + 1
          close(pipe_fds[j]);
        }
      }

      // Child process
      int start = rank * local_len + std::min(rank, remainder);
      int end = start + local_len + (rank < remainder ? 1 : 0);
      std::string filename = "p2p_" + std::to_string(rank) + ".log";

      for (int i = start; i < end; i++) {
        std::string bytes = byte_sizes[i];
        LocalData data = LocalData{
            threshold, // threshold
            0.0,       // relstderr
            0.0,       // mean
            0.0,       // sum
            0.0,       // sum_pow2
            max_iters, // iters
            0,         // count
            true       // benching (if we have no data, we need at least one)
        };

        std::cerr << "Benchmarking with " << bytes << " bytes" << std::endl;
        // Running the executable in a loop

        for (int i = 0; i < max_iters; i++) {

          const std::string i_str = std::to_string(i + 1);

          std::vector<std::string> my_args = {"-iter", "1", benchmark, "-msgsz",
                                              bytes};

          FILE *file = freopen(filename.c_str(), "w", stdout);
          if (!file) {
            std::cerr << "Error opening file" << std::endl;
            exit(1);
          }

          SMPI_executable_start(executable, hosts, my_args);

          engine.run();

          std::vector<BenchmarkData> benchmarkMap = parse_file(filename);

          assert(benchmarkMap.size() == 1);

          data.count++;
          double mb_per_sec = benchmarkMap[0].mb_per_sec;
          data.sum += mb_per_sec;
          data.sum_pow2 += mb_per_sec * mb_per_sec;
          double n = data.count;
          data.mean = data.sum / n;
          data.relstderr =
              std::sqrt((data.sum_pow2 / n - data.mean * data.mean)) /
              data.mean;

          if (!data.need_more_benchs()) {
            final_benchmarks.push_back(
                boost::str(boost::format("%.2f") % data.mean));
            break;
          }
        }
      }

      SMPI_finalize();
      std::cerr << "RANK[" << rank << "]: Iteration done!" << std::endl;

      std::string result = boost::algorithm::join(final_benchmarks, " ");

      write(pipe_fds[(rank - 1) * 2 + 1], result.c_str(), result.length() + 1);

      std::cerr << "RANK[" << rank << "]: Done writing to pipe!" << std::endl;

      _exit(0);
    } else if (pid > 0) {
      child_pids[rank - 1] = pid;
      close(pipe_fds[(rank - 1) * 2 + 1]); // Close the write end of the pipe
    } else {
      perror("fork");
      return 1;
    }
  }

  // Parent process
  int start = 0 * local_len + std::min(0, remainder);
  int end = start + local_len + (0 < remainder ? 1 : 0);
  std::string filename = "p2p_" + std::to_string(0) + ".log";

  FILE *original_stdout = fdopen(dup(fileno(stdout)), "w");

  for (int i = start; i < end; i++) {
    std::string bytes = byte_sizes[start + i];
    LocalData data = LocalData{
        threshold, // threshold
        0.0,       // relstderr
        0.0,       // mean
        0.0,       // sum
        0.0,       // sum_pow2
        max_iters, // iters
        0,         // count
        true       // benching (if we have no data, we need at least one)
    };

    std::cerr << "Benchmarking with " << bytes << " bytes" << std::endl;
    // Running the executable in a loop

    for (int i = 0; i < max_iters; i++) {

      const std::string i_str = std::to_string(i + 1);

      std::vector<std::string> my_args = {"-iter", "1", benchmark, "-msgsz",
                                          bytes};

      FILE *file = freopen(filename.c_str(), "w", stdout);
      if (!file) {
        std::cerr << "RANK[0]: Error opening file" << std::endl;
        exit(1);
      }

      SMPI_executable_start(executable, hosts, my_args);

      // fflush(stdout);
      engine.run();

      std::vector<BenchmarkData> benchmarkMap = parse_file(filename);

      assert(benchmarkMap.size() == 1);

      data.count++;
      double mb_per_sec = benchmarkMap[0].mb_per_sec;
      data.sum += mb_per_sec;
      data.sum_pow2 += mb_per_sec * mb_per_sec;
      double n = data.count;
      data.mean = data.sum / n;
      data.relstderr =
          std::sqrt((data.sum_pow2 / n - data.mean * data.mean)) / data.mean;

      if (!data.need_more_benchs()) {
        final_benchmarks.push_back(
            boost::str(boost::format("%.2f") % data.mean));
        break;
      }
    }
  }

  fflush(stdout);
  stdout = original_stdout;

  SMPI_finalize();
  std::cerr << "RANK[0]: Iteration done!" << std::endl;

  std::string result = boost::algorithm::join(final_benchmarks, " ");

  char buffer[BUFFER_SIZE];
  for (int i = 0; i < num_procs - 1; i++) {
    ssize_t bytesRead;
    while ((bytesRead = read(pipe_fds[i * 2], buffer, sizeof(buffer) - 1)) >
           0) {
      buffer[bytesRead] = '\0';
      result = result + " " + buffer;
    }
  }

  fprintf(stdout, "%s\n", result.c_str());

  for (int i = 0; i < num_procs - 1; i++) {
    waitpid(child_pids[i], NULL, 0);
  }

  return 0;
}

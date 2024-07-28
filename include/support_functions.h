/**
* Xueling Luo @ Shanghai Jiao Tong University, 2022
* This code is for multiscale phase field fracture.
**/

#ifndef SUPPORT_FUNCTIONS_H
#define SUPPORT_FUNCTIONS_H

#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>

bool contains(const std::string str1, const std::string str2) {
  std::string::size_type idx = str1.find(str2);
  if (idx != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

/**
 * An extension of std::cout to redirect outputs to a log file for pcout.
 * To be honest, it's not a perfect implementation, but it's enough.
 */
class DualOStream {
public:
  DualOStream(ConditionalOStream &stream, const std::string &log_file)
      : pcout(stream), is_active(stream.is_active()) {
    fout.open(log_file);
  }

  template <typename T> const DualOStream &operator<<(const T &t) const {
    pcout << t;
    if (is_active) {
      std::streambuf *oldcout;
      oldcout = std::cout.rdbuf(fout.rdbuf());
      std::cout << t;
      std::cout.rdbuf(oldcout);
    }
    return *this;
  }

  const DualOStream &operator<<(std::ostream &(*p)(std::ostream &)) const {
    pcout << p;
    if (is_active) {
      std::streambuf *oldcout;
      oldcout = std::cout.rdbuf(fout.rdbuf());
      std::cout << p;
      std::cout.rdbuf(oldcout);
    }
    return *this;
  }

  ConditionalOStream pcout;
  std::ofstream fout;

  bool is_active;
};

class DualTimerOutput : public TimerOutput {
public:
  DualTimerOutput(const MPI_Comm &mpi_communicator, ConditionalOStream &stream,
                  const OutputFrequency output_frequency,
                  const OutputType output_type)
      : TimerOutput(mpi_communicator, stream, output_frequency, output_type){};

  // Use the ofstream object from dcout
  void manual_print_summary(const std::ofstream &fout) {
    TimerOutput::print_summary();
    std::streambuf *oldcout;
    oldcout = std::cout.rdbuf(fout.rdbuf());
    TimerOutput::print_summary();
    std::cout.rdbuf(oldcout);
  }
};

inline bool checkFileExsit (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

#endif
// cypha_kernel_tune — small native grid search over xor_kernel_bench hyperparams.
#define CYPHA_KERNEL_TUNE_STANDALONE
#include "xor_kernel_bench.cpp"

int main(int argc, char** argv) {
  std::vector<char*> args;
  args.push_back(argv[0]);
  args.push_back(const_cast<char*>("--tune"));
  for (int i = 1; i < argc; ++i) {
    args.push_back(argv[i]);
  }
  return run_tune(static_cast<int>(args.size()), args.data());
}

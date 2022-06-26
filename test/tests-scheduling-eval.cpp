#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include <fstream>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"

using namespace taco;
const IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
int WARP_SIZE = 32;

void printToCout(IndexStmt stmt) {
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute", false, true);
  codegen->compile(compute, true);
}

void printToFile(string filename, IndexStmt stmt) {
  stringstream source;

  string file_path = "eval_generated/";
  mkdir(file_path.c_str(), 0777);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen->compile(compute, true);

  ofstream source_file;
  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  source_file.open(file_path + filename + file_ending);
  source_file << source.str();
  source_file.close();
}

IndexStmt scheduleSpMMCPU(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSDDMMCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleSpMMGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int NNZ_PER_WARP=8, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), nnz_pre("nnz_pre");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(nnz)}), taco::dense);
  return stmt.reorder({i, j, k})
          .fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(k, dense_val_unbounded, thread, WARP_SIZE)
          .reorder({block, warp, thread, dense_val_unbounded, nnz})
          //.precompute(precomputedExpr, nnz, nnz, precomputed)
          .bound(dense_val_unbounded, dense_val, 4, BoundType::MaxExact)
          //.unroll(dense_val, 4)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleSDDMMGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  return stmt.reorder({i, k, j})
          .fuse(i, k, f)
          .pos(f, fpos, B(i,k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(j, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, CO_FACTOR, BoundType::MaxExact)
          .reorder({block, warp, nnz, thread, dense_val})
          .unroll(dense_val, CO_FACTOR)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::Atomics)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::ParallelReduction);
}

// splits so same number of rows per warp and then each thread in warp gets 1/32 of the columns space
IndexStmt scheduleSpMMRowsGPU(IndexStmt stmt, Tensor<double> A, int ROWS_PER_WARP=4, int BLOCK_SIZE=256) {
  int ROWS_PER_TB = ROWS_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar i1("i1"), block("block"), warp("warp"), warp_row("warp_row"), thread("thread"), thread_col("thread_col");
  return stmt.split(i, block, i1, ROWS_PER_TB)
          .split(i1, warp, warp_row, ROWS_PER_WARP)
          .split(k, thread, thread_col, 32)
          .reorder({block, warp, warp_row, thread, thread_col, j})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

// splits so same number of nonzero rows per warp and then each thread in warp gets 1/32 of the columns space (no search needed)
IndexStmt scheduleSpMMNZRowsGPU(IndexStmt stmt, Tensor<double> A, int NZ_ROWS_PER_WARP=4, int BLOCK_SIZE=256) {
  int NZ_ROWS_PER_TB = NZ_ROWS_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar ip("ip"), ip1("ip1"), block("block"), warp("warp"), warp_row("warp_row"), thread("thread"), thread_col("thread_col");
  return stmt.pos(i, ip, A(i, j))
          .split(ip, block, ip1, NZ_ROWS_PER_TB)
          .split(ip1, warp, warp_row, NZ_ROWS_PER_WARP)
          .split(k, thread, thread_col, 32)
          .reorder({block, warp, warp_row, thread, thread_col, j})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}


IndexStmt scheduleSpMMCPUNoVec(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

TEST(scheduling_eval, spmmCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMCPU(stmt, A);

  //printToFile("spmm_cpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling_eval, sddmmCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_K}, CSR);
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  srand(268238);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMMCPU(stmt, B);

  //printToFile("sddmm_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i,k) = B(i,k) * C(i,j) * D(j,k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, spmmGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));

  srand(434321);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();
  IndexExpr precomputed = A(i, j);
  C(i, k) = B(j, k) * precomputed;

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMGPU(stmt, A, precomputed);

  //printToFile("spmm_gpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling_eval, spmmDCSRGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Sparse, Sparse});
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(25643);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMNZRowsGPU(stmt, A);

  //printToFile("spmm_dcsr_gpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling_eval, sddmmGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_K = 1039/10;
  int NUM_J = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_K}, CSR);
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  srand(535366);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMMGPU(stmt, B);

  //printToFile("sddmm_gpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i,k) = B(i,k) * C(i,j) * D(j,k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}


TEST(generate_evaluation_files, cpu) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  // 4 to 512 and 4, 8, 16
  vector<vector<int>> spmm_dcsr_parameters = {{16, 8}};
  vector<vector<int>> spmm_parameters = {{16,4}};

  vector<vector<int>> sddmm_parameters = {{8, 8}};

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  int NUM_L = 100;
  int NUM_M = 100;
  int NUM_N = 100;

  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  string file_path = "eval_prepared_cpu/";
  mkdir(file_path.c_str(), 0777);

  // spmm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
    Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});
    C(i, k) = A(i, j) * B(j, k);
    IndexStmt stmt = C.getAssignment().concretize();
    for (auto paramSet : spmm_parameters) {
      IndexStmt scheduled = scheduleSpMMCPU(stmt, A, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, "spmm_csr_cpu_taco",  false, true);
      codegen->compile(compute, true);
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_csr_cpu_taco.h");
    source_file << source.str();
    source_file.close();
  }

  // sddmm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_K}, CSR);
    Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
    Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});
    A(i,k) = B(i,k) * C(i,j) * D(j,k);
    IndexStmt stmt = A.getAssignment().concretize();
    for (auto paramSet : sddmm_parameters) {
      IndexStmt scheduled = scheduleSDDMMCPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, "sddmm_csr_cpu_taco",  false, true);
      codegen->compile(compute, true);
    }
    ofstream source_file;
    source_file.open(file_path + "sddmm_csr_cpu_taco.h");
    source_file << source.str();
    source_file.close();
  }

}

TEST(generate_evaluation_files, gpu) {
  vector<vector<int>> spmm_parameters = {}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}
  spmm_parameters.push_back({16,512});

  vector<vector<int>> spmm_dcsr_parameters = {{4, 256, 4}}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}
  vector<vector<int>> sddmm_parameters = {{4*32, 512, 4}}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  int NUM_L = 100;

  string file_ending = ".cu";
  string file_path = "eval_prepared_gpu/";
  mkdir(file_path.c_str(), 0777);

  // spmm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = make_shared<ir::CodeGen_CUDA>(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    for (auto paramSet : spmm_parameters) {
      int NUM_K = 128;
      Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
      Tensor<double> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));
      IndexExpr precomputed = A(i, j);
      C(i, k) = precomputed * B(j, k);
      IndexStmt stmt = C.getAssignment().concretize();
      IndexStmt scheduled = scheduleSpMMGPU(stmt, A, precomputed, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, "spmm_csr_gpu_taco",  false, true);
      codegen->compile(compute, true);
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_csr_gpu_taco.h");
    source_file << source.str();
    source_file.close();
  }

  // sddmm
  {
    stringstream source;

    Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
    for (auto paramSet : sddmm_parameters) {
      int NUM_K = paramSet[2] * WARP_SIZE;
      std::shared_ptr<ir::CodeGen> codegen = make_shared<ir::CodeGen_CUDA>(source, ir::CodeGen::ImplementationGen);
      Tensor<double> A("A", {NUM_I, NUM_K}, CSR);
      Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
      Tensor<double> D("D", {NUM_J, NUM_K}, Format({{Dense, Dense}, {1, 0}}));
      A(i,k) = B(i,k) * C(i,j) * D(j,k);
      IndexStmt stmt = A.getAssignment().concretize();
      IndexStmt scheduled = scheduleSDDMMGPU(stmt, B, paramSet[0], paramSet[1], paramSet[2]);
      ir::Stmt compute = lower(scheduled, "sddmm_csr_gpu_taco",  false, true);
      codegen->compile(compute, true);
    }
    ofstream source_file;
    source_file.open(file_path + "sddmm_csr_gpu_taco.h");
    source_file << source.str();
    source_file.close();
  }
}

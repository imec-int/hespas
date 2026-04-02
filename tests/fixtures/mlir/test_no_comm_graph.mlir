module @test_no_comm_graph attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
    // Operation a: matrix multiplication
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>

    // Operation b: add bias (depends on a)
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %2 = stablehlo.add %0, %1 : tensor<128x512xf32>

    // Operation c: (was all_reduce, now just identity) (depends on a)
    %3 = stablehlo.tanh %0 : tensor<128x512xf32>

    // Operation d: activation function (depends on c)
    %4 = stablehlo.exponential %3 : tensor<128x512xf32>

    // Operation e: normalization (depends on d)
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %6 = stablehlo.divide %4, %5 : tensor<128x512xf32>

    // Operation f: scale operation (depends on b)
    %cst_1 = stablehlo.constant dense<1.500000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128x512xf32>
    %8 = stablehlo.multiply %2, %7 : tensor<128x512xf32>

    // Operation g: final combination (depends on e and f)
    %9 = stablehlo.add %6, %8 : tensor<128x512xf32>

    return %9 : tensor<128x512xf32>
  }
}

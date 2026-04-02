module @test_split_replica_groups attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 4 : i32} {
  func.func public @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
    // Computation: matrix multiplication
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>

    // All-reduce with SPLIT groups: devices [0,1] and [2,3] communicate separately
    %1 = "stablehlo.all_reduce"(%0) <{replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<128x512xf32>) -> tensor<128x512xf32>

    // Activation after the split-group all-reduce
    %2 = stablehlo.tanh %1 : tensor<128x512xf32>

    // All-reduce with SINGLE group: all 4 devices communicate together
    %3 = "stablehlo.all_reduce"(%2) <{replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<128x512xf32>) -> tensor<128x512xf32>

    // Final computation
    %4 = stablehlo.negate %3 : tensor<128x512xf32>

    return %4 : tensor<128x512xf32>
  }
}

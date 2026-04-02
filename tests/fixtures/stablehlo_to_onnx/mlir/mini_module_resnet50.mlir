module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<80x128x28672xf32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.transpose %arg2, dims = [1, 0, 2] {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[128,80,28672]{2,0,1}"} : (tensor<80x128x28672xf32>) -> tensor<128x80x28672xf32>
    %2 = stablehlo.multiply %1, %1 {result_layout = dense<[2, 0, 1]> : tensor<3xindex>, xla_shape = "f32[128,80,28672]{2,0,1}"} : tensor<128x80x28672xf32>
    %3 = stablehlo.reduce(%2 init: %arg3) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<128x80x28672xf32>, tensor<f32>) -> tensor<f32>
    return %3, %0 : tensor<f32>, tensor<f32>
  }
}

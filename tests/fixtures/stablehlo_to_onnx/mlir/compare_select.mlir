module {
  func.func @main(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %0 = stablehlo.compare  EQ, %arg0, %arg1,  FLOAT : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xi1>
    %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %3 = stablehlo.select %0, %1, %2 : tensor<128xi1>, tensor<128xf32>
    return %3 : tensor<128xf32>
  }
}


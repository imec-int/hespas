module {
  func.func @main(%arg0: tensor<256x112x112x64xf32>) -> tensor<96xbf16> {
        %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x112x112x64xf32>, tensor<f32>) -> tensor<64xf32>
        %cst_1 = stablehlo.constant dense<0x4A440000> : tensor<f32>
        %1 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64xf32>
        %2 = stablehlo.divide %0, %1: (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
        %3 = stablehlo.convert %2 : (tensor<64xf32>) -> tensor<64xbf16>
        %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
        %4 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<bf16>) -> tensor<64xbf16>
        %5 = stablehlo.concatenate %3, %4,  dim = 0 : (tensor<64xbf16>, tensor<64xbf16>) -> tensor<128xbf16>
        %6 = stablehlo.slice %5 [0:96] : (tensor<128xbf16>) -> tensor<96xbf16>
        return %6 : tensor<96xbf16>
    }
}


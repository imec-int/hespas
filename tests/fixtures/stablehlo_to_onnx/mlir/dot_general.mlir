module {
  func.func @main(%arg0: tensor<1x4096x2624xbf16>, %arg1: tensor<1x2624x41x128xbf16>) -> tensor<1x4096x41x128xbf16> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x4096x2624xbf16>, tensor<1x2624x41x128xbf16>) -> tensor<1x4096x41x128xbf16>
    %1 = stablehlo.rsqrt %0 : tensor<1x4096x41x128xbf16>
    return %1 : tensor<1x4096x41x128xbf16>
  }
}

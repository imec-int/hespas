module {
  func.func @main(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
    %cst = stablehlo.constant dense<[[1.0, 2.0], [3.5, -4.0]]> : tensor<2x2xbf16>
    %0 = stablehlo.add %arg0, %cst : tensor<2x2xbf16>
    return %0 : tensor<2x2xbf16>
  }
}

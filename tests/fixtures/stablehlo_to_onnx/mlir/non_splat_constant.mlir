module {
  func.func @main(%arg0: tensor<4xf16>) -> tensor<4xf16> {
    %cst = stablehlo.constant dense<[1.0, 2.0, 3.5, -4.0]> : tensor<4xf16>
    %0 = stablehlo.add %arg0, %cst : tensor<4xf16>
    return %0 : tensor<4xf16>
  }
}

module {
  func.func @main(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<bf16>
    %1 = stablehlo.sqrt %0 : tensor<bf16>
    return %1 : tensor<bf16>
  }
}

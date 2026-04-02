module {
  func.func @main() -> tensor<6xf64> {
    // Edge values for fp16-like reduce precision:
    // +Inf, NaN, tiny denormal, 0.0, 65519.0, 65520.0
    %operand = stablehlo.constant dense<[0x7FF0000000000000, 0x7FFFFFFFFFFFFFFF, 0x0000000000000001, 0.0, 65519.0, 65520.0]> : tensor<6xf64>

    %output = "stablehlo.reduce_precision"(%operand) {
      exponent_bits = 5 : i32,
      mantissa_bits = 10 : i32
    } : (tensor<6xf64>) -> tensor<6xf64>

    return %output : tensor<6xf64>
  }
}

func.func @main(%lhs: tensor<1x4x4x1xi64>, %rhs: tensor<3x3x1x1xi64>) -> (tensor<1x2x2x1xi64>) {

  %result = stablehlo.convolution(%lhs, %rhs)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {
      stride = [4, 4],
      lhs_dilate = [2, 2]
    } {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    }
  : (tensor<1x4x4x1xi64>, tensor<3x3x1x1xi64>) -> tensor<1x2x2x1xi64>
  return %result : tensor<1x2x2x1xi64>
}

module {
  func.func @main(%arg0: tensor<1x1x4x4xf32>) -> (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32>) {
    %cst_neg = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %cst_pos = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
    %cst_zero = stablehlo.constant dense<0.0> : tensor<f32>

    %max = "stablehlo.reduce_window"(%arg0, %cst_neg) ({
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %r = stablehlo.maximum %a, %b : tensor<f32>
        stablehlo.return %r : tensor<f32>
    }) {
      window_dimensions = array<i64: 1, 1, 2, 2>,
      window_strides = array<i64: 1, 1, 2, 2>,
      base_dilations = array<i64: 1, 1, 1, 1>,
      window_dilations = array<i64: 1, 1, 1, 1>,
      padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>
    } : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x2x2xf32>

    %min = "stablehlo.reduce_window"(%arg0, %cst_pos) ({
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %r = stablehlo.minimum %a, %b : tensor<f32>
        stablehlo.return %r : tensor<f32>
    }) {
      window_dimensions = array<i64: 1, 1, 2, 2>,
      window_strides = array<i64: 1, 1, 2, 2>,
      base_dilations = array<i64: 1, 1, 1, 1>,
      window_dilations = array<i64: 1, 1, 1, 1>,
      padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>
    } : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x2x2xf32>

    %avg = "stablehlo.reduce_window"(%arg0, %cst_zero) ({
      ^bb0(%a: tensor<f32>, %b: tensor<f32>):
        %r = stablehlo.add %a, %b : tensor<f32>
        stablehlo.return %r : tensor<f32>
    }) {
      window_dimensions = array<i64: 1, 1, 4, 4>,
      window_strides = array<i64: 1, 1, 1, 1>,
      base_dilations = array<i64: 1, 1, 1, 1>,
      window_dilations = array<i64: 1, 1, 1, 1>,
      padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>
    } : (tensor<1x1x4x4xf32>, tensor<f32>) -> tensor<1x1x1x1xf32>

    return %max, %min, %avg : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32>
  }
}

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<2x256xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<256xf32>, %arg7: tensor<f32>, %arg8: tensor<256xf32>, %arg9: tensor<256x56x56x256xbf16>, %arg10: tensor<f32>, %arg11: tensor<f32>, %arg12: tensor<256xf32>, %arg13: tensor<256xf32>, %arg14: tensor<1x1x64x256xf32>, %arg15: tensor<256x56x56x64xbf16>, %arg16: tensor<f32>) -> (tensor<2x256xf32>, tensor<256x56x56x256xf32>, tensor<256x56x56x256xbf16>, tensor<1x1x1x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x256xf32>
    %1 = stablehlo.divide %arg1, %0 : tensor<2x256xf32>
    %2 = stablehlo.slice %1 [0:1, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %3 = stablehlo.reshape %2 : (tensor<1x256xf32>) -> tensor<256xf32>
    %4 = stablehlo.slice %1 [1:2, 0:256] : (tensor<2x256xf32>) -> tensor<1x256xf32>
    %5 = stablehlo.reshape %4 : (tensor<1x256xf32>) -> tensor<256xf32>
    %6 = stablehlo.multiply %3, %3 : tensor<256xf32>
    %7 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %8 = stablehlo.multiply %7, %3 : tensor<256xf32>
    %9 = stablehlo.subtract %5, %6 : tensor<256xf32>
    %10 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %11 = stablehlo.maximum %10, %9 : tensor<256xf32>
    %12 = stablehlo.compare  EQ, %9, %11,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %13 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %14 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %15 = stablehlo.select %12, %13, %14 : tensor<256xi1>, tensor<256xf32>
    %16 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %17 = stablehlo.compare  EQ, %16, %11,  FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
    %18 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %19 = stablehlo.broadcast_in_dim %arg4, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %20 = stablehlo.select %17, %18, %19 : tensor<256xi1>, tensor<256xf32>
    %21 = stablehlo.divide %15, %20 : tensor<256xf32>
    %22 = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %23 = stablehlo.multiply %22, %arg6 : tensor<256xf32>
    %24 = stablehlo.broadcast_in_dim %arg7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %25 = stablehlo.multiply %24, %3 : tensor<256xf32>
    %26 = stablehlo.add %23, %25 : tensor<256xf32>
    %27 = stablehlo.broadcast_in_dim %arg5, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %28 = stablehlo.multiply %27, %arg8 : tensor<256xf32>
    %29 = stablehlo.broadcast_in_dim %arg7, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %30 = stablehlo.multiply %29, %11 : tensor<256xf32>
    %31 = stablehlo.add %28, %30 : tensor<256xf32>
    %32 = stablehlo.broadcast_in_dim %3, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %33 = stablehlo.broadcast_in_dim %11, dims = [3] : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %34 = stablehlo.convert %arg9 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %35 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %36 = stablehlo.subtract %34, %35 : tensor<256x56x56x256xf32>
    %37 = stablehlo.broadcast_in_dim %arg10, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %38 = stablehlo.add %33, %37 : tensor<1x1x1x256xf32>
    %39 = stablehlo.rsqrt %38 : tensor<1x1x1x256xf32>
    %40 = stablehlo.divide %39, %38 : tensor<1x1x1x256xf32>
    %41 = stablehlo.broadcast_in_dim %arg11, dims = [] : (tensor<f32>) -> tensor<1x1x1x256xf32>
    %42 = stablehlo.multiply %41, %40 : tensor<1x1x1x256xf32>
    %43 = stablehlo.reshape %arg12 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %44 = stablehlo.multiply %39, %43 : tensor<1x1x1x256xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %46 = stablehlo.multiply %36, %45 : tensor<256x56x56x256xf32>
    %47 = stablehlo.reshape %arg13 : (tensor<256xf32>) -> tensor<1x1x1x256xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 2, 3] : (tensor<1x1x1x256xf32>) -> tensor<256x56x56x256xf32>
    %49 = stablehlo.add %46, %48 : tensor<256x56x56x256xf32>
    %50 = stablehlo.convert %49 : (tensor<256x56x56x256xf32>) -> tensor<256x56x56x256xbf16>
    %51 = stablehlo.convert %arg14 : (tensor<1x1x64x256xf32>) -> tensor<1x1x64x256xbf16>
    %52 = stablehlo.convolution(%arg15, %51) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<256x56x56x64xbf16>, tensor<1x1x64x256xbf16>) -> tensor<256x56x56x256xbf16>
    %53 = stablehlo.convert %52 : (tensor<256x56x56x256xbf16>) -> tensor<256x56x56x256xf32>
    %54 = stablehlo.multiply %53, %53 : tensor<256x56x56x256xf32>
    %55 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<256x56x56x256xf32>
    %56 = stablehlo.multiply %55, %53 : tensor<256x56x56x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %57 = stablehlo.reduce(%53 init: %cst) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x256xf32>, tensor<f32>) -> tensor<256xf32>
    %58 = stablehlo.broadcast_in_dim %arg16, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %59 = stablehlo.divide %57, %58 : tensor<256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %60 = stablehlo.reduce(%54 init: %cst_0) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<256x56x56x256xf32>, tensor<f32>) -> tensor<256xf32>
    %61 = stablehlo.broadcast_in_dim %arg16, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %62 = stablehlo.divide %60, %61 : tensor<256xf32>
    %63 = stablehlo.broadcast_in_dim %59, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %64 = stablehlo.broadcast_in_dim %62, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %65 = stablehlo.concatenate %63, %64, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<2x256xf32>
    return %65, %56, %50, %42, %31, %26, %21, %8 : tensor<2x256xf32>, tensor<256x56x56x256xf32>, tensor<256x56x56x256xbf16>, tensor<1x1x1x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>
  }
}
